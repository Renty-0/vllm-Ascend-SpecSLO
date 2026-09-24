# SPDX-License-Identifier: Apache-2.0
"""Optional MC2 fused dispatch for PEARL's TP residual path.

The production vLLM compiler pass already emits the AscendC
``matmul_allreduce_add_rmsnorm`` operator.  Native PEARL is intentionally
standalone, so this adapter exposes the same operator with a numerically
equivalent fallback.  Callers can probe support before opting in and retain
the fallback on CANN versions where the custom extension is unavailable.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm_ascend import envs as ascend_envs

_MC2_LOAD_ATTEMPTED = False
_MC2_LOGGED_FUSED_SHAPES: set[tuple[int, int, int, str, str, bool]] = set()
_MC2_LOGGED_FALLBACK_SHAPES: set[tuple[int, int, int, str, str, bool]] = set()
_MC2_LOGGED_EXCEPTION_SHAPES: set[tuple[int, int, int, str, str, bool]] = set()
MC2_FULL_FUSED_OPERATOR = "matmul_allreduce_add_rmsnorm"
MC2_TP3_FULL_FUSED_OPERATOR = "tp3_matmul_allreduce_add_rmsnorm"
MC2_TP3_NATIVE_RMSNORM_OPERATOR = "tp3_matmul_allreduce+native_add_rmsnorm"
MC2_TP3_NATIVE_EPILOGUE_OPERATOR = "tp3_native_matmul+allreduce_add_rmsnorm"
_MC2_FULL_FUSED_OPERATORS = frozenset(
    (
        MC2_FULL_FUSED_OPERATOR,
        MC2_TP3_FULL_FUSED_OPERATOR,
        MC2_TP3_NATIVE_RMSNORM_OPERATOR,
    )
)
_MC2_SUPPORTED_OPERATORS = _MC2_FULL_FUSED_OPERATORS | {
    MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
}
_MC2_DISPATCH_COUNTER_NAMES = (
    "fused_attempt",
    "fused_success",
    "fallback",
    "exception",
    "full_fused_attempt",
    "full_fused_success",
    "native_epilogue_attempt",
    "native_epilogue_success",
    "native_epilogue_chained_attempt",
    "native_epilogue_chained_success",
    "native_epilogue_chain_flush",
)
_MC2_DISPATCH_COUNTERS = {name: 0 for name in _MC2_DISPATCH_COUNTER_NAMES}
_MC2_DISPATCH_COUNTER_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

_MC2_HCCL_DETERMINISTIC = "true"
_MC2_HCCL_OP_EXPANSION_MODE = "AIV"
_MC2_REDUCTION_MODE = "global_01"
_MC2_FULL_EPILOGUE_EPSILON = 1e-6
_MC2_RUNTIME_BINDING_FIELDS = (
    "hccl_deterministic",
    "hccl_op_expansion_mode",
    "reduction_mode",
    "cann_version",
    "hccl_version",
    "vendor_payload_sha256",
    "opapi_symbol_provider_sha256",
    "adapter_binary_sha256",
    "tp_rank_device_mapping",
)
_MC2_REQUIRED_OPAPI_SYMBOLS = (
    "aclnnMatmulAllreduceAddRmsnormGetWorkspaceSize",
    "aclnnMatmulAllreduceAddRmsnorm",
)
_MC2_NATIVE_EPILOGUE_REQUIRED_OPAPI_SYMBOLS = (
    "aclnnAllreduceAddRmsnormGetWorkspaceSize",
    "aclnnAllreduceAddRmsnorm",
)
_MC2_OPTIONAL_OPAPI_SYMBOLS = (
    "InitHugeMemThreadLocal",
    "UnInitHugeMemThreadLocal",
    "ReleaseHugeMem",
)
_MC2_OPAPI_SYMBOLS = _MC2_REQUIRED_OPAPI_SYMBOLS + _MC2_OPTIONAL_OPAPI_SYMBOLS
_MC2_NATIVE_EPILOGUE_OPAPI_SYMBOLS = (
    _MC2_NATIVE_EPILOGUE_REQUIRED_OPAPI_SYMBOLS + _MC2_OPTIONAL_OPAPI_SYMBOLS
)
_MC2_CONTIGUOUS_LAYOUT_FIELDS = (
    "activation_layout",
    "weight_layout",
    "residual_layout",
    "gamma_layout",
)


def _normalize_mc2_operator(operator: str | None) -> str:
    resolved = MC2_FULL_FUSED_OPERATOR if operator is None else str(operator)
    if resolved not in _MC2_SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported MC2 operator identity: {resolved!r}")
    return resolved


def _mc2_required_opapi_symbols(operator: str | None = None) -> tuple[str, ...]:
    resolved = _normalize_mc2_operator(operator)
    if resolved == MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
        return _MC2_NATIVE_EPILOGUE_REQUIRED_OPAPI_SYMBOLS
    return _MC2_REQUIRED_OPAPI_SYMBOLS


def _mc2_opapi_symbols(operator: str | None = None) -> tuple[str, ...]:
    return _mc2_required_opapi_symbols(operator) + _MC2_OPTIONAL_OPAPI_SYMBOLS


def _mc2_package_operator_name(operator: str | None = None) -> str:
    return (
        "allreduce_add_rmsnorm"
        if _normalize_mc2_operator(operator) == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
        else "matmul_allreduce_add_rmsnorm"
    )


@dataclass(frozen=True)
class MC2Capability:
    available: bool
    device: str
    tp_size: int
    operator: str
    reason: str


@dataclass(frozen=True)
class MC2Qualification:
    qualified: bool
    reason: str
    baseline_p95_ms: float | None = None
    fused_p95_ms: float | None = None


_MC2_DISPATCH_TICKET_SEAL = object()
_MC2_STATIC_ROUTE_SEAL = object()


@dataclass(frozen=True)
class _MC2DispatchTicket:
    qualification: MC2Qualification
    signature: tuple[Any, ...]
    seal: object


@dataclass(frozen=True)
class MC2StaticRoute:
    """One immutable, post-weight-load MC2 dispatch decision.

    The route binds a logical decoder projection to its exact weight and
    normalization parameter objects, one already-resolved HCCL communicator,
    and the finite set of profiled row counts that are allowed to enter the
    fused operator.  It is deliberately created before ACLGraph capture so a
    model forward never has to rediscover the communicator, inspect the
    custom-op package, or rerun numerical/latency qualification.
    """

    projection_kind: str
    operator_kind: str
    layer_index: int
    group_tp: str
    tp_rank_size: int
    tp_rank_id: int
    epsilon: float
    is_trans_b: bool
    profile_identity: int
    weight_identity: int
    gamma_identity: int
    input_size: int
    output_size: int
    dtype: str
    weight_format: str
    weight_shape: tuple[int, ...]
    weight_stride: tuple[int, ...]
    gamma_shape: tuple[int, ...]
    gamma_stride: tuple[int, ...]
    activation_layout: str
    weight_layout: str
    residual_layout: str
    gamma_layout: str
    device: str
    decisions: Mapping[int, MC2Qualification]
    enabled: bool
    disabled_reason: str
    seal: object

    def consensus_document(self) -> dict[str, Any]:
        """Return rank-independent route state for target-group consensus."""

        return {
            "projection_kind": self.projection_kind,
            "operator_kind": self.operator_kind,
            "layer_index": self.layer_index,
            "tp_rank_size": self.tp_rank_size,
            # A disabled route can legitimately own rank-dependent shards
            # (for example balanced TP3 FFN partitions).  Those parameters
            # never enter MC2, so consensus covers the shared disable
            # decision rather than rejecting a valid split fallback layout.
            "epsilon": self.epsilon if self.enabled else None,
            "is_trans_b": self.is_trans_b if self.enabled else None,
            "input_size": self.input_size if self.enabled else None,
            "output_size": self.output_size if self.enabled else None,
            "dtype": self.dtype if self.enabled else None,
            "weight_format": self.weight_format if self.enabled else None,
            "weight_shape": self.weight_shape if self.enabled else None,
            "weight_stride": self.weight_stride if self.enabled else None,
            "gamma_shape": self.gamma_shape if self.enabled else None,
            "gamma_stride": self.gamma_stride if self.enabled else None,
            "activation_layout": self.activation_layout if self.enabled else None,
            "weight_layout": self.weight_layout if self.enabled else None,
            "residual_layout": self.residual_layout if self.enabled else None,
            "gamma_layout": self.gamma_layout if self.enabled else None,
            "device_type": torch.device(self.device).type,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "decisions": {
                str(rows): asdict(qualification)
                for rows, qualification in sorted(self.decisions.items())
            },
            # Communicator handles and rank IDs may be process-local.  Their
            # presence/range are checked locally; consensus covers only the
            # fact that every rank resolved a non-empty handle.
            "communicator_resolved": bool(self.group_tp),
        }


@dataclass(frozen=True)
class MC2StaticRouteManifest:
    """Frozen per-model route set agreed by all target TP ranks."""

    routes: tuple[MC2StaticRoute, ...]
    profile_sha256: str
    digest: str


def _mc2_profile_sha256(profile: MC2Profile) -> str:
    document = profile._document()
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _increment_mc2_dispatch_counter(name: str) -> None:
    with _MC2_DISPATCH_COUNTER_LOCK:
        _MC2_DISPATCH_COUNTERS[name] += 1


def snapshot_mc2_dispatch_counters() -> dict[str, int]:
    """Return this process's host-side MC2 dispatch counters.

    ACLGraph replays do not re-enter this Python adapter, and counters are not
    aggregated across tensor-parallel ranks. Workers can therefore inspect
    each process independently without introducing a collective on the model
    execution path.
    """

    with _MC2_DISPATCH_COUNTER_LOCK:
        return dict(_MC2_DISPATCH_COUNTERS)


def reset_mc2_dispatch_counters() -> None:
    """Reset this process's host-side MC2 dispatch counters."""

    with _MC2_DISPATCH_COUNTER_LOCK:
        for name in _MC2_DISPATCH_COUNTER_NAMES:
            _MC2_DISPATCH_COUNTERS[name] = 0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _normalize_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


@lru_cache(maxsize=2)
def _installed_component_version(component: str) -> str | None:
    """Read the CANN component version selected by the current environment."""

    if component not in ("cann", "hccl"):
        raise ValueError(f"unsupported Ascend component {component!r}")
    roots: list[Path] = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if value:
            root = Path(value).resolve()
            if root not in roots:
                roots.append(root)
    default = Path("/usr/local/Ascend/ascend-toolkit/latest")
    if default.exists():
        resolved = default.resolve()
        if resolved not in roots:
            roots.append(resolved)
    header_name = f"{component}_version.h"
    macro_name = f"{component.upper()}_VERSION_STR"
    for root in roots:
        candidates = (
            root / "include" / "version" / header_name,
            root / "aarch64-linux" / "include" / "version" / header_name,
            root / "x86_64-linux" / "include" / "version" / header_name,
        )
        for path in candidates:
            try:
                contents = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            match = re.search(rf'^\s*#define\s+{macro_name}\s+"([^"]+)"', contents, re.MULTILINE)
            if match:
                return match.group(1)
    return None


def _visible_device_mapping() -> tuple[str, ...]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _active_physical_device_id(device: torch.device) -> str | None:
    if device.type != "npu":
        return None
    logical_index = device.index
    if logical_index is None:
        try:
            logical_index = int(torch.npu.current_device())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    visible = _visible_device_mapping()
    if visible:
        if not 0 <= logical_index < len(visible):
            return None
        return visible[logical_index]
    return str(logical_index)


def _mc2_vendor_payload_files(
    vendor_path: Path,
    operator: str | None = None,
) -> tuple[Path, ...]:
    """Return the executable/source payload that identifies one MC2 vendor.

    CANN discovers kernels through ``ASCEND_CUSTOM_OPP_PATH`` while the host
    executor comes from the vendor's shared libraries.  Binding a profile only
    to the repository sources is insufficient for development installs: an
    isolated candidate vendor can be loaded without changing this worktree.
    Keep the digest independent of the installation directory by hashing
    relative paths and contents, but include every MC2 source/kernel artifact
    plus the host libraries that select/launch it.
    """

    vendor = vendor_path.resolve(strict=True)
    package_operator = _mc2_package_operator_name(operator)
    required = (
        vendor / "op_api/lib/libcust_opapi.so",
        vendor / f"op_api/include/aclnnop/aclnn_{package_operator}.h",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"active MC2 vendor payload is incomplete: {missing[0]}")

    candidates: set[Path] = set(required)
    roots_and_patterns = (
        (
            vendor / "op_api/include/aclnnop",
            f"*{package_operator}.h",
        ),
        (
            vendor / "op_impl/ai_core/tbe",
            f"*_impl/ascendc/{package_operator}/*",
        ),
        (
            vendor / "op_impl/ai_core/tbe",
            f"*_impl/dynamic/{package_operator}.py",
        ),
        (vendor / "op_impl/ai_core/tbe/config", "**/aic-*-ops-info.json"),
        (vendor / "op_impl/ai_core/tbe/kernel", f"**/{package_operator}/*"),
        (vendor / "op_impl/ai_core/tbe/kernel/config", f"**/{package_operator}.json"),
        (vendor / "op_impl/ai_core/tbe/kernel/config", "**/binary_info_config.json"),
        (vendor / "op_impl/ai_core/tbe/op_tiling/lib", "**/libcust_opmaster*.so"),
        (vendor / "op_proto/inc", f"*{package_operator}*"),
        (vendor / "op_proto/lib", "**/libcust_opsproto*.so"),
    )
    for root, pattern in roots_and_patterns:
        if root.is_dir():
            candidates.update(path for path in root.glob(pattern) if path.is_file())
    objects = [path for path in candidates if path.suffix == ".o"]
    manifests = [
        path
        for path in candidates
        if path.name == f"{package_operator}.json"
    ]
    if not objects or not manifests:
        raise RuntimeError("active MC2 vendor payload lacks compiled kernels or a manifest")
    return tuple(sorted(candidates, key=lambda path: str(path.relative_to(vendor))))


@lru_cache(maxsize=32)
def _mc2_vendor_payload_sha256_for_path(
    vendor_path: str,
    operator: str | None = None,
) -> str:
    vendor = Path(vendor_path).resolve(strict=True)
    digest = hashlib.sha256()
    for path in _mc2_vendor_payload_files(vendor, operator):
        digest.update(str(path.relative_to(vendor)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _resolve_mc2_symbol_vendor(
    search_path: str,
    operator: str | None = None,
) -> str:
    """Mirror the C++ custom-path search for both required MC2 symbols.

    The C++ adapter treats every ASCEND_CUSTOM_OPP_PATH entry as a vendor
    root and appends ``op_api/lib`` directly.  Do the same here instead of
    accepting package-root aliases that the actual loader cannot use.  Each
    symbol is resolved independently by the C++ macro; reject a mixed-DSO
    result rather than hashing a vendor different from the one executed.
    """

    symbol_libraries: dict[str, Path] = {}
    for symbol in _mc2_required_opapi_symbols(operator):
        for entry in (item for item in search_path.split(os.pathsep) if item):
            library = Path(entry).expanduser() / "op_api/lib/libcust_opapi.so"
            try:
                library = library.resolve(strict=True)
                handle = ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_LAZY)
                getattr(handle, symbol)
            except (AttributeError, OSError):
                continue
            symbol_libraries[symbol] = library
            break
        if symbol not in symbol_libraries:
            raise RuntimeError(f"cannot resolve active MC2 OPAPI symbol {symbol}")
    unique_libraries = set(symbol_libraries.values())
    if len(unique_libraries) != 1:
        details = ", ".join(f"{name}={path}" for name, path in symbol_libraries.items())
        raise RuntimeError(f"MC2 OPAPI symbols resolve from mixed DSOs: {details}")
    library = next(iter(unique_libraries))
    vendor = library.parents[2]
    return str(vendor)


def active_mc2_vendor_payload_sha256(operator: str | None = None) -> str:
    """Hash the exact symbol-providing vendor payload used by MC2 dispatch."""

    required_symbols = _mc2_required_opapi_symbols(operator)
    providers = dict(_freeze_and_get_mc2_opapi_symbol_providers(operator))
    library = Path(providers[required_symbols[0]])
    vendor_path = library.parents[2]
    return _mc2_vendor_payload_sha256_for_path(vendor_path, operator)


@lru_cache(maxsize=8)
def _sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _freeze_and_get_mc2_opapi_symbol_providers(
    operator: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Freeze the native OPAPI cache and report each provider via ``dladdr``."""

    resolved_operator = _normalize_mc2_operator(operator)
    expected_symbols = _mc2_opapi_symbols(resolved_operator)
    query_name = (
        "freeze_and_get_allreduce_add_rmsnorm_opapi_symbol_providers"
        if resolved_operator == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
        else "freeze_and_get_mc2_opapi_symbol_providers"
    )
    try:
        native_query = getattr(torch.ops._C_ascend, query_name)
        rows = native_query()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("loaded MC2 adapter cannot report its frozen OPAPI symbol providers") from error
    if not isinstance(rows, (list, tuple)):
        raise RuntimeError("MC2 adapter returned an invalid OPAPI provider record")
    providers: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, str) or "=" not in row:
            raise RuntimeError("MC2 adapter returned a malformed OPAPI provider record")
        symbol, path = row.split("=", 1)
        if symbol not in expected_symbols or symbol in providers:
            raise RuntimeError(f"MC2 adapter returned an unexpected OPAPI provider symbol: {symbol!r}")
        if path:
            try:
                path = str(Path(path).resolve(strict=True))
            except OSError as error:
                raise RuntimeError(f"MC2 OPAPI provider is unavailable for {symbol}: {path}") from error
        providers[symbol] = path
    if set(providers) != set(expected_symbols):
        missing = sorted(set(expected_symbols).difference(providers))
        raise RuntimeError(f"MC2 adapter omitted OPAPI provider symbols: {', '.join(missing)}")
    required_symbols = _mc2_required_opapi_symbols(resolved_operator)
    required_paths = {providers[symbol] for symbol in required_symbols}
    if "" in required_paths:
        raise RuntimeError("MC2 workspace or execute OPAPI symbol is unavailable")
    if len(required_paths) != 1:
        details = ", ".join(f"{name}={providers[name]}" for name in required_symbols)
        raise RuntimeError(f"MC2 workspace and execute symbols resolve from mixed DSOs: {details}")
    return tuple((symbol, providers[symbol]) for symbol in expected_symbols)


def active_mc2_opapi_symbol_provider_sha256(
    operator: str | None = None,
) -> dict[str, str | None]:
    """Hash every DSO that owns a frozen MC2 adapter symbol."""

    return {
        symbol: _sha256_file(path) if path else None
        for symbol, path in _freeze_and_get_mc2_opapi_symbol_providers(operator)
    }


def active_mc2_adapter_binary_sha256() -> str:
    """Hash the loaded Torch extension that owns the MC2 OPAPI adapter."""

    try:
        import vllm_ascend.vllm_ascend_C as extension
    except ImportError as error:
        raise RuntimeError("vllm_ascend_C is not loaded for MC2 qualification") from error
    module_path = getattr(extension, "__file__", None)
    if not module_path:
        raise RuntimeError("cannot identify the loaded vllm_ascend_C binary")
    resolved = Path(module_path).resolve(strict=True)
    return _sha256_file(str(resolved))


@lru_cache(maxsize=32)
def _resolve_active_mc2_package_cached(
    search_path: str,
    operator: str | None = None,
) -> Any:
    """Resolve capability once for an immutable worker custom-op path."""

    from vllm_ascend.custom_op_package import (
        CUSTOM_OPP_ENV,
        resolve_active_allreduce_add_rmsnorm_package,
        resolve_active_matmul_allreduce_add_rmsnorm_package,
    )

    resolver = (
        resolve_active_allreduce_add_rmsnorm_package
        if _normalize_mc2_operator(operator) == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
        else resolve_active_matmul_allreduce_add_rmsnorm_package
    )
    return resolver(
        environ={CUSTOM_OPP_ENV: search_path},
    )


def current_mc2_runtime_binding(
    *,
    tp_rank_device_mapping: Sequence[str | int],
    operator: str | None = None,
) -> dict[str, Any]:
    """Build the environment identity required by a production MC2 profile.

    The rank mapping is deliberately supplied by the qualification runner:
    the process-local logical NPU index is not sufficient to reconstruct the
    order of a tensor-parallel subgroup embedded in a larger visible set.
    """

    if os.environ.get("HCCL_DETERMINISTIC", "").strip().lower() != _MC2_HCCL_DETERMINISTIC:
        raise RuntimeError("MC2 qualification requires HCCL_DETERMINISTIC=true")
    if os.environ.get("HCCL_OP_EXPANSION_MODE", "").strip() != _MC2_HCCL_OP_EXPANSION_MODE:
        raise RuntimeError("MC2 qualification requires HCCL_OP_EXPANSION_MODE=AIV")
    normalized_mapping = [str(item).strip() for item in tp_rank_device_mapping]
    if any(not item for item in normalized_mapping) or len(set(normalized_mapping)) != len(normalized_mapping):
        raise ValueError("MC2 qualification requires unique non-empty TP rank device IDs")
    cann_version = _installed_component_version("cann")
    hccl_version = _installed_component_version("hccl")
    if cann_version is None or hccl_version is None:
        raise RuntimeError("cannot identify the active CANN/HCCL installation")
    resolved_operator = _normalize_mc2_operator(operator)
    if resolved_operator == MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
        vendor_payload_sha256 = active_mc2_vendor_payload_sha256(resolved_operator)
        provider_sha256 = active_mc2_opapi_symbol_provider_sha256(resolved_operator)
    else:
        # Preserve the long-standing no-argument hook used by external
        # qualification harnesses and tests for the legacy full-fused op.
        vendor_payload_sha256 = active_mc2_vendor_payload_sha256()
        provider_sha256 = active_mc2_opapi_symbol_provider_sha256()
    return {
        "hccl_deterministic": _MC2_HCCL_DETERMINISTIC,
        "hccl_op_expansion_mode": _MC2_HCCL_OP_EXPANSION_MODE,
        "reduction_mode": _MC2_REDUCTION_MODE,
        "cann_version": cann_version,
        "hccl_version": hccl_version,
        "vendor_payload_sha256": vendor_payload_sha256,
        "opapi_symbol_provider_sha256": provider_sha256,
        "adapter_binary_sha256": active_mc2_adapter_binary_sha256(),
        "tp_rank_device_mapping": normalized_mapping,
    }


def _runtime_binding_mismatch(
    metadata: Mapping[str, Any],
    device: torch.device,
    *,
    tp_rank_id: int | None,
) -> str | None:
    try:
        operator = _normalize_mc2_operator(metadata.get("operator"))
    except ValueError as error:
        return str(error)
    binding = metadata.get("runtime_binding")
    if not isinstance(binding, Mapping):
        return "legacy MC2 profile lacks a production runtime binding"
    if any(name not in binding for name in _MC2_RUNTIME_BINDING_FIELDS):
        return "MC2 profile runtime binding is incomplete"
    if str(binding.get("hccl_deterministic", "")).strip().lower() != _MC2_HCCL_DETERMINISTIC:
        return "MC2 profile deterministic collective mode is unsupported"
    if binding.get("hccl_op_expansion_mode") != _MC2_HCCL_OP_EXPANSION_MODE:
        return "MC2 profile HCCL expansion mode is unsupported"
    if os.environ.get("HCCL_DETERMINISTIC", "").strip().lower() != _MC2_HCCL_DETERMINISTIC:
        return "HCCL_DETERMINISTIC=true is required by the MC2 profile"
    if os.environ.get("HCCL_OP_EXPANSION_MODE", "").strip() != _MC2_HCCL_OP_EXPANSION_MODE:
        return "HCCL_OP_EXPANSION_MODE=AIV is required by the MC2 profile"
    if binding["reduction_mode"] != _MC2_REDUCTION_MODE:
        return "MC2 profile reduction mode is not supported by this adapter"
    for component in ("cann", "hccl"):
        actual_version = _installed_component_version(component)
        if actual_version is None:
            return f"cannot identify the active {component.upper()} version"
        if actual_version != binding[f"{component}_version"]:
            return f"MC2 profile {component.upper()} version does not match execution"
    expected_vendor_sha = binding["vendor_payload_sha256"]
    try:
        actual_vendor_sha = (
            active_mc2_vendor_payload_sha256(operator)
            if operator == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
            else active_mc2_vendor_payload_sha256()
        )
    except (OSError, RuntimeError, ValueError) as error:
        return f"cannot identify the active MC2 vendor payload: {error}"
    if actual_vendor_sha != expected_vendor_sha:
        return "MC2 profile vendor payload does not match execution"
    expected_provider_sha = binding["opapi_symbol_provider_sha256"]
    try:
        actual_provider_sha = (
            active_mc2_opapi_symbol_provider_sha256(operator)
            if operator == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
            else active_mc2_opapi_symbol_provider_sha256()
        )
    except (OSError, RuntimeError, ValueError) as error:
        return f"cannot identify the frozen MC2 OPAPI symbol providers: {error}"
    if actual_provider_sha != expected_provider_sha:
        return "MC2 profile OPAPI symbol providers do not match execution"
    expected_adapter_sha = binding["adapter_binary_sha256"]
    try:
        actual_adapter_sha = active_mc2_adapter_binary_sha256()
    except (OSError, RuntimeError, ValueError) as error:
        return f"cannot identify the active MC2 adapter binary: {error}"
    if actual_adapter_sha != expected_adapter_sha:
        return "MC2 profile adapter binary does not match execution"
    active_device = _active_physical_device_id(device)
    if active_device is None:
        return "cannot resolve the active physical NPU for the MC2 profile"
    raw_rank_mapping = binding["tp_rank_device_mapping"]
    if isinstance(raw_rank_mapping, (str, bytes)) or not isinstance(raw_rank_mapping, Sequence):
        return "MC2 profile TP rank device mapping is invalid"
    rank_mapping = tuple(str(item) for item in raw_rank_mapping)
    if tp_rank_id is None:
        if active_device not in rank_mapping:
            return "active physical NPU is absent from the MC2 profile TP mapping"
    elif not 0 <= tp_rank_id < len(rank_mapping):
        return "MC2 tensor-parallel rank is outside the profiled device mapping"
    elif active_device != rank_mapping[tp_rank_id]:
        return "MC2 profile TP rank-to-device mapping does not match execution"
    return None


def validate_mc2_runtime_binding(
    profile: MC2Profile,
    device: torch.device | str,
    *,
    tp_rank_id: int,
) -> str | None:
    """Freeze and validate one worker's runtime before any graph capture."""

    return _runtime_binding_mismatch(
        profile.metadata,
        torch.device(device),
        tp_rank_id=tp_rank_id,
    )


def mc2_source_sha256() -> str:
    """Hash the Python adapter and compiled MC2 sources used by this tree."""

    adapter = Path(__file__).resolve()
    repository = adapter.parents[3]
    sources = [adapter]
    # Profile admission also depends on how the model orders chained
    # epilogues and chooses the final flush.  Hash that production wiring so
    # changing the graph dependency shape cannot silently reuse stale device
    # qualification produced for an older model-runner.
    model_adapter = adapter.with_name("native_model.py")
    if model_adapter.is_file():
        sources.append(model_adapter)
    source_root = repository / "csrc" / "mc2"
    if source_root.is_dir():
        sources.extend(sorted(path for path in source_root.rglob("*") if path.is_file()))
    # The adapter's DSO search order and per-symbol static caching are part of
    # the MC2 execution contract even though they live outside csrc/mc2.
    loader_source = repository / "csrc/aclnn_torch_adapter/op_api_common.h"
    if loader_source.is_file():
        sources.append(loader_source)
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(repository) if path.is_relative_to(repository) else path.name).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _freeze_profile_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_profile_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_profile_value(item) for item in value)
    return value


def _thaw_profile_value(value: Any) -> Any:
    """Return JSON-shaped mutable containers for strict revalidation."""

    if isinstance(value, Mapping):
        return {key: _thaw_profile_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_profile_value(item) for item in value]
    if isinstance(value, list):
        return [_thaw_profile_value(item) for item in value]
    return value


def _weight_format(tensor: torch.Tensor) -> str:
    if tensor.device.type != "npu":
        return "ND"
    try:
        import torch_npu

        value = torch_npu.get_npu_format(tensor)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return "unknown"
    if value is None:
        return "unknown"
    normalized = str(value).strip()
    # Newer torch-npu wraps the raw format code in ``Format`` and renders -1
    # as ``UNDEFINED``; older releases may return the integer directly.  None,
    # empty/sentinel names and the raw -1 code must all fail closed rather than
    # accidentally becoming profile keys that can authorize fused dispatch.
    if normalized.upper() in {"", "NONE", "UNDEFINED", "UNKNOWN"} or normalized == "-1":
        return "unknown"
    return normalized


def _native_epilogue_static_shape_error(
    key: tuple[int, int, int, str, str, bool],
    *,
    tp_size: int,
    epsilon: float,
) -> str | None:
    """Return why a profile row cannot enter the TP3 v126 epilogue."""

    rows, _input_size, output_size, dtype, _weight_format_name, is_trans_b = key
    if int(tp_size) != 3:
        return "native-MatMul MC2 epilogue requires tensor parallel size 3"
    if not 0 < int(rows) <= 160:
        return "native-MatMul MC2 epilogue requires 1 <= M <= 160"
    if int(output_size) != 5120:
        return "native-MatMul MC2 epilogue requires hidden size 5120"
    if dtype != "bfloat16":
        return "native-MatMul MC2 epilogue requires bfloat16 tensors"
    if not bool(is_trans_b):
        return "native-MatMul MC2 epilogue requires an F.linear/transposed weight"
    if float(epsilon) != _MC2_FULL_EPILOGUE_EPSILON:
        return "native-MatMul MC2 epilogue requires epsilon=1e-6"
    return None


def _is_nd_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor has the ND storage format required by v126."""

    return _weight_format(tensor).upper() in {"ND", "2"}


def _native_epilogue_dispatch_error(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    *,
    tp_rank_size: int,
    epsilon: float,
    is_trans_b: bool,
    is_gather_add_out: bool,
) -> str | None:
    output_size = int(weight.shape[0] if is_trans_b else weight.shape[-1])
    key = (
        int(math.prod(x.shape[:-1])),
        int(x.shape[-1]),
        output_size,
        _dtype_name(x),
        _weight_format(weight),
        bool(is_trans_b),
    )
    error = _native_epilogue_static_shape_error(
        key,
        tp_size=tp_rank_size,
        epsilon=epsilon,
    )
    if error is not None:
        return error
    if not is_gather_add_out:
        return "native-MatMul MC2 epilogue requires the residual add_out result"
    tensors = (x, weight, residual, gamma)
    if any(tensor.device.type != "npu" for tensor in tensors):
        return "native-MatMul MC2 epilogue requires Ascend NPU tensors"
    if len({str(tensor.device) for tensor in tensors}) != 1:
        return "native-MatMul MC2 epilogue tensors must share one NPU device"
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        return "native-MatMul MC2 epilogue requires bfloat16 tensors"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "native-MatMul MC2 epilogue requires contiguous tensors"
    if not _is_nd_tensor(residual) or not _is_nd_tensor(gamma):
        return "native-MatMul MC2 epilogue requires ND residual and gamma"
    return None


def _mc2_tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(tensor),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _mc2_dispatch_signature(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    *,
    tp_rank_size: int,
    tp_rank_id: int,
    epsilon: float,
    is_trans_b: bool,
    profile: MC2Profile | None,
) -> tuple[Any, ...]:
    return (
        id(profile),
        _mc2_tensor_signature(x),
        _mc2_tensor_signature(weight),
        _mc2_tensor_signature(residual),
        _mc2_tensor_signature(gamma),
        int(tp_rank_size),
        int(tp_rank_id),
        float(epsilon),
        bool(is_trans_b),
    )


def _bind_mc2_dispatch_ticket(
    qualification: MC2Qualification,
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    *,
    tp_rank_size: int,
    tp_rank_id: int,
    epsilon: float,
    is_trans_b: bool,
    profile: MC2Profile | None,
) -> _MC2DispatchTicket:
    return _MC2DispatchTicket(
        qualification=qualification,
        signature=_mc2_dispatch_signature(
            x,
            weight,
            residual,
            gamma,
            tp_rank_size=tp_rank_size,
            tp_rank_id=tp_rank_id,
            epsilon=epsilon,
            is_trans_b=is_trans_b,
            profile=profile,
        ),
        seal=_MC2_DISPATCH_TICKET_SEAL,
    )


@dataclass(frozen=True)
class MC2Profile:
    """Identity-bound numerical and latency qualification for exact MC2 shapes."""

    metadata: Mapping[str, Any]
    entries: Mapping[tuple[int, int, int, str, str, bool], Mapping[str, Any]]
    latency_percentile: float
    minimum_samples: int
    minimum_speedup: float

    def _document(self) -> dict[str, Any]:
        """Serialize through the public schema, never through mappingproxy."""

        return {
            "schema_version": 1,
            "metadata": _thaw_profile_value(self.metadata),
            "latency_percentile": self.latency_percentile,
            "minimum_samples": self.minimum_samples,
            "minimum_speedup": self.minimum_speedup,
            "measurements": [_thaw_profile_value(row) for row in self.entries.values()],
        }

    def __reduce__(self):
        # PEARLEngine passes NativePearlConfig through multiprocessing spawn.
        # mappingproxy is intentionally immutable but not picklable, so rebuild
        # from the strict public document and rerun every identity/numeric gate
        # in the child process.
        return (normalize_mc2_profile, (self._document(),))

    def qualify(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        *,
        tp_size: int,
        is_trans_b: bool,
        tp_rank_id: int | None = None,
        epsilon: float = _MC2_FULL_EPILOGUE_EPSILON,
    ) -> MC2Qualification:
        if not x.is_contiguous() or not weight.is_contiguous() or not residual.is_contiguous():
            return MC2Qualification(
                False,
                "MC2 qualification requires contiguous activation, weight, and residual tensors",
            )
        runtime_mismatch = _runtime_binding_mismatch(
            self.metadata,
            x.device,
            tp_rank_id=tp_rank_id,
        )
        if runtime_mismatch is not None:
            return MC2Qualification(False, runtime_mismatch)
        try:
            actual_hardware = str(torch.npu.get_device_name(x.device.index))
        except (AttributeError, RuntimeError, TypeError):
            return MC2Qualification(False, "cannot identify the active Ascend device")
        if _normalize_identity(actual_hardware) != _normalize_identity(str(self.metadata["hardware"])):
            return MC2Qualification(False, "MC2 profile hardware does not match the active device")
        profiled_epsilon = self.metadata.get("rms_norm_epsilon")
        if profiled_epsilon is None:
            return MC2Qualification(False, "legacy MC2 profile lacks an RMSNorm epsilon binding")
        if float(epsilon) != float(profiled_epsilon):
            return MC2Qualification(False, "MC2 profile RMSNorm epsilon does not match execution")
        m = math.prod(x.shape[:-1])
        k = int(x.shape[-1])
        n = int(residual.shape[-1])
        weight_format = _weight_format(weight)
        if weight_format == "unknown":
            return MC2Qualification(False, "cannot identify the MC2 projection weight format")
        key = (int(m), k, n, _dtype_name(x), weight_format, bool(is_trans_b))
        return self.qualify_static_shape(
            key,
            tp_size=tp_size,
            epsilon=epsilon,
        )

    def qualify_static_shape(
        self,
        key: tuple[int, int, int, str, str, bool],
        *,
        tp_size: int,
        epsilon: float,
    ) -> MC2Qualification:
        """Evaluate immutable profile gates without consulting live tensors.

        Runtime binding, hardware identity, and exact tensor ownership are
        validated once by worker admission/static-route construction.  This
        helper evaluates the remaining profile row and is therefore safe to
        cache in a post-weight-load route manifest.
        """

        profiled_epsilon = self.metadata.get("rms_norm_epsilon")
        if profiled_epsilon is None:
            return MC2Qualification(False, "legacy MC2 profile lacks an RMSNorm epsilon binding")
        if float(epsilon) != float(profiled_epsilon):
            return MC2Qualification(False, "MC2 profile RMSNorm epsilon does not match execution")
        if int(self.metadata["tensor_parallel_size"]) != int(tp_size):
            return MC2Qualification(False, "MC2 profile tensor parallel size does not match execution")
        if self.metadata.get("operator") == MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
            contract_error = _native_epilogue_static_shape_error(
                key,
                tp_size=tp_size,
                epsilon=epsilon,
            )
            if contract_error is not None:
                return MC2Qualification(False, contract_error)
        row = self.entries.get(key)
        if row is None:
            return MC2Qualification(False, f"unprofiled MC2 shape {key!r}")
        if any(row.get(name) != "contiguous" for name in _MC2_CONTIGUOUS_LAYOUT_FIELDS):
            return MC2Qualification(False, "MC2 profile row is not bound to contiguous tensor layouts")
        # A captured operator benchmark that only replays its original input
        # cannot prove that ACLGraph reads the current activation buffer.  The
        # first replay in ``measure_specslo_mc2`` intentionally uses the
        # capture input, so at least two replays and a non-zero mutation are
        # required before a row may drive production graph dispatch.
        changed_input_replays = row.get("changed_input_replays")
        changed_input_delta = row.get("changed_input_delta")
        if (
            isinstance(changed_input_replays, bool)
            or not isinstance(changed_input_replays, int)
            or changed_input_replays < 2
            or isinstance(changed_input_delta, bool)
            or not isinstance(changed_input_delta, (int, float))
            or not math.isfinite(float(changed_input_delta))
            or float(changed_input_delta) == 0.0
        ):
            return MC2Qualification(False, "MC2 profile lacks changed-input ACLGraph qualification")
        baseline = _percentile(row["baseline_latency_ms"], self.latency_percentile)
        fused = _percentile(row["fused_latency_ms"], self.latency_percentile)
        if "max_scaled_norm" in row:
            numerical = float(row["max_scaled_norm"]) <= 1.0 and float(row["max_scaled_added"]) <= 1.0
        else:
            numerical = float(row["max_abs_norm"]) <= float(row["norm_atol"]) and float(row["max_abs_added"]) <= float(
                row["added_atol"]
            )
        faster = fused * self.minimum_speedup <= baseline
        if not numerical:
            return MC2Qualification(False, "MC2 full-output numerical tolerance failed", baseline, fused)
        if not faster:
            diagnostic_force_native = (
                self.metadata.get("operator") == "tp3_matmul_allreduce+native_add_rmsnorm"
                and ascend_envs.VLLM_ASCEND_PEARL_MC2_DIAGNOSTIC_FORCE_NATIVE_EPILOGUE
            )
            if diagnostic_force_native:
                return MC2Qualification(
                    True,
                    "diagnostic-only native epilogue latency-gate override; "
                    "exact shape passed numerical gate",
                    baseline,
                    fused,
                )
            return MC2Qualification(False, "MC2 repeated p95 latency is not faster than fallback", baseline, fused)
        return MC2Qualification(True, "exact shape passed numerical and repeated p95 gates", baseline, fused)


def normalize_mc2_profile(value: MC2Profile | Mapping[str, Any] | str | Path) -> MC2Profile:
    """Load a strict profile; handwritten enable flags never qualify MC2."""

    # Revalidate public MC2Profile instances instead of trusting a directly
    # constructed or subsequently mutated mapping. NativePearlConfig reruns
    # normalization through dataclasses.replace(); the final mappings are
    # recursively frozen below so later mutation cannot bypass the gates.
    if isinstance(value, MC2Profile):
        value = value._document()
    if isinstance(value, (str, Path)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read MC2 qualification profile: {error}") from error
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("MC2 qualification requires schema_version=1")
    metadata = value.get("metadata")
    measurements = value.get("measurements")
    if not isinstance(metadata, Mapping) or not isinstance(measurements, list) or not measurements:
        raise ValueError("MC2 qualification requires metadata and non-empty measurements")
    operator_kind = metadata.get("operator")
    if operator_kind not in _MC2_SUPPORTED_OPERATORS:
        raise ValueError("MC2 profile operator identity is invalid")
    if not isinstance(metadata.get("hardware"), str) or not metadata["hardware"].strip():
        raise ValueError("MC2 profile requires a hardware identity")
    tp_size = metadata.get("tensor_parallel_size")
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 2:
        raise ValueError("MC2 profile tensor_parallel_size must describe a TP group")
    if metadata.get("source_sha256") != mc2_source_sha256():
        raise ValueError("MC2 profile source hash does not match the current adapter/kernel sources")
    normalized_metadata = dict(metadata)
    profiled_epsilon = metadata.get("rms_norm_epsilon")
    # Older schema-v1 documents remain readable for offline reports, but
    # ``qualify`` rejects them.  A newly supplied binding must be a finite,
    # positive scalar and is compared exactly with the model configuration.
    if profiled_epsilon is not None:
        if (
            isinstance(profiled_epsilon, bool)
            or not isinstance(profiled_epsilon, (int, float))
            or not math.isfinite(float(profiled_epsilon))
            or float(profiled_epsilon) <= 0.0
        ):
            raise ValueError("MC2 profile rms_norm_epsilon must be finite and positive")
        normalized_metadata["rms_norm_epsilon"] = float(profiled_epsilon)
    runtime_binding = metadata.get("runtime_binding")
    # Schema-v1 profiles produced before runtime binding was introduced remain
    # readable for reports and diagnostics, but ``qualify`` rejects them.  A
    # partially specified new binding is more dangerous than a clearly legacy
    # profile, so reject it while loading.
    if runtime_binding is not None:
        if not isinstance(runtime_binding, Mapping):
            raise ValueError("MC2 profile runtime_binding must be an object")
        missing = [name for name in _MC2_RUNTIME_BINDING_FIELDS if name not in runtime_binding]
        if missing:
            raise ValueError(f"MC2 profile runtime_binding is missing {', '.join(missing)}")
        if str(runtime_binding["hccl_deterministic"]).strip().lower() != _MC2_HCCL_DETERMINISTIC:
            raise ValueError("MC2 profile requires hccl_deterministic=true")
        if runtime_binding["hccl_op_expansion_mode"] != _MC2_HCCL_OP_EXPANSION_MODE:
            raise ValueError("MC2 profile requires hccl_op_expansion_mode=AIV")
        if runtime_binding["reduction_mode"] != _MC2_REDUCTION_MODE:
            raise ValueError("MC2 profile requires reduction_mode=global_01")
        for name in ("cann_version", "hccl_version"):
            if not isinstance(runtime_binding[name], str) or not runtime_binding[name].strip():
                raise ValueError(f"MC2 profile runtime_binding {name} must be non-empty")
        vendor_payload_sha256 = runtime_binding["vendor_payload_sha256"]
        if (
            not isinstance(vendor_payload_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", vendor_payload_sha256)
        ):
            raise ValueError("MC2 profile runtime_binding vendor_payload_sha256 must be a lowercase SHA256")
        provider_symbols = _mc2_opapi_symbols(str(operator_kind))
        optional_symbols = set(_MC2_OPTIONAL_OPAPI_SYMBOLS)
        provider_sha256 = runtime_binding["opapi_symbol_provider_sha256"]
        if not isinstance(provider_sha256, Mapping) or set(provider_sha256) != set(provider_symbols):
            raise ValueError("MC2 profile runtime_binding OPAPI provider identity is incomplete")
        normalized_provider_sha256: dict[str, str | None] = {}
        for symbol in provider_symbols:
            identity = provider_sha256[symbol]
            if identity is None and symbol in optional_symbols:
                normalized_provider_sha256[symbol] = None
            elif isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity):
                normalized_provider_sha256[symbol] = identity
            else:
                raise ValueError(f"MC2 profile runtime_binding OPAPI provider SHA256 is invalid for {symbol}")
        adapter_binary_sha256 = runtime_binding["adapter_binary_sha256"]
        if (
            not isinstance(adapter_binary_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", adapter_binary_sha256)
        ):
            raise ValueError("MC2 profile runtime_binding adapter_binary_sha256 must be a lowercase SHA256")
        rank_mapping = runtime_binding["tp_rank_device_mapping"]
        if (
            isinstance(rank_mapping, (str, bytes))
            or not isinstance(rank_mapping, Sequence)
            or len(rank_mapping) != tp_size
        ):
            raise ValueError("MC2 profile TP rank device mapping must have one entry per TP rank")
        normalized_mapping = tuple(str(item).strip() for item in rank_mapping)
        if any(not item for item in normalized_mapping) or len(set(normalized_mapping)) != tp_size:
            raise ValueError("MC2 profile TP rank device mapping must contain unique non-empty device IDs")
        normalized_metadata["runtime_binding"] = {
            "hccl_deterministic": _MC2_HCCL_DETERMINISTIC,
            "hccl_op_expansion_mode": _MC2_HCCL_OP_EXPANSION_MODE,
            "reduction_mode": _MC2_REDUCTION_MODE,
            "cann_version": runtime_binding["cann_version"].strip(),
            "hccl_version": runtime_binding["hccl_version"].strip(),
            "vendor_payload_sha256": vendor_payload_sha256,
            "opapi_symbol_provider_sha256": normalized_provider_sha256,
            "adapter_binary_sha256": adapter_binary_sha256,
            "tp_rank_device_mapping": normalized_mapping,
        }
    latency_percentile = float(value.get("latency_percentile", 95.0))
    minimum_samples = int(value.get("minimum_samples", 5))
    minimum_speedup = float(value.get("minimum_speedup", 1.02))
    if not 0.0 <= latency_percentile <= 100.0 or minimum_samples < 3 or minimum_speedup <= 1.0:
        raise ValueError("MC2 profile gates require percentile [0,100], >=3 samples, and speedup >1")
    entries: dict[tuple[int, int, int, str, str, bool], Mapping[str, Any]] = {}
    for row in measurements:
        if not isinstance(row, Mapping):
            raise ValueError("MC2 measurement rows must be objects")
        integers = []
        for name in ("m", "k", "n"):
            item = row.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"MC2 measurement {name} must be a positive integer")
            integers.append(item)
        dtype = row.get("dtype")
        weight_format = row.get("weight_format")
        if dtype not in ("bfloat16", "float16") or not isinstance(weight_format, str) or not weight_format:
            raise ValueError("MC2 measurement dtype/weight_format is invalid")
        invalid_layouts = [
            name for name in _MC2_CONTIGUOUS_LAYOUT_FIELDS if row.get(name) != "contiguous"
        ]
        if invalid_layouts:
            raise ValueError(
                "MC2 measurement must bind contiguous activation/weight/residual/gamma layouts: "
                + ", ".join(invalid_layouts)
            )
        timings = []
        for name in ("baseline_latency_ms", "fused_latency_ms"):
            samples = row.get(name)
            if not isinstance(samples, list) or len(samples) < minimum_samples:
                raise ValueError(f"MC2 measurement {name} needs at least {minimum_samples} samples")
            if any(
                isinstance(sample, bool)
                or not isinstance(sample, (int, float))
                or not math.isfinite(float(sample))
                or float(sample) <= 0.0
                for sample in samples
            ):
                raise ValueError(f"MC2 measurement {name} must contain finite positive samples")
            timings.append([float(sample) for sample in samples])
        errors = []
        for name in ("max_abs_norm", "norm_atol", "max_abs_added", "added_atol"):
            error = row.get(name)
            if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(float(error)):
                raise ValueError(f"MC2 measurement {name} must be finite")
            errors.append(float(error))
        if errors[1] < 0.0 or errors[3] < 0.0 or errors[0] < 0.0 or errors[2] < 0.0:
            raise ValueError("MC2 numerical errors and tolerances must be non-negative")
        scaled_names = ("max_scaled_norm", "norm_rtol", "max_scaled_added", "added_rtol")
        scaled_present = [name in row for name in scaled_names]
        if any(scaled_present) and not all(scaled_present):
            raise ValueError("MC2 relative numerical qualification requires all scaled-error fields")
        if all(scaled_present):
            scaled_values = []
            for name in scaled_names:
                value = row[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(f"MC2 measurement {name} must be finite and non-negative")
                scaled_values.append(float(value))
            if scaled_values[1] == 0.0 and errors[1] == 0.0:
                raise ValueError("MC2 norm numerical gate cannot have zero atol and rtol")
            if scaled_values[3] == 0.0 and errors[3] == 0.0:
                raise ValueError("MC2 added numerical gate cannot have zero atol and rtol")
        key = (*integers, dtype, weight_format, bool(row.get("is_trans_b", True)))
        if key in entries:
            raise ValueError(f"Duplicate MC2 measurement shape {key!r}")
        entries[key] = {
            **dict(row),
            "baseline_latency_ms": timings[0],
            "fused_latency_ms": timings[1],
        }
    return MC2Profile(
        _freeze_profile_value(normalized_metadata),
        _freeze_profile_value(entries),
        latency_percentile,
        minimum_samples,
        minimum_speedup,
    )


def validate_mc2_static_environment(
    profile: MC2Profile,
    device: torch.device | str,
    *,
    tp_size: int,
    epsilon: float,
) -> str | None:
    """Validate model-wide invariants once after weights have been loaded.

    Runtime/provider identity is intentionally handled by
    :func:`validate_mc2_runtime_binding` during worker admission.  This check
    binds the already-admitted profile to the loaded target model and active
    hardware without repeating those queries for every decoder layer.
    """

    resolved = torch.device(device)
    if resolved.type != "npu":
        return "MC2 static routes require an Ascend NPU target model"
    try:
        actual_hardware = str(torch.npu.get_device_name(resolved.index))
    except (AttributeError, RuntimeError, TypeError):
        return "cannot identify the active Ascend device"
    if _normalize_identity(actual_hardware) != _normalize_identity(str(profile.metadata["hardware"])):
        return "MC2 profile hardware does not match the active device"
    if int(profile.metadata["tensor_parallel_size"]) != int(tp_size):
        return "MC2 profile tensor parallel size does not match execution"
    profiled_epsilon = profile.metadata.get("rms_norm_epsilon")
    if profiled_epsilon is None:
        return "legacy MC2 profile lacks an RMSNorm epsilon binding"
    if float(profiled_epsilon) != float(epsilon):
        return "MC2 profile RMSNorm epsilon does not match execution"
    return None


def build_mc2_static_route(
    profile: MC2Profile,
    *,
    projection_kind: str,
    layer_index: int,
    group_tp: str,
    tp_rank_size: int,
    tp_rank_id: int,
    epsilon: float,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    enabled: bool = True,
    disabled_reason: str = "",
    is_trans_b: bool = True,
) -> MC2StaticRoute:
    """Build one sealed route from immutable post-load model parameters."""

    if projection_kind not in ("attention", "down"):
        raise ValueError("MC2 static route projection kind must be 'attention' or 'down'")
    if layer_index < 0:
        raise ValueError("MC2 static route layer index must be non-negative")
    if tp_rank_size < 2 or not 0 <= tp_rank_id < tp_rank_size:
        raise ValueError("MC2 static route tensor-parallel metadata is invalid")
    if not group_tp:
        raise RuntimeError("MC2 static route requires a pre-resolved HCCL communicator")
    if weight.ndim != 2 or gamma.ndim != 1 or weight.shape[0] != gamma.numel():
        raise ValueError("MC2 static route weight/gamma shapes are inconsistent")
    if enabled and not (weight.is_contiguous() and gamma.is_contiguous()):
        raise RuntimeError("MC2 static route requires contiguous projection weight and RMSNorm gamma")
    weight_format = _weight_format(weight)
    if enabled and weight_format == "unknown":
        raise RuntimeError("MC2 static route cannot identify the projection weight format")
    dtype = _dtype_name(weight)
    input_size = int(weight.shape[1] if is_trans_b else weight.shape[0])
    output_size = int(weight.shape[0] if is_trans_b else weight.shape[1])
    reason = str(disabled_reason)
    if enabled and profile.metadata.get("operator") == MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
        static_error = _native_epilogue_static_shape_error(
            (1, input_size, output_size, dtype, weight_format, bool(is_trans_b)),
            tp_size=tp_rank_size,
            epsilon=epsilon,
        )
        if static_error is None and (_dtype_name(gamma) != "bfloat16" or not _is_nd_tensor(gamma)):
            static_error = "native-MatMul MC2 epilogue requires BF16 ND gamma"
        if static_error is not None:
            enabled = False
            reason = static_error
    decisions: dict[int, MC2Qualification] = {}
    if enabled:
        for key in profile.entries:
            rows, k, n, entry_dtype, entry_format, entry_trans_b = key
            if (
                k == input_size
                and n == output_size
                and entry_dtype == dtype
                and entry_format == weight_format
                and entry_trans_b == bool(is_trans_b)
            ):
                decisions[int(rows)] = profile.qualify_static_shape(
                    key,
                    tp_size=tp_rank_size,
                    epsilon=epsilon,
                )
    if not enabled and not reason:
        reason = "MC2 static route is disabled by model policy"
    return MC2StaticRoute(
        projection_kind=projection_kind,
        operator_kind=str(profile.metadata["operator"]),
        layer_index=int(layer_index),
        group_tp=group_tp,
        tp_rank_size=int(tp_rank_size),
        tp_rank_id=int(tp_rank_id),
        epsilon=float(epsilon),
        is_trans_b=bool(is_trans_b),
        profile_identity=id(profile),
        weight_identity=id(weight),
        gamma_identity=id(gamma),
        input_size=input_size,
        output_size=output_size,
        dtype=dtype,
        weight_format=weight_format,
        weight_shape=tuple(int(value) for value in weight.shape),
        weight_stride=tuple(int(value) for value in weight.stride()),
        gamma_shape=tuple(int(value) for value in gamma.shape),
        gamma_stride=tuple(int(value) for value in gamma.stride()),
        activation_layout="contiguous",
        weight_layout="contiguous",
        residual_layout="contiguous",
        gamma_layout="contiguous",
        device=str(weight.device),
        decisions=MappingProxyType(decisions),
        enabled=bool(enabled),
        disabled_reason=reason,
        seal=_MC2_STATIC_ROUTE_SEAL,
    )


def build_mc2_static_route_manifest(
    profile: MC2Profile,
    routes: Sequence[MC2StaticRoute],
) -> MC2StaticRouteManifest:
    """Freeze routes and compute their rank-independent consensus digest."""

    frozen_routes = tuple(routes)
    if not frozen_routes:
        raise ValueError("MC2 static route manifest cannot be empty")
    if any(route.seal is not _MC2_STATIC_ROUTE_SEAL for route in frozen_routes):
        raise ValueError("MC2 static route manifest contains an untrusted route")
    logical_keys = tuple((route.layer_index, route.projection_kind) for route in frozen_routes)
    if len(set(logical_keys)) != len(logical_keys):
        raise ValueError("MC2 static route manifest contains duplicate layer projections")
    profile_sha256 = _mc2_profile_sha256(profile)
    document = {
        "profile_sha256": profile_sha256,
        "routes": [route.consensus_document() for route in frozen_routes],
    }
    digest = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return MC2StaticRouteManifest(
        routes=frozen_routes,
        profile_sha256=profile_sha256,
        digest=digest,
    )


def bind_mc2_static_route(
    route: MC2StaticRoute,
    profile: MC2Profile,
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
) -> _MC2DispatchTicket:
    """Bind a precomputed route to one exact forward invocation.

    Dynamic row count is the only permitted varying shape.  Any mutation of
    parameter ownership, dtype/device, or projection dimensions is a model
    lifecycle violation and fails closed instead of silently requalifying.
    """

    if route.seal is not _MC2_STATIC_ROUTE_SEAL or route.profile_identity != id(profile):
        raise RuntimeError("MC2 static route does not belong to this qualification profile")
    if route.operator_kind != profile.metadata.get("operator"):
        raise RuntimeError("MC2 static route operator identity changed after freeze")
    if route.weight_identity != id(weight) or route.gamma_identity != id(gamma):
        raise RuntimeError("MC2 static route parameter ownership changed after freeze")
    if (
        tuple(int(value) for value in weight.shape) != route.weight_shape
        or tuple(int(value) for value in weight.stride()) != route.weight_stride
        or tuple(int(value) for value in gamma.shape) != route.gamma_shape
        or tuple(int(value) for value in gamma.stride()) != route.gamma_stride
    ):
        raise RuntimeError("MC2 static route parameter layout changed after freeze")
    if str(x.device) != route.device or str(weight.device) != route.device or str(residual.device) != route.device:
        raise RuntimeError("MC2 static route device changed after freeze")
    if any(
        layout != "contiguous"
        for layout in (
            route.activation_layout,
            route.weight_layout,
            route.residual_layout,
            route.gamma_layout,
        )
    ) or not (
        x.is_contiguous()
        and weight.is_contiguous()
        and residual.is_contiguous()
        and gamma.is_contiguous()
    ):
        raise RuntimeError(
            "MC2 static route requires contiguous activation, weight, residual, and gamma tensors"
        )
    if (
        _dtype_name(x) != route.dtype
        or _dtype_name(weight) != route.dtype
        or _dtype_name(residual) != route.dtype
        or _dtype_name(gamma) != route.dtype
    ):
        raise RuntimeError("MC2 static route dtype changed after freeze")
    if (
        int(x.shape[-1]) != route.input_size
        or int(residual.shape[-1]) != route.output_size
        or residual.shape[:-1] != x.shape[:-1]
        or gamma.numel() != route.output_size
    ):
        raise RuntimeError("MC2 static route tensor shape changed outside its row dimension")
    rows = int(math.prod(x.shape[:-1]))
    qualification = route.decisions.get(rows)
    if qualification is None:
        reason = route.disabled_reason if not route.enabled else f"unprofiled MC2 static row count {rows}"
        qualification = MC2Qualification(False, reason)
    return _bind_mc2_dispatch_ticket(
        qualification,
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=route.tp_rank_size,
        tp_rank_id=route.tp_rank_id,
        epsilon=route.epsilon,
        is_trans_b=route.is_trans_b,
        profile=profile,
    )


def resolve_hccl_comm_name(
    process_group: dist.ProcessGroup | None = None,
    *,
    device: torch.device | str | None = None,
    rank: int | None = None,
) -> str:
    """Resolve the communicator handle expected by the Ascend MC2 op.

    vLLM-Ascend has used both ``get_hccl_comm_name`` and backend-specific
    process-group helpers across CANN releases.  Keep this compatibility
    probing isolated so the numerical fallback remains usable when no helper
    is exposed (for example in CPU unit tests).
    """
    if process_group is None or not dist.is_available() or not dist.is_initialized():
        return ""
    try:
        backend = process_group._get_backend(torch.device(device or "npu"))  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, TypeError):
        return ""
    getter = getattr(backend, "get_hccl_comm_name", None)
    if getter is None:
        getter = getattr(process_group, "get_hccl_comm_name", None)
    if getter is None:
        return ""
    # HCCL's communicator lookup is keyed by the *global* rank for a
    # non-contiguous subgroup in current vLLM-Ascend releases.  Native PEARL
    # keeps the model-parallel rank in ``NativeTPContext`` for tensor slicing,
    # so resolve both namespaces and retain the local-rank fallback used by
    # older torch-npu versions.
    local_rank = rank if rank is not None else dist.get_rank(process_group)
    candidates: list[int | None] = []
    try:
        global_rank = dist.get_global_rank(process_group, local_rank)
    except (AttributeError, RuntimeError, TypeError):
        global_rank = None
    for candidate in (global_rank, local_rank, None):
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        try:
            value = getter() if candidate is None else getter(candidate)
        except (TypeError, RuntimeError, AttributeError):
            continue
        if isinstance(value, str) and value:
            return value
    return ""


def _matmul_op() -> Any | None:
    try:
        return torch.ops._C_ascend.matmul_allreduce_add_rmsnorm
    except (AttributeError, RuntimeError):
        return None


def _native_epilogue_op() -> Any | None:
    try:
        return torch.ops._C_ascend.allreduce_add_rmsnorm
    except (AttributeError, RuntimeError):
        return None


def _native_epilogue_chained_op() -> Any | None:
    try:
        return torch.ops._C_ascend.allreduce_add_rmsnorm_chained
    except (AttributeError, RuntimeError):
        return None


def detect_mc2_capability(
    device: torch.device | str | None = None,
    tp_size: int = 1,
    *,
    require_fused: bool = True,
    operator: str | None = None,
) -> MC2Capability:
    """Report whether the compiled MC2 operator can be dispatched."""

    resolved = torch.device(device or "cpu")
    operator_kind = _normalize_mc2_operator(operator)
    package_operator = _mc2_package_operator_name(operator_kind)
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if resolved.type != "npu":
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            operator_kind,
            "MC2 is an Ascend NPU operator",
        )
    if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR and tp_size != 3:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            operator_kind,
            "native-MatMul MC2 epilogue requires tensor parallel size 3",
        )
    global _MC2_LOAD_ATTEMPTED
    if not _MC2_LOAD_ATTEMPTED:
        _MC2_LOAD_ATTEMPTED = True
        try:
            # PEARL can be imported before the regular worker bootstrap. Load
            # the extension lazily for capability checks, but never fail a
            # request just because an optional custom-op library is absent.
            from vllm_ascend.utils import enable_custom_op

            enable_custom_op()
        except Exception:  # optional extension failures must keep fallback usable
            pass
    try:
        payload = _resolve_active_mc2_package_cached(
            os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
            operator_kind,
        )
    except (ImportError, OSError, ValueError) as error:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            operator_kind,
            f"cannot validate the active MC2 vendor payload: {error}",
        )
    if not payload.available:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            operator_kind,
            payload.reason,
        )
    fused_op = (
        _native_epilogue_op()
        if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
        else _matmul_op()
    )
    if fused_op is None:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            operator_kind,
            f"vllm_ascend {package_operator} custom extension is not loaded",
        )
    if require_fused and tp_size < 2:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            operator_kind,
            "fused MC2 is only useful for tensor-parallel groups",
        )
    return MC2Capability(
        True,
        str(resolved),
        int(tp_size),
        operator_kind,
        "custom AscendC/aclnn dispatch is registered",
    )


def matmul_allreduce_add_rmsnorm_or_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    *,
    group_tp: str = "",
    tp_rank_size: int = 1,
    tp_rank_id: int = 0,
    epsilon: float = 1e-6,
    is_trans_b: bool = True,
    is_gather_add_out: bool = False,
    process_group: dist.ProcessGroup | None = None,
    use_fused: bool | None = None,
    strict_fused: bool = False,
    profile: MC2Profile | None = None,
    prequalification: _MC2DispatchTicket | None = None,
    chain_state: torch.Tensor | None = None,
    flush_chain: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch MC2 or return the equivalent matmul/all-reduce/RMSNorm pair."""

    operator_kind = _normalize_mc2_operator(
        profile.metadata.get("operator") if profile is not None else None
    )
    output_size = weight.shape[0] if is_trans_b else weight.shape[-1]
    if x.ndim < 2 or residual.shape != x.shape[:-1] + (output_size,):
        raise ValueError("MC2 input/output shapes are inconsistent")
    if gamma.numel() != residual.shape[-1]:
        raise ValueError("RMSNorm gamma must match the residual hidden size")
    if tp_rank_size < 1 or not 0 <= tp_rank_id < tp_rank_size:
        raise ValueError("invalid tensor-parallel rank metadata")
    chained_dispatch = chain_state is not None
    if chained_dispatch:
        if operator_kind != MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
            raise ValueError("MC2 chained state is supported only by the TP3 native epilogue")
        if not strict_fused or use_fused is not True:
            raise ValueError("MC2 chained state requires strict profiled fused dispatch")
        if (
            chain_state.dtype != torch.int64
            or tuple(chain_state.shape) != (64, 4)
            or chain_state.device != residual.device
            or not chain_state.is_contiguous()
            or not _is_nd_tensor(chain_state)
        ):
            raise ValueError("MC2 chain state must be a contiguous int64[64, 4] ND tensor on the target NPU")
    if prequalification is not None:
        if use_fused is not True or not strict_fused or profile is None:
            raise ValueError("MC2 prequalification is only valid for strict profiled fused dispatch")
        expected_signature = _mc2_dispatch_signature(
            x,
            weight,
            residual,
            gamma,
            tp_rank_size=tp_rank_size,
            tp_rank_id=tp_rank_id,
            epsilon=epsilon,
            is_trans_b=is_trans_b,
            profile=profile,
        )
        if (
            not isinstance(prequalification, _MC2DispatchTicket)
            or prequalification.seal is not _MC2_DISPATCH_TICKET_SEAL
            or prequalification.signature != expected_signature
        ):
            raise RuntimeError("MC2 prequalification ticket does not match this exact dispatch")
        qualification = prequalification.qualification
        dispatch_fused = qualification.qualified
    else:
        capability = (
            detect_mc2_capability(
                x.device,
                tp_rank_size,
                operator=operator_kind,
            )
            if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
            else detect_mc2_capability(x.device, tp_rank_size)
        )
        dispatch_fused = capability.available if use_fused is None else bool(use_fused) and capability.available
        qualification = (
            profile.qualify(
                x,
                weight,
                residual,
                tp_size=tp_rank_size,
                is_trans_b=is_trans_b,
                tp_rank_id=tp_rank_id,
                epsilon=epsilon,
            )
            if dispatch_fused and profile is not None
            else MC2Qualification(False, "no identity-bound MC2 qualification profile")
        )
        dispatch_fused = dispatch_fused and qualification.qualified
    if dispatch_fused and operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
        contract_error = _native_epilogue_dispatch_error(
            x,
            weight,
            residual,
            gamma,
            tp_rank_size=tp_rank_size,
            epsilon=epsilon,
            is_trans_b=is_trans_b,
            is_gather_add_out=is_gather_add_out,
        )
        if contract_error is not None:
            qualification = MC2Qualification(False, contract_error)
            dispatch_fused = False
    if bool(use_fused) and strict_fused and not dispatch_fused:
        raise RuntimeError(f"MC2 fused dispatch is not qualified: {qualification.reason}")
    dispatch_key = (
        math.prod(x.shape[:-1]),
        int(x.shape[-1]),
        int(residual.shape[-1]),
        _dtype_name(x),
        _weight_format(weight),
        bool(is_trans_b),
    )
    fallback_reason = qualification.reason
    fused_failure = False
    if dispatch_fused:
        _increment_mc2_dispatch_counter("fused_attempt")
        path_counter = (
            "native_epilogue"
            if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
            else "full_fused"
        )
        _increment_mc2_dispatch_counter(f"{path_counter}_attempt")
        if chained_dispatch:
            _increment_mc2_dispatch_counter("native_epilogue_chained_attempt")
        if chained_dispatch:
            fused_op = _native_epilogue_chained_op()
        elif operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
            fused_op = _native_epilogue_op()
        else:
            fused_op = _matmul_op()
        if fused_op is None:
            error = RuntimeError("qualified MC2 operator is unavailable at dispatch")
            _increment_mc2_dispatch_counter("exception")
            fused_failure = True
            fallback_reason = f"{type(error).__name__}: {error}"
            if strict_fused:
                raise RuntimeError("MC2 fused dispatch failed in strict mode") from error
        else:
            try:
                # The profile binds dispatch to the measured epilogue.  The
                # projection-only experiment remains available, but production
                # selects the one-launch fused epilogue unless that exact two-op
                # pipeline was explicitly qualified on this source/hardware.
                projection_only = (
                    tp_rank_size == 3
                    and math.prod(x.shape[:-1]) <= 160
                    and profile is not None
                    and profile.metadata.get("operator") == "tp3_matmul_allreduce+native_add_rmsnorm"
                )
                if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR:
                    local_projection = F.linear(x, weight)
                    if not local_projection.is_contiguous() or not _is_nd_tensor(local_projection):
                        raise RuntimeError(
                            "native F.linear did not produce the contiguous ND projection required by MC2"
                        )
                    if chained_dispatch:
                        outputs = fused_op(  # type: ignore[misc]
                            local_projection,
                            residual,
                            gamma,
                            chain_state,
                            group_tp,
                            int(tp_rank_size),
                            int(tp_rank_id),
                            float(epsilon),
                            bool(is_gather_add_out),
                            bool(flush_chain),
                        )
                    else:
                        outputs = fused_op(  # type: ignore[misc]
                            local_projection,
                            residual,
                            gamma,
                            group_tp,
                            int(tp_rank_size),
                            int(tp_rank_id),
                            float(epsilon),
                            bool(is_gather_add_out),
                        )
                else:
                    outputs = fused_op(  # type: ignore[misc]
                        x,
                        weight,
                        residual,
                        gamma,
                        group_tp,
                        int(tp_rank_size),
                        int(tp_rank_id),
                        float(epsilon),
                        bool(is_trans_b),
                        bool(is_gather_add_out),
                        projection_only,
                    )
                if projection_only:
                    import torch_npu

                    normalized, _, added = torch_npu.npu_add_rms_norm(outputs[0], residual, gamma, float(epsilon))
                    outputs = (normalized, added)
                _increment_mc2_dispatch_counter("fused_success")
                _increment_mc2_dispatch_counter(f"{path_counter}_success")
                if chained_dispatch:
                    _increment_mc2_dispatch_counter("native_epilogue_chained_success")
                    if flush_chain:
                        _increment_mc2_dispatch_counter("native_epilogue_chain_flush")
                if dispatch_key not in _MC2_LOGGED_FUSED_SHAPES:
                    _MC2_LOGGED_FUSED_SHAPES.add(dispatch_key)
                    logger.warning(
                        "Using qualified fused TP%d MatMul+AllReduce%s for exact shape %s "
                        "(fallback p95 %.6f ms, fused p95 %.6f ms; qualification: %s)",
                        tp_rank_size,
                        (
                            "+native AllReduce+AddRMSNorm"
                            if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
                            else "+native AddRMSNorm"
                            if projection_only
                            else "+AddRMSNorm"
                        ),
                        dispatch_key,
                        qualification.baseline_p95_ms,
                        qualification.fused_p95_ms,
                        qualification.reason,
                    )
                return outputs
            except Exception as error:
                _increment_mc2_dispatch_counter("exception")
                fused_failure = True
                fallback_reason = f"{type(error).__name__}: {error}"
                if strict_fused:
                    raise RuntimeError("MC2 fused dispatch failed in strict mode") from error
    if chained_dispatch:
        raise RuntimeError("MC2 chained dispatch cannot fall back with an outstanding mailbox dependency")
    _increment_mc2_dispatch_counter("fallback")
    logged_fallback_shapes = _MC2_LOGGED_EXCEPTION_SHAPES if fused_failure else _MC2_LOGGED_FALLBACK_SHAPES
    if (bool(use_fused) or fused_failure) and dispatch_key not in logged_fallback_shapes:
        logged_fallback_shapes.add(dispatch_key)
        logger.warning(
            "Using split matmul+all-reduce fallback for TP%d MC2 shape %s: %s",
            tp_rank_size,
            dispatch_key,
            fallback_reason,
        )
    if is_trans_b:
        matmul = F.linear(x, weight)
    else:
        matmul = torch.matmul(x, weight)
    if tp_rank_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(matmul, group=process_group)
    added = matmul + residual
    norm = added * torch.rsqrt(added.float().pow(2).mean(dim=-1, keepdim=True) + float(epsilon)).to(added.dtype)
    return norm * gamma, added


def capability_dict(device: torch.device | str | None = None, tp_size: int = 1) -> dict[str, Any]:
    """Return JSON-friendly MC2 capability information."""

    return asdict(detect_mc2_capability(device, tp_size))


__all__ = [
    "active_mc2_adapter_binary_sha256",
    "active_mc2_opapi_symbol_provider_sha256",
    "active_mc2_vendor_payload_sha256",
    "MC2Capability",
    "MC2_FULL_FUSED_OPERATOR",
    "MC2Profile",
    "MC2Qualification",
    "MC2_TP3_FULL_FUSED_OPERATOR",
    "MC2_TP3_NATIVE_EPILOGUE_OPERATOR",
    "MC2_TP3_NATIVE_RMSNORM_OPERATOR",
    "capability_dict",
    "current_mc2_runtime_binding",
    "detect_mc2_capability",
    "matmul_allreduce_add_rmsnorm_or_fallback",
    "mc2_source_sha256",
    "normalize_mc2_profile",
    "reset_mc2_dispatch_counters",
    "resolve_hccl_comm_name",
    "snapshot_mc2_dispatch_counters",
    "validate_mc2_runtime_binding",
]
