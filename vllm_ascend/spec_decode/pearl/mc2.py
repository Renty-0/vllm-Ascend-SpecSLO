# SPDX-License-Identifier: Apache-2.0
"""Optional MC2 fused dispatch for PEARL's TP residual path.

The production vLLM compiler pass already emits the AscendC
``matmul_allreduce_add_rmsnorm`` operator.  Native PEARL is intentionally
standalone, so this adapter exposes the same operator with a numerically
equivalent fallback.  Callers can probe support before opting in and retain
the fallback on CANN versions where the custom extension is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

_MC2_LOAD_ATTEMPTED = False


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


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _normalize_identity(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def mc2_source_sha256() -> str:
    """Hash the Python adapter and compiled MC2 sources used by this tree."""

    adapter = Path(__file__).resolve()
    repository = adapter.parents[3]
    sources = [adapter]
    source_root = repository / "csrc" / "mc2"
    if source_root.is_dir():
        sources.extend(sorted(path for path in source_root.rglob("*") if path.is_file()))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(repository) if path.is_relative_to(repository) else path.name).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _weight_format(tensor: torch.Tensor) -> str:
    if tensor.device.type != "npu":
        return "ND"
    try:
        import torch_npu

        return str(torch_npu.get_npu_format(tensor))
    except (AttributeError, RuntimeError, TypeError):
        return "unknown"


@dataclass(frozen=True)
class MC2Profile:
    """Identity-bound numerical and latency qualification for exact MC2 shapes."""

    metadata: Mapping[str, Any]
    entries: Mapping[tuple[int, int, int, str, str, bool], Mapping[str, Any]]
    latency_percentile: float
    minimum_samples: int
    minimum_speedup: float

    def qualify(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        *,
        tp_size: int,
        is_trans_b: bool,
    ) -> MC2Qualification:
        try:
            actual_hardware = str(torch.npu.get_device_name(x.device.index))
        except (AttributeError, RuntimeError, TypeError):
            return MC2Qualification(False, "cannot identify the active Ascend device")
        if _normalize_identity(actual_hardware) != _normalize_identity(str(self.metadata["hardware"])):
            return MC2Qualification(False, "MC2 profile hardware does not match the active device")
        m = math.prod(x.shape[:-1])
        k = int(x.shape[-1])
        n = int(residual.shape[-1])
        key = (int(m), k, n, _dtype_name(x), _weight_format(weight), bool(is_trans_b))
        row = self.entries.get(key)
        if row is None:
            return MC2Qualification(False, f"unprofiled MC2 shape {key!r}")
        if int(self.metadata["tensor_parallel_size"]) != int(tp_size):
            return MC2Qualification(False, "MC2 profile tensor parallel size does not match execution")
        baseline = _percentile(row["baseline_latency_ms"], self.latency_percentile)
        fused = _percentile(row["fused_latency_ms"], self.latency_percentile)
        numerical = float(row["max_abs_norm"]) <= float(row["norm_atol"]) and float(
            row["max_abs_added"]
        ) <= float(row["added_atol"])
        faster = fused * self.minimum_speedup <= baseline
        if not numerical:
            return MC2Qualification(False, "MC2 full-output numerical tolerance failed", baseline, fused)
        if not faster:
            return MC2Qualification(False, "MC2 repeated p95 latency is not faster than fallback", baseline, fused)
        return MC2Qualification(True, "exact shape passed numerical and repeated p95 gates", baseline, fused)


def normalize_mc2_profile(value: Mapping[str, Any] | str | Path) -> MC2Profile:
    """Load a strict profile; handwritten enable flags never qualify MC2."""

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
    if metadata.get("operator") != "matmul_allreduce_add_rmsnorm":
        raise ValueError("MC2 profile operator identity is invalid")
    if not isinstance(metadata.get("hardware"), str) or not metadata["hardware"].strip():
        raise ValueError("MC2 profile requires a hardware identity")
    tp_size = metadata.get("tensor_parallel_size")
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 2:
        raise ValueError("MC2 profile tensor_parallel_size must describe a TP group")
    if metadata.get("source_sha256") != mc2_source_sha256():
        raise ValueError("MC2 profile source hash does not match the current adapter/kernel sources")
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
        key = (*integers, dtype, weight_format, bool(row.get("is_trans_b", True)))
        if key in entries:
            raise ValueError(f"Duplicate MC2 measurement shape {key!r}")
        entries[key] = {
            **dict(row),
            "baseline_latency_ms": timings[0],
            "fused_latency_ms": timings[1],
        }
    return MC2Profile(dict(metadata), entries, latency_percentile, minimum_samples, minimum_speedup)


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


def detect_mc2_capability(
    device: torch.device | str | None = None,
    tp_size: int = 1,
    *,
    require_fused: bool = True,
) -> MC2Capability:
    """Report whether the compiled MC2 operator can be dispatched."""

    resolved = torch.device(device or "cpu")
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if resolved.type != "npu":
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            "matmul_allreduce_add_rmsnorm",
            "MC2 is an Ascend NPU operator",
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
    if _matmul_op() is None:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            "matmul_allreduce_add_rmsnorm",
            "vllm_ascend custom extension is not loaded",
        )
    if require_fused and tp_size < 2:
        return MC2Capability(
            False,
            str(resolved),
            int(tp_size),
            "matmul_allreduce_add_rmsnorm",
            "fused MC2 is only useful for tensor-parallel groups",
        )
    return MC2Capability(
        True,
        str(resolved),
        int(tp_size),
        "matmul_allreduce_add_rmsnorm",
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch MC2 or return the equivalent matmul/all-reduce/RMSNorm pair."""

    output_size = weight.shape[0] if is_trans_b else weight.shape[-1]
    if x.ndim < 2 or residual.shape != x.shape[:-1] + (output_size,):
        raise ValueError("MC2 input/output shapes are inconsistent")
    if gamma.numel() != residual.shape[-1]:
        raise ValueError("RMSNorm gamma must match the residual hidden size")
    if tp_rank_size < 1 or not 0 <= tp_rank_id < tp_rank_size:
        raise ValueError("invalid tensor-parallel rank metadata")
    capability = detect_mc2_capability(x.device, tp_rank_size)
    dispatch_fused = capability.available if use_fused is None else bool(use_fused) and capability.available
    qualification = (
        profile.qualify(
            x,
            weight,
            residual,
            tp_size=tp_rank_size,
            is_trans_b=is_trans_b,
        )
        if dispatch_fused and profile is not None
        else MC2Qualification(False, "no identity-bound MC2 qualification profile")
    )
    dispatch_fused = dispatch_fused and qualification.qualified
    if bool(use_fused) and strict_fused and not dispatch_fused:
        raise RuntimeError(f"MC2 fused dispatch is not qualified: {qualification.reason}")
    if dispatch_fused and _matmul_op() is not None:
        try:
            return _matmul_op()(  # type: ignore[misc]
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
            )
        except Exception as error:
            if strict_fused:
                raise RuntimeError("MC2 fused dispatch failed in strict mode") from error
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
    "MC2Capability",
    "MC2Profile",
    "MC2Qualification",
    "capability_dict",
    "detect_mc2_capability",
    "matmul_allreduce_add_rmsnorm_or_fallback",
    "mc2_source_sha256",
    "normalize_mc2_profile",
    "resolve_hccl_comm_name",
]
