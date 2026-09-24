"""Locate custom-op artifacts bundled with the vLLM Ascend package.

Keep this module dependency-free: packaging checks and early worker startup may
need it before torch/CANN initialization is safe.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

CUSTOM_OP_VENDOR_NAME = "custom_transformer"
CUSTOM_OPAPI_RELATIVE_PATH = Path("op_api/lib/libcust_opapi.so")
CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH = Path("op_api/include/aclnnop/aclnn_kv_cache_block_gather.h")
CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR = Path("op_impl/ai_core/tbe/kernel/config")
CUSTOM_OP_GATHER_KERNEL_MANIFEST = "kv_cache_block_gather.json"
CUSTOM_OP_MC2_NAME = "matmul_allreduce_add_rmsnorm"
CUSTOM_OP_MC2_HEADER_RELATIVE_PATH = Path("op_api/include/aclnnop/aclnn_matmul_allreduce_add_rmsnorm.h")
CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR = CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR
CUSTOM_OP_MC2_KERNEL_MANIFEST = f"{CUSTOM_OP_MC2_NAME}.json"
CUSTOM_OP_MC2_EPILOGUE_NAME = "allreduce_add_rmsnorm"
CUSTOM_OP_MC2_EPILOGUE_HEADER_RELATIVE_PATH = Path(
    "op_api/include/aclnnop/aclnn_allreduce_add_rmsnorm.h"
)
CUSTOM_OP_MC2_EPILOGUE_KERNEL_MANIFEST = f"{CUSTOM_OP_MC2_EPILOGUE_NAME}.json"
CUSTOM_OPP_ENV = "ASCEND_CUSTOM_OPP_PATH"

_CUSTOM_OP_PAYLOADS = {
    "kv_cache_block_gather": (
        CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH,
        CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR,
        CUSTOM_OP_GATHER_KERNEL_MANIFEST,
    ),
    CUSTOM_OP_MC2_NAME: (
        CUSTOM_OP_MC2_HEADER_RELATIVE_PATH,
        CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR,
        CUSTOM_OP_MC2_KERNEL_MANIFEST,
    ),
    CUSTOM_OP_MC2_EPILOGUE_NAME: (
        CUSTOM_OP_MC2_EPILOGUE_HEADER_RELATIVE_PATH,
        CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR,
        CUSTOM_OP_MC2_EPILOGUE_KERNEL_MANIFEST,
    ),
}


@dataclass(frozen=True)
class CustomOpPackageResolution:
    """Result of resolving the opapi used by in-tree custom operators."""

    available: bool
    vendor_path: Path | None
    opapi_library: Path | None
    source: str
    reason: str


def bundled_custom_op_vendor_path(package_dir: str | Path | None = None) -> Path:
    """Return the expected vendor root for a source, editable, or wheel install."""
    base_dir = Path(__file__).resolve().parent if package_dir is None else Path(package_dir)
    return base_dir / "_cann_ops_custom" / "vendors" / CUSTOM_OP_VENDOR_NAME


def required_custom_op_payload_paths(
    vendor_path: str | Path,
    operator_name: str,
    *,
    kernel_soc: str,
) -> tuple[Path, ...]:
    """Return the packaged files required to advertise one custom operator."""
    try:
        header, config_dir, manifest = _CUSTOM_OP_PAYLOADS[operator_name]
    except KeyError as exc:
        raise ValueError(f"unknown custom operator payload: {operator_name}") from exc

    vendor = Path(vendor_path)
    return (
        vendor / CUSTOM_OPAPI_RELATIVE_PATH,
        vendor / header,
        vendor / config_dir / kernel_soc / manifest,
    )


def missing_custom_op_payload_artifacts(
    vendor_path: str | Path,
    operator_names: tuple[str, ...],
    *,
    kernel_soc: str,
) -> tuple[Path, ...]:
    """Return missing files for an exact wheel target without importing CANN."""
    required = {
        path
        for operator_name in operator_names
        for path in required_custom_op_payload_paths(
            vendor_path,
            operator_name,
            kernel_soc=kernel_soc,
        )
    }
    return tuple(sorted((path for path in required if not path.is_file()), key=str))


def custom_op_payload_declared(
    vendor_path: str | Path,
    operator_name: str,
    *,
    source_dir: str | Path | None = None,
) -> bool:
    """Whether source or generated build artifacts advertise an operator."""
    try:
        header, config_dir, manifest = _CUSTOM_OP_PAYLOADS[operator_name]
    except KeyError as exc:
        raise ValueError(f"unknown custom operator payload: {operator_name}") from exc

    if source_dir is not None and (Path(source_dir) / "CMakeLists.txt").is_file():
        return True

    vendor = Path(vendor_path)
    if (vendor / header).is_file():
        return True
    return any(path.is_file() for path in (vendor / config_dir).glob(f"*/{manifest}"))


def resolve_custom_op_vendor_payload(
    vendor_path: str | Path,
    operator_name: str,
    *,
    source: str = "explicit",
) -> CustomOpPackageResolution:
    """Resolve one operator from an already selected CANN vendor root."""
    try:
        header_relative_path, kernel_config_relative_dir, kernel_manifest = _CUSTOM_OP_PAYLOADS[operator_name]
    except KeyError as exc:
        raise ValueError(f"unknown custom operator payload: {operator_name}") from exc

    vendor_path = Path(vendor_path)
    if not vendor_path.is_dir():
        return CustomOpPackageResolution(
            available=False,
            vendor_path=None,
            opapi_library=None,
            source=source,
            reason=f"{source} custom-op vendor directory is missing: {vendor_path}",
        )

    opapi_library = vendor_path / CUSTOM_OPAPI_RELATIVE_PATH
    if not opapi_library.is_file():
        return CustomOpPackageResolution(
            available=False,
            vendor_path=vendor_path,
            opapi_library=None,
            source=source,
            reason=f"{source} custom-op opapi library is missing: {opapi_library}",
        )

    header = vendor_path / header_relative_path
    if not header.is_file():
        return CustomOpPackageResolution(
            available=False,
            vendor_path=vendor_path,
            opapi_library=None,
            source=source,
            reason=f"{source} {operator_name} opapi header is missing: {header}",
        )

    kernel_config_dir = vendor_path / kernel_config_relative_dir
    manifests = kernel_config_dir.glob(f"*/{kernel_manifest}")
    if not any(path.is_file() for path in manifests):
        return CustomOpPackageResolution(
            available=False,
            vendor_path=vendor_path,
            opapi_library=None,
            source=source,
            reason=f"{source} {operator_name} kernel manifest is missing under: {kernel_config_dir}",
        )

    return CustomOpPackageResolution(
        available=True,
        vendor_path=vendor_path,
        opapi_library=opapi_library,
        source=source,
        reason=f"using {source} {operator_name} custom-op package: {vendor_path}",
    )


def _resolve_custom_op_payload(
    operator_name: str,
    *,
    package_dir: str | Path | None,
) -> CustomOpPackageResolution:
    """Resolve one bundled operator payload without importing torch or CANN."""
    return resolve_custom_op_vendor_payload(
        bundled_custom_op_vendor_path(package_dir),
        operator_name,
        source="bundled",
    )


def resolve_custom_op_package(
    *,
    package_dir: str | Path | None = None,
) -> CustomOpPackageResolution:
    """Resolve the gather-capable package bundled with vLLM Ascend."""
    resolution = _resolve_custom_op_payload(
        "kv_cache_block_gather",
        package_dir=package_dir,
    )
    if resolution.available:
        return CustomOpPackageResolution(
            available=True,
            vendor_path=resolution.vendor_path,
            opapi_library=resolution.opapi_library,
            source=resolution.source,
            reason=f"using bundled custom-op package: {resolution.vendor_path}",
        )
    return resolution


def resolve_matmul_allreduce_add_rmsnorm_package(
    *,
    package_dir: str | Path | None = None,
) -> CustomOpPackageResolution:
    """Resolve the complete bundled MC2 payload independently of gather."""
    return _resolve_custom_op_payload(
        CUSTOM_OP_MC2_NAME,
        package_dir=package_dir,
    )


def resolve_allreduce_add_rmsnorm_package(
    *,
    package_dir: str | Path | None = None,
) -> CustomOpPackageResolution:
    """Resolve the bundled native-MatMul MC2 epilogue payload."""
    return _resolve_custom_op_payload(
        CUSTOM_OP_MC2_EPILOGUE_NAME,
        package_dir=package_dir,
    )


def _resolve_active_operator_package(
    operator_name: str,
    *,
    package_dir: str | Path | None,
    environ: MutableMapping[str, str] | None,
    capability_name: str | None = None,
) -> CustomOpPackageResolution:
    """Resolve one operator from the active CANN search path or wheel."""
    target_environ = os.environ if environ is None else environ
    candidates: list[tuple[Path, str]] = []
    for entry in target_environ.get(CUSTOM_OPP_ENV, "").split(os.pathsep):
        if not entry:
            continue
        root = Path(entry)
        vendor = root / "vendors" / CUSTOM_OP_VENDOR_NAME if (root / "vendors").is_dir() else root
        item = (vendor, "environment")
        if not any(candidate == vendor for candidate, _ in candidates):
            candidates.append(item)
    bundled_item = (bundled_custom_op_vendor_path(package_dir), "bundled")
    if not any(candidate == bundled_item[0] for candidate, _ in candidates):
        candidates.append(bundled_item)

    failures: list[str] = []
    for vendor, source in candidates:
        resolution = resolve_custom_op_vendor_payload(
            vendor,
            operator_name,
            source=source,
        )
        if resolution.available:
            return resolution
        failures.append(resolution.reason)
    return CustomOpPackageResolution(
        available=False,
        vendor_path=None,
        opapi_library=None,
        source="active",
        reason=(
            f"no complete active {capability_name or operator_name} vendor payload: "
            + "; ".join(failures)
        ),
    )


def resolve_active_matmul_allreduce_add_rmsnorm_package(
    *,
    package_dir: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> CustomOpPackageResolution:
    """Resolve MC2 from the active CANN search path or bundled wheel.

    Development qualification can point ``ASCEND_CUSTOM_OPP_PATH`` at a
    freshly built vendor before rebuilding the Python wheel. Check every
    active entry as well as the bundled package: a gather-only bundled vendor
    may legitimately precede an MC2-capable development vendor.
    """
    return _resolve_active_operator_package(
        CUSTOM_OP_MC2_NAME,
        package_dir=package_dir,
        environ=environ,
        capability_name="MC2",
    )


def resolve_active_allreduce_add_rmsnorm_package(
    *,
    package_dir: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> CustomOpPackageResolution:
    """Resolve the native-MatMul MC2 epilogue from the active package."""
    return _resolve_active_operator_package(
        CUSTOM_OP_MC2_EPILOGUE_NAME,
        package_dir=package_dir,
        environ=environ,
        capability_name="native-MatMul MC2 epilogue",
    )


def _prepend_env_path(environ: MutableMapping[str, str], name: str, path: Path) -> None:
    path_str = os.fspath(path)
    entries = [entry for entry in environ.get(name, "").split(os.pathsep) if entry]
    if path_str not in entries:
        entries.insert(0, path_str)
        environ[name] = os.pathsep.join(entries)


def bootstrap_custom_op_package_env(
    *,
    package_dir: str | Path | None = None,
    include_vendor_lib: bool = False,
    environ: MutableMapping[str, str] | None = None,
) -> CustomOpPackageResolution:
    """Expose bundled kernels to CANN and return the capability decision."""
    target_environ = os.environ if environ is None else environ
    resolution = resolve_custom_op_package(package_dir=package_dir)

    if resolution.vendor_path is not None:
        _prepend_env_path(target_environ, CUSTOM_OPP_ENV, resolution.vendor_path)
        vendor_lib = resolution.vendor_path / "op_api" / "lib"
        if include_vendor_lib and vendor_lib.is_dir():
            _prepend_env_path(target_environ, "LD_LIBRARY_PATH", vendor_lib)

    return resolution


def activate_kv_cache_block_gather_runtime(
    torch_module,
    *,
    opapi_library: str | Path | None = None,
    package_dir: str | Path | None = None,
) -> Path:
    """Load the gather OPAPI library through the registered Torch adapter.

    ``opapi_library`` is a development-only argument used by the benchmark and
    smoke test. Production callers omit it and use the wheel-bundled package.
    No process-wide user configuration or silent fallback is involved.
    """
    resolution = bootstrap_custom_op_package_env(package_dir=package_dir)
    if opapi_library is None:
        if not resolution.available or resolution.opapi_library is None:
            raise RuntimeError(resolution.reason)
        selected_library = resolution.opapi_library
    else:
        selected_library = Path(opapi_library).expanduser().resolve()
        if not selected_library.is_file():
            raise RuntimeError(f"kv_cache_block_gather OPAPI library is not a file: {selected_library}")

    namespace = getattr(torch_module.ops, "_C_ascend", None)
    loader = None if namespace is None else getattr(namespace, "load_kv_cache_block_gather_runtime", None)
    capability = None if namespace is None else getattr(namespace, "has_kv_cache_block_gather_runtime", None)
    if loader is None or capability is None:
        raise RuntimeError("vllm_ascend extension is missing the kv_cache_block_gather runtime loader")
    if not loader(os.fspath(selected_library)) or not capability():
        raise RuntimeError(f"custom-op library does not expose kv_cache_block_gather: {selected_library}")
    return selected_library
