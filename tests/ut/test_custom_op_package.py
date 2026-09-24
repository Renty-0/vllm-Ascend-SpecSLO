from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.custom_op_package import (
    CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH,
    CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR,
    CUSTOM_OP_GATHER_KERNEL_MANIFEST,
    CUSTOM_OP_MC2_EPILOGUE_HEADER_RELATIVE_PATH,
    CUSTOM_OP_MC2_EPILOGUE_KERNEL_MANIFEST,
    CUSTOM_OP_MC2_EPILOGUE_NAME,
    CUSTOM_OP_MC2_HEADER_RELATIVE_PATH,
    CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR,
    CUSTOM_OP_MC2_KERNEL_MANIFEST,
    CUSTOM_OP_MC2_NAME,
    CUSTOM_OPAPI_RELATIVE_PATH,
    CUSTOM_OPP_ENV,
    activate_kv_cache_block_gather_runtime,
    bootstrap_custom_op_package_env,
    bundled_custom_op_vendor_path,
    custom_op_payload_declared,
    missing_custom_op_payload_artifacts,
    required_custom_op_payload_paths,
    resolve_active_allreduce_add_rmsnorm_package,
    resolve_active_matmul_allreduce_add_rmsnorm_package,
    resolve_allreduce_add_rmsnorm_package,
    resolve_custom_op_package,
    resolve_matmul_allreduce_add_rmsnorm_package,
)


def _create_bundled_package(package_dir: Path) -> Path:
    opapi = bundled_custom_op_vendor_path(package_dir) / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()
    vendor = bundled_custom_op_vendor_path(package_dir)
    gather_header = vendor / CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH
    gather_header.parent.mkdir(parents=True)
    gather_header.touch()
    gather_manifest = (
        vendor / CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR / "ascend910b" / CUSTOM_OP_GATHER_KERNEL_MANIFEST
    )
    gather_manifest.parent.mkdir(parents=True)
    gather_manifest.touch()
    return opapi


def _add_mc2_payload(package_dir: Path, *, add_manifest: bool = True) -> None:
    vendor = bundled_custom_op_vendor_path(package_dir)
    mc2_header = vendor / CUSTOM_OP_MC2_HEADER_RELATIVE_PATH
    mc2_header.parent.mkdir(parents=True, exist_ok=True)
    mc2_header.touch()
    if add_manifest:
        mc2_manifest = vendor / CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR / "ascend910b" / CUSTOM_OP_MC2_KERNEL_MANIFEST
        mc2_manifest.parent.mkdir(parents=True, exist_ok=True)
        mc2_manifest.touch()


def _add_mc2_epilogue_payload(package_dir: Path, *, add_manifest: bool = True) -> None:
    vendor = bundled_custom_op_vendor_path(package_dir)
    header = vendor / CUSTOM_OP_MC2_EPILOGUE_HEADER_RELATIVE_PATH
    header.parent.mkdir(parents=True, exist_ok=True)
    header.touch()
    if add_manifest:
        manifest = (
            vendor
            / CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR
            / "ascend910b"
            / CUSTOM_OP_MC2_EPILOGUE_KERNEL_MANIFEST
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.touch()


def test_resolve_bundled_custom_op_package(tmp_path: Path):
    opapi = _create_bundled_package(tmp_path)

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert resolution.available
    assert resolution.source == "bundled"
    assert resolution.vendor_path == bundled_custom_op_vendor_path(tmp_path)
    assert resolution.opapi_library == opapi
    assert "using bundled custom-op package" in resolution.reason


def test_legacy_gather_vendor_is_not_misreported_as_mc2_capable(tmp_path: Path):
    _create_bundled_package(tmp_path)

    gather_resolution = resolve_custom_op_package(package_dir=tmp_path)
    mc2_resolution = resolve_matmul_allreduce_add_rmsnorm_package(package_dir=tmp_path)

    assert gather_resolution.available
    assert not mc2_resolution.available
    assert mc2_resolution.reason == (
        f"bundled {CUSTOM_OP_MC2_NAME} opapi header is missing: "
        f"{bundled_custom_op_vendor_path(tmp_path) / CUSTOM_OP_MC2_HEADER_RELATIVE_PATH}"
    )


def test_resolve_complete_bundled_mc2_payload(tmp_path: Path):
    opapi = bundled_custom_op_vendor_path(tmp_path) / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()
    _add_mc2_payload(tmp_path)

    resolution = resolve_matmul_allreduce_add_rmsnorm_package(package_dir=tmp_path)

    assert resolution.available
    assert resolution.opapi_library == opapi
    assert resolution.vendor_path == bundled_custom_op_vendor_path(tmp_path)
    assert CUSTOM_OP_MC2_NAME in resolution.reason


def test_resolve_reports_missing_bundled_mc2_manifest(tmp_path: Path):
    _create_bundled_package(tmp_path)
    _add_mc2_payload(tmp_path, add_manifest=False)

    resolution = resolve_matmul_allreduce_add_rmsnorm_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.reason == (
        f"bundled {CUSTOM_OP_MC2_NAME} kernel manifest is missing under: "
        f"{bundled_custom_op_vendor_path(tmp_path) / CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR}"
    )


def test_native_matmul_epilogue_payload_is_resolved_independently(tmp_path: Path):
    opapi = bundled_custom_op_vendor_path(tmp_path) / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()
    _add_mc2_epilogue_payload(tmp_path)

    epilogue = resolve_allreduce_add_rmsnorm_package(package_dir=tmp_path)
    legacy = resolve_matmul_allreduce_add_rmsnorm_package(package_dir=tmp_path)

    assert epilogue.available
    assert epilogue.opapi_library == opapi
    assert CUSTOM_OP_MC2_EPILOGUE_NAME in epilogue.reason
    assert not legacy.available


def test_active_native_matmul_epilogue_resolver_requires_its_own_manifest(tmp_path: Path):
    _create_bundled_package(tmp_path)
    _add_mc2_epilogue_payload(tmp_path, add_manifest=False)

    resolution = resolve_active_allreduce_add_rmsnorm_package(
        package_dir=tmp_path,
        environ={},
    )

    assert not resolution.available
    assert "native-MatMul MC2 epilogue" in resolution.reason
    assert CUSTOM_OP_MC2_EPILOGUE_NAME in resolution.reason


def test_active_mc2_resolver_skips_gather_only_vendor(tmp_path: Path):
    bundled_package = tmp_path / "bundled"
    external_package = tmp_path / "external"
    _create_bundled_package(bundled_package)
    external_opapi = bundled_custom_op_vendor_path(external_package) / CUSTOM_OPAPI_RELATIVE_PATH
    external_opapi.parent.mkdir(parents=True)
    external_opapi.touch()
    _add_mc2_payload(external_package)

    bundled_vendor = bundled_custom_op_vendor_path(bundled_package)
    external_vendor = bundled_custom_op_vendor_path(external_package)
    resolution = resolve_active_matmul_allreduce_add_rmsnorm_package(
        package_dir=bundled_package,
        environ={CUSTOM_OPP_ENV: f"{bundled_vendor}:{external_vendor}"},
    )

    assert resolution.available
    assert resolution.source == "environment"
    assert resolution.vendor_path == external_vendor
    assert resolution.opapi_library == external_opapi


def test_active_mc2_resolver_rejects_all_incomplete_vendors(tmp_path: Path):
    _create_bundled_package(tmp_path)

    resolution = resolve_active_matmul_allreduce_add_rmsnorm_package(
        package_dir=tmp_path,
        environ={},
    )

    assert not resolution.available
    assert resolution.source == "active"
    assert "no complete active MC2 vendor payload" in resolution.reason
    assert CUSTOM_OP_MC2_NAME in resolution.reason


def test_packaging_payload_helpers_require_exact_soc_files(tmp_path: Path):
    _create_bundled_package(tmp_path)
    _add_mc2_payload(tmp_path)
    vendor = bundled_custom_op_vendor_path(tmp_path)

    assert custom_op_payload_declared(vendor, CUSTOM_OP_MC2_NAME)
    assert required_custom_op_payload_paths(
        vendor,
        CUSTOM_OP_MC2_NAME,
        kernel_soc="ascend910b",
    ) == (
        vendor / CUSTOM_OPAPI_RELATIVE_PATH,
        vendor / CUSTOM_OP_MC2_HEADER_RELATIVE_PATH,
        vendor / CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR / "ascend910b" / CUSTOM_OP_MC2_KERNEL_MANIFEST,
    )
    assert (
        missing_custom_op_payload_artifacts(
            vendor,
            ("kv_cache_block_gather", CUSTOM_OP_MC2_NAME),
            kernel_soc="ascend910b",
        )
        == ()
    )

    missing_for_a3 = missing_custom_op_payload_artifacts(
        vendor,
        ("kv_cache_block_gather", CUSTOM_OP_MC2_NAME),
        kernel_soc="ascend910_93",
    )
    assert missing_for_a3 == (
        vendor / CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR / "ascend910_93" / CUSTOM_OP_GATHER_KERNEL_MANIFEST,
        vendor / CUSTOM_OP_MC2_KERNEL_CONFIG_RELATIVE_DIR / "ascend910_93" / CUSTOM_OP_MC2_KERNEL_MANIFEST,
    )


def test_payload_declaration_accepts_cmake_source_without_generated_files(tmp_path: Path):
    vendor = bundled_custom_op_vendor_path(tmp_path)
    source_dir = tmp_path / "csrc" / CUSTOM_OP_MC2_NAME
    source_dir.mkdir(parents=True)

    assert not custom_op_payload_declared(vendor, CUSTOM_OP_MC2_NAME, source_dir=source_dir)

    (source_dir / "CMakeLists.txt").touch()
    assert custom_op_payload_declared(vendor, CUSTOM_OP_MC2_NAME, source_dir=source_dir)


def test_resolve_reports_missing_vendor_directory(tmp_path: Path):
    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path is None
    assert resolution.opapi_library is None
    assert resolution.reason == (
        f"bundled custom-op vendor directory is missing: {bundled_custom_op_vendor_path(tmp_path)}"
    )


def test_resolve_reports_missing_bundled_opapi(tmp_path: Path):
    vendor_path = bundled_custom_op_vendor_path(tmp_path)
    vendor_path.mkdir(parents=True)

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path == vendor_path
    assert resolution.opapi_library is None
    assert resolution.reason == (
        f"bundled custom-op opapi library is missing: {vendor_path / CUSTOM_OPAPI_RELATIVE_PATH}"
    )


def test_resolve_reports_missing_bundled_gather_header(tmp_path: Path):
    vendor_path = bundled_custom_op_vendor_path(tmp_path)
    opapi = vendor_path / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path == vendor_path
    assert resolution.opapi_library is None
    assert resolution.reason == (
        f"bundled kv_cache_block_gather opapi header is missing: {vendor_path / CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH}"
    )


def test_resolve_reports_missing_bundled_gather_manifest(tmp_path: Path):
    vendor_path = bundled_custom_op_vendor_path(tmp_path)
    opapi = vendor_path / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()
    gather_header = vendor_path / CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH
    gather_header.parent.mkdir(parents=True)
    gather_header.touch()

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path == vendor_path
    assert resolution.opapi_library is None
    assert resolution.reason == (
        "bundled kv_cache_block_gather kernel manifest is missing under: "
        f"{vendor_path / CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR}"
    )


def test_bootstrap_selects_bundled_artifacts_without_manual_paths(tmp_path: Path):
    opapi = _create_bundled_package(tmp_path)
    old_opp = tmp_path / "existing-opp"
    environ = {CUSTOM_OPP_ENV: str(old_opp)}

    resolution = bootstrap_custom_op_package_env(
        package_dir=tmp_path,
        include_vendor_lib=True,
        environ=environ,
    )

    vendor = bundled_custom_op_vendor_path(tmp_path)
    assert resolution.available
    assert resolution.opapi_library == opapi
    assert environ[CUSTOM_OPP_ENV].split(":") == [str(vendor), str(old_opp)]
    assert environ["LD_LIBRARY_PATH"] == str(vendor / "op_api" / "lib")


def test_activate_uses_registered_extension_loader(tmp_path: Path):
    opapi = _create_bundled_package(tmp_path)
    loaded = []
    namespace = SimpleNamespace(
        load_kv_cache_block_gather_runtime=lambda path: loaded.append(path) or True,
        has_kv_cache_block_gather_runtime=lambda: True,
    )
    fake_torch = SimpleNamespace(ops=SimpleNamespace(_C_ascend=namespace))

    selected = activate_kv_cache_block_gather_runtime(
        fake_torch,
        package_dir=tmp_path,
    )

    assert selected == opapi
    assert loaded == [str(opapi)]


def test_packaging_uses_generated_vendor_name_and_checks_wheel_payload():
    repo_root = Path(__file__).resolve().parents[2]
    binding = (repo_root / "csrc/kv_cache_block_gather_binding.cpp").read_text(encoding="utf-8")
    setup = (repo_root / "setup.py").read_text(encoding="utf-8")

    assert "load_kv_cache_block_gather_runtime" in binding
    assert '"custom_transformer",' in setup
    assert "custom_op_package.py" in setup
    assert "CUSTOM_OP_MC2_NAME" in setup
    assert "custom_op_payload_declared" in setup
    assert "missing_custom_op_payload_artifacts" in setup
    assert "Custom-op build did not produce the complete packaged" in setup
    assert "shutil.copytree(src_cann_ops_custom, dst_cann_ops_custom)" in setup
