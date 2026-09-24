#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/blob/main/setup.py
#

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from sysconfig import get_paths

from packaging.version import InvalidVersion, Version
from setuptools import Command, Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools_scm import ScmVersion, get_version


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = str(Path(__file__).resolve().parent)
UPSTREAM_METADATA_FILE = os.path.join(ROOT_DIR, "upstream_version.json")
DISTRIBUTION_NAME = "vllm-ascend-hust"
logger = logging.getLogger(__name__)
custom_op_package = load_module_from_path(
    "_vllm_ascend_custom_op_package_for_build",
    os.path.join(ROOT_DIR, "vllm_ascend", "custom_op_package.py"),
)


def load_upstream_metadata() -> dict[str, str]:
    with open(UPSTREAM_METADATA_FILE, encoding="utf-8") as file:
        metadata = json.load(file)

    required_fields = ("release_version", "upstream_version", "upstream_commit")
    missing_fields = [field for field in required_fields if not metadata.get(field)]
    if missing_fields:
        missing_list = ", ".join(missing_fields)
        raise RuntimeError(f"Missing required fields in {UPSTREAM_METADATA_FILE}: {missing_list}")

    return metadata


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", ROOT_DIR, *args], text=True).strip()


def _git_maybe_output(*args: str) -> str | None:
    try:
        return _git_output(*args)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _list_reachable_release_tags(release_version: str) -> list[tuple[Version, str, str]]:
    tag_pattern = f"v{release_version}.post*"
    tags_output = _git_maybe_output("tag", "--merged", "HEAD", "--list", tag_pattern)
    if not tags_output:
        return []

    release_tags: list[tuple[Version, str, str]] = []
    for tag_name in tags_output.splitlines():
        try:
            parsed_version = Version(tag_name.removeprefix("v"))
        except InvalidVersion:
            continue

        commit_id = _git_output("rev-list", "-n", "1", tag_name)
        release_tags.append((parsed_version, tag_name, commit_id))

    release_tags.sort(key=lambda item: item[0])
    return release_tags


def _count_commits_since(commit_id: str) -> int:
    count = _git_maybe_output("rev-list", "--count", f"{commit_id}..HEAD")
    if count is not None:
        return int(count)

    fallback_count = _git_maybe_output("rev-list", "--count", "HEAD")
    if fallback_count is None:
        raise RuntimeError(f"Unable to count commits from {commit_id} to HEAD")

    logger.warning(
        "Upstream commit %s is not reachable from HEAD; falling back to total "
        "repository commit count for development distance.",
        commit_id,
    )
    return int(fallback_count)


def parse_vllm_ascend_scm_version(root: str, *, config) -> ScmVersion:
    del root
    metadata = load_upstream_metadata()
    head_commit = _git_output("rev-parse", "HEAD")
    short_commit = _git_output("rev-parse", "--short", "HEAD")
    branch_name = _git_maybe_output("symbolic-ref", "--quiet", "--short", "HEAD")
    dirty = bool(_git_maybe_output("status", "--porcelain", "--untracked-files=no"))

    release_tags = _list_reachable_release_tags(metadata["release_version"])
    if release_tags:
        tag_version, _, tag_commit = release_tags[-1]
        distance = _count_commits_since(tag_commit)
        return ScmVersion(
            tag=config.version_cls(str(tag_version)),
            config=config,
            distance=distance,
            node=f"g{short_commit}",
            dirty=dirty,
            branch=branch_name,
        )

    distance = _count_commits_since(metadata["upstream_commit"])
    if head_commit == metadata["upstream_commit"]:
        distance = 0

    return ScmVersion(
        tag=config.version_cls(metadata["release_version"]),
        config=config,
        distance=distance,
        node=f"g{short_commit}",
        dirty=dirty,
        branch=branch_name,
    )


def vllm_ascend_version_scheme(version: ScmVersion) -> str:
    metadata = load_upstream_metadata()
    release_version = metadata["release_version"]
    tag_version = Version(str(version.tag))

    if version.exact:
        return str(tag_version)

    next_post = 1 if tag_version.post is None else tag_version.post + 1
    return f"{release_version}.post{next_post}.dev{version.distance}"


def vllm_ascend_local_scheme(version: ScmVersion) -> str:
    if version.exact and not version.dirty:
        return ""
    if version.node is None:
        return "+unknown"
    return version.format_choice("+{node}", "+{node}.dirty")


def vllm_ascend_version_template() -> str:
    metadata = load_upstream_metadata()
    return f"""# file generated by vcs-versioning
# don't change, don't track in version control
from __future__ import annotations

__all__ = [
    "__version__",
    "__version_tuple__",
    "version",
    "version_tuple",
    "__upstream_version__",
    "upstream_version",
    "__upstream_commit__",
    "upstream_commit",
    "__commit_id__",
    "commit_id",
]

version: str
__version__: str
__version_tuple__: tuple[int | str, ...]
version_tuple: tuple[int | str, ...]
upstream_version: str
__upstream_version__: str
upstream_commit: str
__upstream_commit__: str
commit_id: str | None
__commit_id__: str | None

__version__ = version = {{version!r}}
__version_tuple__ = version_tuple = {{version_tuple!r}}

__upstream_version__ = upstream_version = {metadata["upstream_version"]!r}
__upstream_commit__ = upstream_commit = {metadata["upstream_commit"]!r}
__commit_id__ = commit_id = {{scm_version.short_node!r}}
"""


def resolve_version() -> str:
    version_file = Path(ROOT_DIR) / "vllm_ascend" / "_version.py"
    version_file_rel = os.path.relpath(version_file, ROOT_DIR)

    try:
        return get_version(
            root=ROOT_DIR,
            relative_to=__file__,
            write_to=version_file_rel,
            write_to_template=vllm_ascend_version_template(),
            version_scheme=vllm_ascend_version_scheme,
            local_scheme=vllm_ascend_local_scheme,
            parse=parse_vllm_ascend_scm_version,
            dist_name=DISTRIBUTION_NAME,
        )
    except (ImportError, LookupError):
        namespace = {}
        if version_file.exists():
            exec(version_file.read_text(encoding="utf-8"), namespace)
            version = namespace.get("__version__") or namespace.get("version")
            if isinstance(version, str) and version:
                return version

        # The checkout action in github action CI does not checkout the tag. It
        # only checks out the commit. In this case, we set a dummy version.
        return "0.0.0"


def check_or_set_default_env(cmake_args, env_name, env_variable, default_path=""):
    if env_variable is None:
        logging.warning(
            "No %s found in your environment, please try to set %s if you customize the installation path of this "
            "library, otherwise default path will be adapted during build this project",
            env_name,
            env_name,
        )
        logging.warning("Set default %s: %s", env_name, default_path)
        env_variable = default_path
    else:
        logging.info("Found existing %s: %s", env_name, env_variable)
    # cann package seems will check this environments in cmake, need write this env variable back.
    if env_name == "ASCEND_HOME_PATH":
        os.environ["ASCEND_HOME_PATH"] = env_variable
    cmake_args += [f"-D{env_name}={env_variable}"]
    return cmake_args


def get_value_from_lines(lines: list[str], key: str) -> str:
    for line in lines:
        line = " ".join(line.split())
        if key in line:
            return line.split(":")[-1].strip()
    return ""


def get_chip_type() -> str:
    try:
        # Get NPU ID
        npu_info_lines = subprocess.check_output(["npu-smi", "info", "-l"]).decode().strip().split("\n")
        npu_id = int(get_value_from_lines(npu_info_lines, "NPU ID"))

        # Stage 1: query board info without -c flag
        board_info_lines = (
            subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(npu_id)]).decode().strip().split("\n")
        )

        # Check if Chip Name exists (Ascend950 includes it directly)
        chip_name = get_value_from_lines(board_info_lines, "Chip Name")

        # Stage 2: query with -c flag only if Chip Name not found (A2/A3/310P)
        if not chip_name:
            chip_info_lines = (
                subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(npu_id), "-c", "0"])
                .decode()
                .strip()
                .split("\n")
            )
        else:
            # Ascend950 already has complete info
            chip_info_lines = board_info_lines

        # Extract required fields
        chip_name = get_value_from_lines(chip_info_lines, "Chip Name")
        chip_type = get_value_from_lines(chip_info_lines, "Chip Type")
        npu_name = get_value_from_lines(chip_info_lines, "NPU Name")

        if "310" in chip_name:
            # 310P case
            assert chip_type
            return (chip_type + chip_name).lower()
        elif "910" in chip_name:
            if chip_type:
                # A2 case
                assert not npu_name
                return (chip_type + chip_name).lower()
            else:
                # A3 case
                assert npu_name
                return (chip_name + "_" + npu_name).lower()
        elif "950" in chip_name:
            assert npu_name
            return (chip_name + "_" + npu_name).lower()
        else:
            raise ValueError(f"Unable to recognize chip name: {chip_name}, please manually set env SOC_VERSION")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Get chip info failed: {e}")
    except FileNotFoundError:
        logging.warning(
            "npu-smi command not found, if this is an npu envir, please check if npu driver is installed correctly."
        )
        return ""


envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "vllm_ascend", "envs.py"))

if not envs.SOC_VERSION:
    soc_version = get_chip_type()
    if not soc_version:
        raise RuntimeError(
            "Could not determine chip type automatically via 'npu-smi'. "
            "This can happen in a CPU-only environment. "
            "Please set the 'SOC_VERSION' environment variable to specify the target chip, for example:\n"
            '  - Atlas A2: export SOC_VERSION="ascend910b1"\n'
            '  - Atlas A3: export SOC_VERSION="ascend910_9391"\n'
            '  - Atlas 300I: export SOC_VERSION="ascend310p1"\n'
            '  - Atlas A5: export SOC_VERSION="<value starting with ascend950>"\n'
            "You can also refer to the SOC_VERSION defaults in Dockerfile*."
        )
    envs.SOC_VERSION = soc_version


def gen_build_info():
    soc_version = envs.SOC_VERSION

    soc_to_device = {
        "910b": "A2",
        "910c": "A3",
        "310p": "_310P",
        "ascend910b1": "A2",
        "ascend910b2": "A2",
        "ascend910b2c": "A2",
        "ascend910b3": "A2",
        "ascend910b4": "A2",
        "ascend910b4-1": "A2",
        "ascend910_9391": "A3",
        "ascend910_9381": "A3",
        "ascend910_9372": "A3",
        "ascend910_9392": "A3",
        "ascend910_9382": "A3",
        "ascend910_9362": "A3",
        "ascend310p1": "_310P",
        "ascend310p3": "_310P",
        "ascend310p5": "_310P",
        "ascend310p7": "_310P",
        "ascend310p3vir01": "_310P",
        "ascend310p3vir02": "_310P",
        "ascend310p3vir04": "_310P",
        "ascend310p3vir08": "_310P",
    }
    if "ascend950" in soc_version:
        device_type = "A5"
    else:
        assert soc_version in soc_to_device, (
            f"Undefined soc_version: {soc_version}. Please file an issue to vllm-ascend."
        )
        device_type = soc_to_device[soc_version]

    package_dir = os.path.join(ROOT_DIR, "vllm_ascend", "_build_info.py")
    with open(package_dir, "w+") as f:
        f.write("# Auto-generated file\n")
        f.write(f"__device_type__ = '{device_type}'\n")
    logging.info("Generated _build_info.py with SOC version: %s", soc_version)


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwargs) -> None:
        super().__init__(name, sources=[], py_limited_api=False, **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class custom_develop(develop):
    def run(self):
        gen_build_info()
        super().run()


class custom_build_info(build_py):
    def run(self):
        gen_build_info()
        super().run()


class build_and_install_aclnn(Command):
    description = "Build and install AclNN by running build_aclnn.sh"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            print("Running bash build_aclnn.sh ...")
            build_env = os.environ.copy()
            build_env["PYTHON_EXECUTABLE"] = sys.executable
            subprocess.check_call(
                ["bash", "csrc/build_aclnn.sh", ROOT_DIR, envs.SOC_VERSION],
                env=build_env,
            )
            print("build_aclnn.sh executed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error running build_aclnn.sh: {e}")
            raise SystemExit(e.returncode)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()
        num_jobs = max(1, num_jobs)

        return num_jobs

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        source_dir = os.path.abspath(ROOT_DIR)
        python_executable = sys.executable
        cmake_args = ["cmake"]
        # Default use release mode to compile the csrc code
        # Turbo now support compiled with Release, Debug and RelWithDebugInfo
        if envs.CMAKE_BUILD_TYPE is None or envs.CMAKE_BUILD_TYPE not in [
            "Debug",
            "Release",
            "RelWithDebugInfo",
        ]:
            envs.CMAKE_BUILD_TYPE = "Release"
        cmake_args += [f"-DCMAKE_BUILD_TYPE={envs.CMAKE_BUILD_TYPE}"]
        # Default dump the compile commands for lsp
        cmake_args += ["-DCMAKE_EXPORT_COMPILE_COMMANDS=1"]
        if envs.CXX_COMPILER is not None:
            cmake_args += [f"-DCMAKE_CXX_COMPILER={envs.CXX_COMPILER}"]
        if envs.C_COMPILER is not None:
            cmake_args += [f"-DCMAKE_C_COMPILER={envs.C_COMPILER}"]
        if envs.VERBOSE:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        # find ASCEND_HOME_PATH
        check_or_set_default_env(
            cmake_args,
            "ASCEND_HOME_PATH",
            envs.ASCEND_HOME_PATH,
            "/usr/local/Ascend/ascend-toolkit/latest",
        )

        # find PYTHON_EXECUTABLE
        check_or_set_default_env(cmake_args, "PYTHON_EXECUTABLE", sys.executable)
        # Modern CMake FindPython and pybind11 use the CamelCase spelling.
        # Keep the uppercase value above for older CMake consumers as well.
        cmake_args += [f"-DPython_EXECUTABLE={python_executable}"]
        cmake_args += [f"-DPython_INCLUDE_DIR={get_paths()['include']}"]

        # find PYTHON_INCLUDE_PATH
        check_or_set_default_env(cmake_args, "PYTHON_INCLUDE_PATH", get_paths()["include"])

        # ccache and ninja can not be applied at ascendc kernels now

        try:
            # if pybind11 is installed via pip
            pybind11_cmake_path = (
                subprocess.check_output([python_executable, "-m", "pybind11", "--cmakedir"]).decode().strip()
            )
        except subprocess.CalledProcessError as e:
            # else specify pybind11 path installed from source code on CI container
            raise RuntimeError(f"CMake configuration failed: {e}")

        install_path = os.path.join(ROOT_DIR, self.build_lib)
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path = os.path.join(ROOT_DIR, "vllm_ascend")
        # add CMAKE_INSTALL_PATH
        cmake_args += [f"-DCMAKE_INSTALL_PREFIX={install_path}"]

        try:
            torch_cmake_prefix_output = subprocess.check_output(
                [python_executable, "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
            ).decode()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to locate torch CMake prefix path: {e}")

        torch_cmake_prefix_path = next(
            (
                line.strip()
                for line in reversed(torch_cmake_prefix_output.splitlines())
                if line.strip() and os.path.isabs(line.strip())
            ),
            "",
        )
        if not torch_cmake_prefix_path:
            raise RuntimeError(f"Failed to parse torch CMake prefix path from output: {torch_cmake_prefix_output!r}")

        cmake_prefix_paths = [pybind11_cmake_path, torch_cmake_prefix_path]
        cmake_args += [f"-DCMAKE_PREFIX_PATH={';'.join(path for path in cmake_prefix_paths if path)}"]

        soc_version_map = {
            "910b": "ascend910b1",
            "910c": "ascend910_9392",
            "310p": "ascend310p1",
        }
        CANN_SOC_VERSION = soc_version_map.get(envs.SOC_VERSION, envs.SOC_VERSION)
        cmake_args += [f"-DSOC_VERSION={CANN_SOC_VERSION}"]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        torch_npu_command = f"{sys.executable} -m pip show torch-npu | grep '^Location:' | awk '{{print $2}}'"
        try:
            torch_npu_path = subprocess.check_output(torch_npu_command, shell=True).decode().strip()
            torch_npu_path += "/torch_npu"
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Retrieve torch version version failed: {e}")

        # add TORCH_NPU_PATH
        cmake_args += [f"-DTORCH_NPU_PATH={torch_npu_path}"]

        # Pass VLLM_ASCEND_ENABLE_BATCH_MEMCPY to CMake if explicitly set.
        # When unset (None), CMake will auto-detect from CANN headers.
        if envs.VLLM_ASCEND_ENABLE_BATCH_MEMCPY is not None:
            cmake_args += [f"-DVLLM_ASCEND_ENABLE_BATCH_MEMCPY={envs.VLLM_ASCEND_ENABLE_BATCH_MEMCPY}"]

        build_tool = []
        # TODO(ganyi): ninja and ccache support for ascend c auto codegen. now we can only use make build
        # if which('ninja') is not None:
        #     build_tool += ['-G', 'Ninja']
        # Default build tool to whatever cmake picks.

        cmake_args += [source_dir]
        logging.info("cmake config command: %s", cmake_args)
        proc = subprocess.Popen(cmake_args, cwd=self.build_temp, stderr=subprocess.PIPE, text=True)
        # Forward CMake stderr to the real stderr in real time while also
        # accumulating it so the full error is visible on failure.
        stderr_lines: list[str] = []
        if proc.stderr is not None:
            for line in proc.stderr:
                sys.stderr.write(line)
                stderr_lines.append(line)
        proc.wait()
        if proc.returncode != 0:
            stderr_text = "".join(stderr_lines).strip()
            if stderr_text:
                logging.error("CMake configuration failed with stderr:")
                for line in stderr_text.splitlines():
                    logging.error("  | %s", line)
            raise RuntimeError(f"CMake configuration failed (exit {proc.returncode})")

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        if not envs.COMPILE_CUSTOM_KERNELS:
            return
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError(f"Cannot find CMake executable: {e}")

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        os.makedirs(os.path.join(self.build_lib, "vllm_ascend"), exist_ok=True)

        def target_name(s: str) -> str:
            return s.removeprefix("vllm_ascend.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]
        try:
            subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)
        except OSError as e:
            raise RuntimeError(f"Build library failed: {e}")
        # Install the libraries
        install_args = [
            "cmake",
            "--install",
            ".",
        ]
        try:
            subprocess.check_call(install_args, cwd=self.build_temp)
        except OSError as e:
            raise RuntimeError(f"Install library failed: {e}")

        # copy back to build folder for editable build
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            import shutil

            for root, _, files in os.walk(self.build_temp):
                for file in files:
                    if file.endswith(".so"):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(self.build_lib, "vllm_ascend", file)
                        shutil.copy(src_path, dst_path)
                        print(f"Copy: {src_path} -> {dst_path}")

        # Copy the generated package into build_lib so wheel assembly sees it.
        # A2/A3 builds advertise kv_cache_block_gather. They also advertise MC2
        # when its source or generated payload is present, so reject incomplete
        # payloads rather than failing later during worker construction.
        src_cann_ops_custom = os.path.join(ROOT_DIR, "vllm_ascend", "_cann_ops_custom")
        dst_cann_ops_custom = os.path.join(self.build_lib, "vllm_ascend", "_cann_ops_custom")
        if envs.SOC_VERSION.startswith(("ascend910b", "ascend910_93")):
            vendor_dir = Path(
                src_cann_ops_custom,
                "vendors",
                "custom_transformer",
            )
            kernel_soc = "ascend910b" if envs.SOC_VERSION.startswith("ascend910b") else "ascend910_93"
            required_operators = ["kv_cache_block_gather"]
            # The current build recipe selects MC2 on A2. A3 packaging still
            # detects a generated/partial MC2 payload, but does not claim the
            # operator merely because the shared source tree contains it.
            for operator_name in (
                custom_op_package.CUSTOM_OP_MC2_NAME,
                custom_op_package.CUSTOM_OP_MC2_EPILOGUE_NAME,
            ):
                mc2_source_dir = (
                    Path(ROOT_DIR, "csrc", "mc2", operator_name)
                    if kernel_soc == "ascend910b"
                    else None
                )
                if custom_op_package.custom_op_payload_declared(
                    vendor_dir,
                    operator_name,
                    source_dir=mc2_source_dir,
                ):
                    required_operators.append(operator_name)
            missing = custom_op_package.missing_custom_op_payload_artifacts(
                vendor_dir,
                tuple(required_operators),
                kernel_soc=kernel_soc,
            )
            if missing:
                raise RuntimeError(
                    "Custom-op build did not produce the complete packaged "
                    f"capabilities {required_operators}; missing: " + ", ".join(map(str, missing))
                )

        if os.path.exists(src_cann_ops_custom):
            import shutil

            if os.path.exists(dst_cann_ops_custom):
                shutil.rmtree(dst_cann_ops_custom)
            shutil.copytree(src_cann_ops_custom, dst_cann_ops_custom)
            print(f"Copy: {src_cann_ops_custom} -> {dst_cann_ops_custom}")

    def run(self):
        if envs.COMPILE_CUSTOM_KERNELS:
            # First, ensure ACLNN custom-ops is built and installed.
            self.run_command("build_aclnn")

        # Then, run the standard build_ext command to compile the extensions
        super().run()


class custom_install(install):
    def run(self):
        self.run_command("build_ext")
        install.run(self)


VERSION = resolve_version()

ext_modules = []
if envs.COMPILE_CUSTOM_KERNELS:
    ext_modules = [CMakeExtension(name="vllm_ascend.vllm_ascend_C")]


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> list[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_ascend.")
    return requirements


cmdclass = {
    "develop": custom_develop,
    "build_py": custom_build_info,
    "build_aclnn": build_and_install_aclnn,
    "build_ext": cmake_build_ext,
    "install": custom_install,
}

setup(
    name=DISTRIBUTION_NAME,
    # Follow:
    # https://packaging.python.org/en/latest/specifications/version-specifiers
    version=VERSION,
    author="vLLM-Ascend team",
    license="Apache 2.0",
    description="vLLM Ascend backend plugin",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm-ascend",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm-ascend",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    python_requires=">=3.10",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require={},
    entry_points={
        "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
        "vllm.general_plugins": [
            "ascend_kv_connector = vllm_ascend:register_connector",
            "ascend_model_loader = vllm_ascend:register_model_loader",
            "ascend_service_profiling = vllm_ascend:register_service_profiling",
            "ascend_model = vllm_ascend:register_model",
        ],
    },
)
