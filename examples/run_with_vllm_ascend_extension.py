# SPDX-License-Identifier: Apache-2.0
"""Run a Python script with an isolated ``vllm_ascend_C`` candidate.

This keeps qualification builds out of the production package.  Each
``torchrun`` child loads the candidate before the requested script imports an
Ascend custom op, so runtime binding reports also identify the candidate DSO.
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path

import torch  # noqa: F401  # Load libtorch before the extension.
import torch_npu  # noqa: F401  # Register PrivateUse1 before custom ops.

import vllm_ascend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument(
        "--required-op",
        default="matmul_allreduce_add_rmsnorm",
        help="operator that the isolated extension must register",
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extension = args.extension.resolve(strict=True)
    module_name = "vllm_ascend.vllm_ascend_C"
    if module_name in sys.modules:
        raise RuntimeError(f"{module_name} was loaded before isolated injection")

    spec = importlib.util.spec_from_file_location(module_name, extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create extension spec for {extension}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    vllm_ascend.vllm_ascend_C = module
    spec.loader.exec_module(module)

    if not hasattr(torch.ops._C_ascend, args.required_op):
        raise RuntimeError(
            f"isolated extension did not register {args.required_op!r}"
        )

    script = args.script.resolve(strict=True)
    sys.argv = [str(script), *args.script_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
