# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OP_API_COMMON = REPO_ROOT / "csrc/aclnn_torch_adapter/op_api_common.h"


def _exec_npu_cmd_macro() -> str:
    source = OP_API_COMMON.read_text(encoding="utf-8")
    start = source.index("#define EXEC_NPU_CMD")
    end = source.index("\n\n#endif", start)
    return source[start:end]


def test_exec_npu_cmd_keeps_workspace_owner_alive_through_launch():
    macro = _exec_npu_cmd_macro()

    declaration = macro.index("at::Tensor workspace_tensor;")
    allocation_branch = macro.index("if (workspace_size != 0)")
    allocation = macro.index("workspace_tensor = at::empty")
    pointer = macro.index("workspace_tensor.storage().data()")
    handler = macro.index("auto acl_call =")
    launch = macro.index("cmd.Run()")

    assert declaration < allocation_branch < allocation < pointer < handler < launch
    assert "auto workspace_tensor =" not in macro
    assert "executor, workspace_tensor" in macro
