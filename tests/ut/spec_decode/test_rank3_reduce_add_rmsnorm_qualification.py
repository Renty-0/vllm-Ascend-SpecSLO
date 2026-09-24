# SPDX-License-Identifier: Apache-2.0
"""CPU/static contracts for the rank-3 reduce/AddRMSNorm qualification path."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from examples import measure_specslo_rank3_reduce_add_rmsnorm as measurement

REPO_ROOT = Path(__file__).resolve().parents[3]
MEASUREMENT = REPO_ROOT / "examples/measure_specslo_rank3_reduce_add_rmsnorm.py"
EXTENSION_RUNNER = REPO_ROOT / "examples/run_with_vllm_ascend_extension.py"
TORCH_BINDING = REPO_ROOT / "csrc/torch_binding.cpp"
TORCH_BINDING_META = REPO_ROOT / "csrc/torch_binding_meta.cpp"
TORCH_ADAPTER = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/allreduce_add_rmsnorm_torch_adpt.h"
)
OP_API_COMMON = REPO_ROOT / "csrc/aclnn_torch_adapter/op_api_common.h"
EPILOGUE_KERNEL = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_kernel/allreduce_add_rmsnorm_aiv_kernel.h"
)
EPILOGUE_ENTRY = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_kernel/allreduce_add_rmsnorm.cpp"
)
EPILOGUE_TILING = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_kernel/allreduce_add_rmsnorm_tiling.h"
)
EPILOGUE_TILER = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_host/allreduce_add_rmsnorm_tiling.cpp"
)
EPILOGUE_DEF = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_host/allreduce_add_rmsnorm_def.cpp"
)
EPILOGUE_PROTO = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_host/allreduce_add_rmsnorm_proto.cpp"
)
EPILOGUE_OP_API = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_host/op_api/aclnn_allreduce_add_rmsnorm.cpp"
)


def _args(*extra: str):
    return measurement._parser().parse_args(
        [
            "--m",
            "100",
            "--k",
            "3072",
            "--n",
            "5120",
            "--input-dir",
            "/tmp/inputs",
            "--output",
            "/tmp/result.json",
            *extra,
        ]
    )


def _assigned_dict(tree: ast.AST, name: str) -> ast.Dict:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and isinstance(node.value, ast.Dict)
        ):
            return node.value
    raise AssertionError(f"could not find dict assigned to {name!r}")


def _dict_entries(node: ast.Dict) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            result[key.value] = value
    return result


def test_measurement_defaults_to_graph_and_one_operation():
    args = _args()

    measurement._validate_args(args)

    assert args.execution_mode == "graph"
    assert args.operations_per_graph == 1


@pytest.mark.parametrize("mode", ("eager", "graph"))
@pytest.mark.parametrize("operations", (1, 4, 64))
def test_measurement_accepts_both_execution_modes_and_positive_operation_counts(
    mode: str,
    operations: int,
):
    args = _args(
        "--execution-mode",
        mode,
        "--operations-per-graph",
        str(operations),
    )

    measurement._validate_args(args)

    assert args.execution_mode == mode
    assert args.operations_per_graph == operations


@pytest.mark.parametrize("operations", (0, -1))
def test_measurement_rejects_non_positive_operation_counts(operations: int):
    args = _args("--operations-per-graph", str(operations))

    with pytest.raises(ValueError, match="operations-per-graph must be positive"):
        measurement._validate_args(args)


def test_measurement_runs_the_requested_operation_count_on_both_paths():
    source = MEASUREMENT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    nested = {
        node.name: node
        for node in main.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for name in ("baseline", "fused"):
        loops = [node for node in ast.walk(nested[name]) if isinstance(node, ast.For)]
        assert len(loops) == 1
        assert ast.unparse(loops[0].iter) == "range(args.operations_per_graph)"


def test_result_schema_records_mode_operation_count_and_per_operation_latency():
    tree = ast.parse(MEASUREMENT.read_text(encoding="utf-8"))
    document = _dict_entries(_assigned_dict(tree, "document"))

    assert ast.literal_eval(document["schema_version"]) == 1
    assert ast.literal_eval(document["status"]) == "measured"
    assert {
        "schema_version",
        "status",
        "metadata",
        "shape",
        "correctness",
        "samples",
        "summary",
    } <= document.keys()

    metadata = _dict_entries(document["metadata"])
    assert ast.unparse(metadata["execution_mode"]) == "args.execution_mode"
    assert ast.unparse(metadata["operations_per_graph"]) == "args.operations_per_graph"
    assert ast.literal_eval(metadata["tensor_parallel_size"]) == 3

    correctness = _dict_entries(document["correctness"])
    assert {
        "exact_all_ranks",
        "max_abs_norm",
        "max_abs_added",
        "changed_input_exact_all_ranks",
        "changed_input_max_abs_norm",
        "changed_input_max_abs_added",
    } <= correctness.keys()

    summary = _dict_entries(document["summary"])
    for path in ("baseline", "fused"):
        for percentile in ("p50", "p95"):
            key = f"{path}_per_operation_{percentile}_ms"
            expression = ast.unparse(summary[key])
            assert expression == f"{path}_{percentile} / args.operations_per_graph"


def test_measurement_restores_original_input_before_hashing_graph_outputs():
    source = MEASUREMENT.read_text(encoding="utf-8")
    restore = source.index("x.copy_(initial)", source.index("for replay in range"))
    replay = source.index("fused_output = fused_runner()", restore)
    hashing = source.index('"fused_norm_sha256": _sha256(fused_output[0])', replay)

    assert restore < replay < hashing


def test_torch_schema_meta_and_adapter_preserve_double_epsilon_abi():
    binding = TORCH_BINDING.read_text(encoding="utf-8")
    meta = TORCH_BINDING_META.read_text(encoding="utf-8")
    adapter = TORCH_ADAPTER.read_text(encoding="utf-8")

    schema = re.search(
        r'ops\.def\("allreduce_add_rmsnorm\((.*?)\)\s*\\?\s*"?\s*->',
        binding,
        flags=re.DOTALL,
    )
    assert schema is not None
    assert "float epsilon" in schema.group(1)

    meta_start = meta.index(" allreduce_add_rmsnorm_meta(")
    meta_signature = meta[meta_start : meta.index("\n{", meta_start)]
    adapter_start = adapter.index(" allreduce_add_rmsnorm(")
    adapter_signature = adapter[adapter_start : adapter.index("\n{", adapter_start)]
    signature_tail = re.compile(
        r"int64_t tp_rank_id,\s*double epsilon,\s*bool is_gather_add_out\)",
        flags=re.DOTALL,
    )
    assert signature_tail.search(meta_signature)
    assert signature_tail.search(adapter_signature)
    assert "float epsilon" not in meta_signature
    assert "float epsilon" not in adapter_signature


def test_adapter_forwards_epsilon_in_the_aclnn_abi_position():
    adapter = TORCH_ADAPTER.read_text(encoding="utf-8")
    invocation = re.search(
        r"EXEC_NPU_CMD\(aclnnAllreduceAddRmsnorm,\s*(.*?)\);",
        adapter,
        flags=re.DOTALL,
    )
    assert invocation is not None
    arguments = [part.strip() for part in invocation.group(1).split(",")]
    assert arguments == [
        "local_projection",
        "residual",
        "gamma",
        "chain_state",
        "group_tp_ptr",
        "tp_rank_size",
        "tp_rank_id",
        "epsilon",
        "is_gather_add_out",
        "flush_chain",
        "output",
        "add_out",
        "next_chain_state",
    ]


def test_chained_torch_schema_meta_and_adapter_have_three_outputs():
    binding = TORCH_BINDING.read_text(encoding="utf-8")
    meta = TORCH_BINDING_META.read_text(encoding="utf-8")
    adapter = TORCH_ADAPTER.read_text(encoding="utf-8")

    chained_schema = re.search(
        r'ops\.def\("allreduce_add_rmsnorm_chained\((.*?)\)\s*\\?\s*"?\s*->\s*'
        r'\(Tensor output, Tensor add_out, Tensor next_chain_state\)',
        binding,
        flags=re.DOTALL,
    )
    assert chained_schema is not None
    schema_args = chained_schema.group(1)
    assert "Tensor chainState" in schema_args
    assert "bool flushChain" in schema_args
    assert "float epsilon" in schema_args
    assert 'ops.impl("allreduce_add_rmsnorm_chained", torch::kPrivateUse1' in binding
    assert " allreduce_add_rmsnorm_chained_meta(" in meta
    assert 'ops.impl("allreduce_add_rmsnorm_chained"' in meta
    assert "std::tuple<at::Tensor, at::Tensor, at::Tensor> allreduce_add_rmsnorm_chained(" in adapter
    assert "at::Tensor next_chain_state = at::empty_like(chain_state);" in adapter
    assert "return {output, add_out, next_chain_state};" in adapter


def test_legacy_two_output_adapter_uses_zero_state_and_flushes():
    adapter = TORCH_ADAPTER.read_text(encoding="utf-8")
    legacy_start = adapter.index(" allreduce_add_rmsnorm(")
    legacy = adapter[legacy_start:]

    assert "at::Tensor chain_state = at::zeros(" in legacy
    chained_call = legacy[legacy.index("allreduce_add_rmsnorm_chained(") :]
    assert re.search(r"is_gather_add_out,\s*true\);", chained_call)
    assert "return {std::get<0>(chained), std::get<1>(chained)};" in legacy


def test_chained_op_def_proto_tiler_and_kernel_entry_share_state_abi():
    op_def = EPILOGUE_DEF.read_text(encoding="utf-8")
    proto = EPILOGUE_PROTO.read_text(encoding="utf-8")
    tiler = EPILOGUE_TILER.read_text(encoding="utf-8")
    entry = EPILOGUE_ENTRY.read_text(encoding="utf-8")
    op_api = EPILOGUE_OP_API.read_text(encoding="utf-8")
    tiling = EPILOGUE_TILING.read_text(encoding="utf-8")

    assert 'this->Input("chain_state")' in op_def
    assert 'this->Output("next_chain_state")' in op_def
    assert '.DataType({ge::DT_INT64})' in op_def
    assert 'this->Attr("flush_chain").AttrType(OPTIONAL).Bool(true);' in op_def
    assert "kChainStateIndex = 3" in proto
    assert "CloneShape(chainState, context->GetOutputShape(2));" in proto
    assert "ATTR_FLUSH_CHAIN" in tiler
    assert "chainState->GetDataType() != ge::DT_INT64" in tiler
    assert "chainStateShape.GetDim(0) != kChainStateRows" in tiler
    assert "chainStateShape.GetDim(1) != kChainStateWords" in tiler
    assert "pp.flushChain = *flushChainPtr ? 1 : 0;" in tiler
    assert "int32_t flushChain = 1;" in tiling
    assert "GM_ADDR chain_state" in entry
    assert "GM_ADDR next_chain_state" in entry
    for name in ("chainState", "flushChain", "nextChainState"):
        assert name in op_api


def test_chained_kernel_defers_only_tail_wait_and_flushes_to_zero():
    source = EPILOGUE_KERNEL.read_text(encoding="utf-8")
    process_start = source.index("__aicore__ inline void Process(")
    process_end = source.index("\n\nprivate:", process_start)
    process = source[process_start:process_end]

    wait_previous = process.index("WaitPreviousConsumersDone();")
    publish_request = process.index("PublishChunkRequests(request_token);")
    copy_payload = process.index("CopyLocalProjectionToWindow();")
    compute = process.index("ParallelWithSplitStepOneAddNorm(")
    publish_done = process.index("PublishChunkReadDone(peer_producer_tokens);")
    flush_branch = process.index("if (flush_chain)")
    wait_current = process.index("WaitChunkConsumersDone(producer_token);", flush_branch)
    clear_state = process.index("ClearNextChainState();", wait_current)
    defer_state = process.index("StoreNextChainToken(producer_token);", clear_state)
    clear_inactive = process.index("ClearInactiveNextChainState();", defer_state)

    assert wait_previous < publish_request < copy_payload < compute
    assert compute < publish_done < flush_branch < wait_current < clear_state < defer_state < clear_inactive
    assert "TP3_CHAIN_STATE_WORDS = 4" in source
    assert "ctrl[1] = ~producer_token;" in source
    assert "ctrl[2] = producer_token ^ salt;" in source
    assert "ctrl[3] = ~ctrl[2];" in source
    assert "record_idx < TP3_AIV_ONLY_MAX_CORES" in source
    assert "record_idx = core_num + core_idx" in source
    assert "record_idx += core_num" in source
    assert "if (core_num >= TP3_AIV_ONLY_MAX_CORES)" in source


def test_exec_npu_cmd_retains_workspace_owner_through_deferred_launch():
    source = OP_API_COMMON.read_text(encoding="utf-8")
    start = source.index("#define EXEC_NPU_CMD")
    macro = source[start : source.index("\n\n#endif", start)]

    declaration = macro.index("at::Tensor workspace_tensor;")
    allocation = macro.index("workspace_tensor = at::empty")
    pointer = macro.index("workspace_tensor.storage().data()")
    capture = macro.index("executor, workspace_tensor")
    launch = macro.index("cmd.Run()")

    assert declaration < allocation < pointer < capture < launch
    assert "auto workspace_tensor =" not in macro
    assert "(void)workspace_tensor;" in macro
    assert "std::numeric_limits<int64_t>::max()" in macro
    assert "static_cast<int64_t>(workspace_size)" in macro


def test_isolated_extension_runner_checks_required_op_before_running_script():
    source = EXTENSION_RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]

    load_index = source.index("spec.loader.exec_module(module)")
    check_index = source.index("hasattr(torch.ops._C_ascend, args.required_op)")
    argv_index = source.index("sys.argv = [str(script), *args.script_args]")
    run_index = source.index('runpy.run_path(str(script), run_name="__main__")')

    assert load_index < check_index < argv_index < run_index
    assert "importlib.util.spec_from_file_location" in calls
    assert "runpy.run_path" in calls


def test_tp3_epilogue_elides_only_the_redundant_self_mailbox_edge():
    source = EPILOGUE_KERNEL.read_text(encoding="utf-8")

    assert "return peer_rank != rank;" in source
    assert source.count("if (!IsRemotePeer(") == 5
    assert "peer_producer_tokens[rank] = producer_token;" in source
    assert "source_ready[rank] = true;" in source
    assert "consumer_received[rank] = true;" in source
    assert "int32_t ready_count = 1;" in source
    assert "int32_t received_count = 1;" in source

    # Cross-rank overwrite safety remains mandatory: every remote reader still
    # publishes READ_DONE and every remote producer still waits for it.
    assert "PublishChunkReadDone(peer_producer_tokens);" in source
    assert "WaitChunkConsumersDone(producer_token);" in source
    assert "Tp3ChunkSlot(source, TP3_AIV_READ_DONE, rank)" in source
    assert "Tp3ChunkSlot(rank, TP3_AIV_READ_DONE, consumer)" in source
