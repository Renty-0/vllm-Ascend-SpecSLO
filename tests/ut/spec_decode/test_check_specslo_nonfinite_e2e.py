# SPDX-License-Identifier: Apache-2.0
"""CPU CLI/report/injection checks; never claims actual NPU acceptance."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from examples import check_specslo_nonfinite_e2e as probe


def _args(*extra):
    return probe._build_parser().parse_args(["--output", "unused.json", *extra])


def _records(healthy=True, fault_rank=2):
    rows = []
    for rank in range(4):
        row = {
            "rank": rank,
            "role": "draft" if rank == 0 else "target",
            "model_forward_calls": 1,
            "cache_allocation_released": True,
            "nonfinite_flag": not healthy and rank == fault_rank,
            "nonfinite_flag_before_generate": not healthy and rank == fault_rank,
            "error": None
            if healthy
            else (
                "SpecRhythm prefill produced nonfinite Q/K/V or output on a rank; "
                "no first token was committed; discard this cache"
            ),
            "outputs": [[7], [8]] if healthy and rank == 1 else None,
            "chunks": [{"request_index": 0, "token_ids": [7]}, {"request_index": 1, "token_ids": [8]}]
            if healthy and rank == 1
            else [],
            "vocabulary_size": 32,
        }
        rows.append(row)
    return rows


def test_default_is_real_target_head_fault_with_one_engine_and_first_token_boundary():
    args = _args()
    assert args.fault_rank == 2
    assert probe._validate_args(args) == "head"
    config = probe._engine_config_values(args)
    assert config["draft_tp_size"] == 1 and config["target_tp_size"] == 3
    assert config["max_tokens"] == 1 and config["max_num_seqs"] == 2
    assert config["enable_spec_rhythm"] and config["spec_rhythm_online_prefill"]
    assert config["spec_rhythm_tree_width"] * config["spec_rhythm_tree_depth"] > 1
    assert config["enforce_eager"] and not config["enable_prefix_caching"]


def test_draft_fault_targets_a_layer_actually_executed_during_one_token_prefill():
    assert probe._validate_args(_args("--fault-rank", "0")) == "final-mlp"
    with pytest.raises(ValueError, match="does not execute its head"):
        probe._validate_args(_args("--fault-rank", "0", "--fault-site", "head"))


@pytest.mark.parametrize(
    "options",
    [
        ["--max-model-len", "32"],
        ["--gpu-memory-utilization", "1"],
        ["--gpu-memory-utilization", "nan"],
        ["--collective-timeout-seconds", "0"],
        ["--collective-timeout-seconds", "inf"],
        ["--seed", "-1"],
    ],
)
def test_invalid_bounds_are_rejected_before_model_loading(options):
    with pytest.raises(ValueError):
        probe._validate_args(_args(*options))


def test_dry_run_writes_honest_manifest_without_importing_runtime(tmp_path):
    output = tmp_path / "manifest.json"
    assert probe.main(["--dry-run", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "dry_run" and report["passed"] is None
    assert not report["npu_executed"] and report["engine_loads_per_rank"] == 0
    assert report["stages"] == []
    assert report["prompt_count"] == 2 and report["max_tokens"] == 1
    assert not report["collective_close_barrier_completed"]
    assert not report["destroy_process_group_returned"]


def test_real_weight_scalar_injection_and_restore_preserve_other_parameters():
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float16).view(3, 4))
    original = parameter.detach().clone()
    identity, pointer = id(parameter), parameter.data_ptr()
    saved = probe._weight_scalar(parameter)
    probe._replace_weight_scalar(parameter, float("nan"))
    assert id(parameter) == identity and parameter.data_ptr() == pointer
    assert torch.isnan(parameter[0, 0])
    assert torch.equal(parameter.flatten()[1:], original.flatten()[1:])
    probe._replace_weight_scalar(parameter, saved)
    assert torch.equal(parameter, original)
    assert saved.dtype == torch.float16


def test_head_injection_must_be_inside_real_active_vocabulary():
    engine = SimpleNamespace(
        model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros(2, 2), vocab_start=4)), draft_vocab_size=4
    )
    with pytest.raises(ValueError, match="non-truncated"):
        probe._weight_target(engine, "head")
    engine.model.lm_head.vocab_start = 2
    parameter, name, token = probe._weight_target(engine, "head")
    assert parameter is engine.model.lm_head.weight and name == "lm_head.weight" and token == 2


@pytest.mark.parametrize("stage", ["healthy", "poisoned", "restored_sticky"])
def test_stage_acceptance_requires_all_four_rank_records(stage):
    records = _records(healthy=stage == "healthy")
    assert probe._summarize_stage(stage, records, fault_rank=2)["passed"]
    assert not probe._summarize_stage(stage, records[:-1], fault_rank=2)["passed"]


@pytest.mark.parametrize("fault", ["rank_succeeded", "chunk", "false_flag", "no_forward", "wrong_error", "live_cache"])
def test_negative_stage_cannot_pass_vacuously_or_after_any_token_delivery(fault):
    records = _records(False)
    if fault == "rank_succeeded":
        records[3]["error"] = None
    elif fault == "chunk":
        records[1]["chunks"] = [{"request_index": 0, "token_ids": [7]}]
    elif fault == "false_flag":
        records[2]["nonfinite_flag"] = False
    elif fault == "no_forward":
        records[0]["model_forward_calls"] = 0
    elif fault == "wrong_error":
        records[2]["error"] = "unrelated shape error"
    else:
        records[0]["cache_allocation_released"] = False
    assert not probe._summarize_stage("poisoned", records, fault_rank=2)["passed"]


def test_healthy_control_requires_exact_stream_concatenation_and_valid_tokens():
    original = _records()
    changed = deepcopy(original)
    changed[1]["chunks"][0]["token_ids"] = [6]
    assert not probe._summarize_stage("healthy", changed, fault_rank=2)["passed"]
    changed = deepcopy(original)
    changed[1]["vocabulary_size"] = 7
    assert not probe._summarize_stage("healthy", changed, fault_rank=2)["passed"]
