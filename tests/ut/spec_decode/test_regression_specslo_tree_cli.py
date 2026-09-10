# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the real-NPU regression script's inputs and verdicts."""

import json
from types import SimpleNamespace

import pytest

from examples.regression_specslo_tree import (
    _baseline_compare,
    _boundary_rows,
    _build_parser,
    _compare_tokens,
    _encode_rows,
    _graph_deltas,
    _stream_check,
    _validate_args,
    main,
)


def _args(*extra):
    return _build_parser().parse_args(["--gsm8k", "/test.parquet", "--output", "/report.json", *extra])


def test_regression_defaults_cover_b1_online_queue_long_decode_and_mode_pairs():
    args = _args()
    _validate_args(args)
    assert args.verification_budgets == [1, 4]
    assert args.max_tokens == [32, 64]
    assert args.modes == ["eager", "graph"]
    assert args.num_prompts > args.batch_size
    assert args.repeats == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--num-prompts", "4"],
        ["--verification-budgets", "0"],
        ["--max-tokens", "0"],
        ["--tree-width", "1", "--tree-depth", "1"],
        ["--arrival-lead", "-1"],
        ["--baseline-inputs", "/unpaired.json"],
    ],
)
def test_regression_rejects_invalid_or_non_online_cases(extra):
    with pytest.raises(ValueError):
        _validate_args(_args(*extra))


def test_boundary_probes_have_exact_page_edge_lengths_and_keep_assistant_suffix():
    prompts = [[5, 6, 7], [1, 2, 3, 4, 5, 6]]
    result = _boundary_rows(prompts, [127, 4], filler_token_id=9)
    assert list(map(len, result)) == [127, 4]
    assert result[0][-3:] == [5, 6, 7]
    assert result[1] == [3, 4, 5, 6]
    assert prompts == [[5, 6, 7], [1, 2, 3, 4, 5, 6]]


def test_manifest_token_ids_do_not_retokenize_or_modify_prompts():
    result = _encode_rows([{"prompt_token_ids": [4, 5, 6]}], tokenizer=None)
    assert result == [[4, 5, 6]]


def test_token_comparison_reports_first_difference_and_short_reference():
    result = _compare_tokens([[1, 2, 3]], [[1, 4, 3]])
    assert not result["exact_match"]
    assert result["mismatches"][0]["first_difference"] == 1
    assert _compare_tokens([[1, 2]], [[1, 2, 3]], prefix_only=True)["exact_match"]
    assert not _compare_tokens([[1, 2, 3]], [[1, 2]], prefix_only=True)["exact_match"]


def test_stream_validation_matches_chunks_by_public_id_not_cumulative_index():
    events = [
        {
            "request_id": "case:1",
            "request_index": 19,
            "token_ids": [5],
            "finished": True,
            "elapsed_seconds": 0.1,
            "host_received_wall_time": 11.0,
        },
        {
            "request_id": "case:0",
            "request_index": 18,
            "token_ids": [2],
            "finished": False,
            "elapsed_seconds": 0.1,
            "host_received_wall_time": 11.0,
        },
        {
            "request_id": "case:0",
            "request_index": 18,
            "token_ids": [3],
            "finished": True,
            "elapsed_seconds": 0.2,
            "host_received_wall_time": 11.1,
        },
    ]
    result = _stream_check(events, ["case:0", "case:1"], [[2, 3], [5]], {"case:0": 10, "case:1": 10})
    assert result["passed"]


def test_stream_validation_rejects_tokens_after_finish_or_before_arrival():
    events = [
        {"request_id": "r", "token_ids": [1], "finished": True, "elapsed_seconds": 0.1, "host_received_wall_time": 9.0},
        {
            "request_id": "r",
            "token_ids": [2],
            "finished": True,
            "elapsed_seconds": 0.2,
            "host_received_wall_time": 11.0,
        },
    ]
    result = _stream_check(events, ["r"], [[1, 2]], {"r": 10})
    assert not result["passed"]
    assert result["missing_or_duplicate_finished"] == ["r"]
    assert len(result["errors"]) == 2


def test_graph_delta_does_not_call_capture_first_replay_reuse():
    before = [{"rank": 1, "aclgraph_captures": 2, "aclgraph_replays": 9}]
    after = [{"rank": 1, "aclgraph_captures": 3, "aclgraph_replays": 10}]
    assert _graph_deltas(before, after)[0]["reused_graph_replays"] == 0
    after[0]["aclgraph_replays"] = 12
    assert _graph_deltas(before, after)[0]["reused_graph_replays"] == 2


def test_legacy_baseline_first_prompt_match_is_not_all_input_identity():
    baseline = {
        "first_prompt_token_ids": [1, 2],
        "respect_eos": False,
        "results": [{"output_token_ids": [[3, 4], [5, 6]]}],
    }
    result = _baseline_compare(baseline, None, [[1, 2], [8, 9]], [[3], [5]], False)
    assert result["token_comparison"]["exact_match"]
    assert not result["all_inputs_verified"]
    assert result["status"] == "legacy_first_input_only_unverified_comparison"
    verified = _baseline_compare(baseline, [[1, 2], [8, 9]], [[1, 2], [8, 9]], [[3], [5]], False)
    assert verified["all_inputs_verified"]
    assert verified["status"] == "compared"


def test_baseline_comparison_rejects_different_inputs_and_eos_policy():
    baseline = {"first_prompt_token_ids": [1, 2], "respect_eos": False, "results": [{"output_token_ids": [[3, 4]]}]}
    assert _baseline_compare(baseline, None, [[8, 9]], [[3]], False)["status"] == "different_or_unproven_inputs"
    assert _baseline_compare(baseline, None, [[1, 2]], [[3]], True)["status"] == "different_eos_policy"


def test_cpu_script_orchestration_reuses_engine_within_mode_and_budget(monkeypatch, tmp_path):
    import transformers

    from vllm_ascend.spec_decode.pearl import api

    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt_token_ids":[1,2]}\n{"prompt_token_ids":[3,4]}\n')
    output = tmp_path / "result.json"
    engines = []

    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self.requests = []
            self.last_worker_metrics = []
            self.calls = 0
            self.closed = False
            engines.append(self)

        def add_request(self, tokens, params):
            self.requests.append((tokens, params))

        def generate(self, *, on_token_commit):
            self.calls += 1
            self.last_metrics = []
            for index, (_, params) in enumerate(self.requests):
                tokens = [7] * params.max_tokens
                on_token_commit(
                    {
                        "request_id": params.request_id,
                        "request_index": index,
                        "token_ids": tokens,
                        "finished": True,
                        "elapsed_seconds": 0.1,
                    }
                )
                self.last_metrics.append(
                    {
                        "completion_token_ids": tokens,
                        "spec_rhythm": {
                            "spec_rhythm_peak_active_requests": self.config.max_num_seqs,
                            "spec_rhythm_prefill_requests": len(self.requests),
                            "spec_rhythm_peak_verify_candidates": self.config.spec_rhythm_verification_budget,
                        },
                    }
                )
            graph = not self.config.enforce_eager
            self.last_worker_metrics = [
                {
                    "rank": 1,
                    "is_draft_rank": 0,
                    "aclgraph_captures": int(graph),
                    "aclgraph_replays": 2 * self.calls if graph else 0,
                }
            ]
            self.requests = []

        def exit(self):
            self.closed = True

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "PEARLConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(api, "PEARLEngine", FakeEngine)
    code = main(
        [
            "--manifest",
            str(source),
            "--output",
            str(output),
            "--num-prompts",
            "2",
            "--batch-size",
            "1",
            "--max-tokens",
            "2",
            "3",
            "--verification-budgets",
            "1",
            "4",
            "--repeats",
            "2",
            "--no-boundary-probes",
            "--arrival-lead",
            "0",
            "--arrival-interval",
            "0",
        ]
    )
    report = json.loads(output.read_text())
    assert code == 0
    assert report["status"] == "passed"
    assert len(engines) == 4
    assert all(engine.calls == 4 and engine.closed for engine in engines)
    assert len(report["runs"]) == 16
    assert all(run["stream_check"]["passed"] for run in report["runs"])
