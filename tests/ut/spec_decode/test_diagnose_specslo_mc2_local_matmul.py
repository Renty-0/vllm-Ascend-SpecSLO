# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for TP3 MC2 one-active-source diagnosis."""

from types import SimpleNamespace

import pytest
import torch

from examples import diagnose_specslo_mc2_local_matmul as diagnostic


def _rank_outputs(value: torch.Tensor) -> list[torch.Tensor]:
    return [value.clone() for _ in range(diagnostic.TP_SIZE)]


def test_single_source_exact_classification() -> None:
    source = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)

    classification, evidence = diagnostic._classify_single_source_case(
        source_local_projection=source,
        split_outputs_by_rank=_rank_outputs(source),
        mc2_outputs_by_rank=_rank_outputs(source),
    )

    assert classification == "exact"
    assert evidence["ordinary_hccl_exact_to_source_local"]
    assert evidence["mc2_rank_consensus"]
    assert evidence["mc2_exact_to_source_local"]


def test_single_source_isolates_local_matmul_or_prepublication_candidate() -> None:
    source = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    mc2 = torch.tensor([[1.0078125, -2.0]], dtype=torch.bfloat16)

    classification, evidence = diagnostic._classify_single_source_case(
        source_local_projection=source,
        split_outputs_by_rank=_rank_outputs(source),
        mc2_outputs_by_rank=_rank_outputs(mc2),
    )

    assert classification == "local_matmul_or_prepublication_candidate"
    assert evidence["ordinary_hccl_exact_to_source_local"]
    assert evidence["mc2_rank_consensus"]
    assert not evidence["mc2_exact_to_source_local"]
    assert evidence["mc2_vs_source_local_by_rank"][0]["mismatch_count"] == 1


def test_single_source_detects_mc2_rank_publication_divergence() -> None:
    source = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    divergent = _rank_outputs(source)
    divergent[2][0, 0] = 3.0

    classification, evidence = diagnostic._classify_single_source_case(
        source_local_projection=source,
        split_outputs_by_rank=_rank_outputs(source),
        mc2_outputs_by_rank=divergent,
    )

    assert classification == "mc2_rank_publication_or_reduction_failure"
    assert not evidence["mc2_rank_consensus"]


def test_single_source_rejects_bad_ordinary_hccl_reference() -> None:
    source = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    split = _rank_outputs(source)
    split[0][0, 1] = -3.0

    classification, evidence = diagnostic._classify_single_source_case(
        source_local_projection=source,
        split_outputs_by_rank=split,
        mc2_outputs_by_rank=_rank_outputs(source),
    )

    assert classification == "ordinary_hccl_reference_failure"
    assert not evidence["ordinary_hccl_exact_to_source_local"]


def test_zero_sentinel_rejects_stale_mc2_window() -> None:
    zero = torch.zeros((1, 2), dtype=torch.bfloat16)
    stale = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)

    classification, evidence = diagnostic._classify_zero_sentinel(
        split_outputs_by_rank=_rank_outputs(zero),
        mc2_outputs_by_rank=_rank_outputs(stale),
    )

    assert classification == "mc2_stale_window_or_nonzero_zero_control"
    assert evidence["ordinary_hccl_all_zero"]
    assert not evidence["mc2_all_zero"]


def test_zero_sentinel_requires_rank_consensus() -> None:
    zero = torch.zeros((1, 2), dtype=torch.bfloat16)
    divergent = _rank_outputs(zero)
    divergent[1][0, 0] = 1.0

    classification, evidence = diagnostic._classify_zero_sentinel(
        split_outputs_by_rank=_rank_outputs(zero),
        mc2_outputs_by_rank=divergent,
    )

    assert classification == "mc2_zero_rank_publication_failure"
    assert not evidence["mc2_rank_consensus"]


def test_cross_replay_gate_requires_stable_repeated_states_and_ab_change() -> None:
    state_a = _rank_outputs(torch.tensor([[1.0]], dtype=torch.bfloat16))
    state_b = _rank_outputs(torch.tensor([[2.0]], dtype=torch.bfloat16))

    evidence = diagnostic._cross_replay_stability(
        [state_a, state_b, _rank_outputs(state_a[0]), _rank_outputs(state_b[0])],
        ["A", "B", "A", "B"],
        require_state_transition=True,
    )

    assert evidence["same_state_bit_exact"]
    assert evidence["state_transition_observed"]
    assert evidence["passed"]


def test_cross_replay_gate_rejects_stale_b_state_and_unstable_a_state() -> None:
    state_a = _rank_outputs(torch.tensor([[1.0]], dtype=torch.bfloat16))
    state_b_stale = _rank_outputs(torch.tensor([[1.0]], dtype=torch.bfloat16))
    changed_a = _rank_outputs(torch.tensor([[3.0]], dtype=torch.bfloat16))

    evidence = diagnostic._cross_replay_stability(
        [state_a, state_b_stale, changed_a, state_b_stale],
        ["A", "B", "A", "B"],
        require_state_transition=True,
    )

    assert not evidence["same_state_bit_exact"]
    assert not evidence["state_transition_observed"]
    assert not evidence["passed"]


def test_source_order_rotates_rank_before_zero_sentinel() -> None:
    assert [diagnostic._source_order(index)[-2] for index in range(3)] == [2, 0, 1]
    assert all(
        diagnostic._source_order(index)[-1] == diagnostic.ZERO_SENTINEL
        for index in range(3)
    )


def _payload(dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    return {
        "activation": torch.ones((2, 3), dtype=dtype),
        "weight": torch.ones((4, 3), dtype=dtype),
        "residual": torch.ones((2, 4), dtype=dtype),
        "gamma": torch.ones((4,), dtype=dtype),
    }


def test_payload_validation_requires_bf16_finite_tensors() -> None:
    shapes = {
        "activation": (2, 3),
        "weight": (4, 3),
        "residual": (2, 4),
        "gamma": (4,),
    }

    result = diagnostic._validate_payload(_payload(), source="capture", expected_shapes=shapes)
    assert set(result) == set(shapes)
    with pytest.raises(ValueError, match="requires BF16"):
        diagnostic._validate_payload(
            _payload(torch.float32), source="capture", expected_shapes=shapes
        )
    non_finite = _payload()
    non_finite["activation"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        diagnostic._validate_payload(
            non_finite, source="capture", expected_shapes=shapes
        )


def test_tensor_comparison_records_exact_mismatch_coordinate() -> None:
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    actual = expected.clone()
    actual[1, 0] = 3.5

    evidence = diagnostic._tensor_comparison(actual, expected)

    assert not evidence["exact_match"]
    assert evidence["mismatch_count"] == 1
    assert evidence["first_mismatch"]["coordinate"] == [1, 0]
    assert evidence["worst_mismatch"]["coordinate"] == [1, 0]
    assert evidence["mismatch_coordinates_sample"] == [[1, 0]]
    assert not evidence["mismatch_coordinates_truncated"]
    assert evidence["max_abs_error"] == 0.5


def test_validate_args_rejects_single_replay_and_duplicate_modes() -> None:
    valid = dict(
        m=100,
        k=8576,
        n=5120,
        replays=4,
        state_delta=0.015625,
        execution_modes=["eager", "graph"],
    )

    diagnostic._validate_args(SimpleNamespace(**valid))
    with pytest.raises(ValueError, match="at least four replays"):
        diagnostic._validate_args(SimpleNamespace(**{**valid, "replays": 1}))
    with pytest.raises(ValueError, match="finite and non-zero"):
        diagnostic._validate_args(SimpleNamespace(**{**valid, "state_delta": 0.0}))
    with pytest.raises(ValueError, match="must not contain duplicates"):
        diagnostic._validate_args(
            SimpleNamespace(**{**valid, "execution_modes": ["eager", "eager"]})
        )
