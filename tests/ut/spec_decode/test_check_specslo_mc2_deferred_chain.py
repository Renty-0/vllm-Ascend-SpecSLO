# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the deferred TP3 MC2 chain diagnostic."""

from __future__ import annotations

import pytest

from examples import check_specslo_mc2_deferred_chain as diagnostic


def _args(*extra: str):
    return diagnostic._parser().parse_args(
        [
            "--attention-input-dir",
            "/tmp/attention",
            "--down-input-dir",
            "/tmp/down",
            "--execution-mode",
            "graph",
            "--output",
            "/tmp/result.json",
            *extra,
        ]
    )


def test_deferred_chain_diagnostic_defaults_have_repeated_liveness_checks():
    args = _args()

    diagnostic._validate_args(args)

    assert args.changed_input_replays == 5
    assert args.changed_input_delta != 0.0
    assert args.samples >= 3
    assert args.layer_count == 1


@pytest.mark.parametrize(
    "extra,match",
    (
        (("--samples", "2"), ">=3 samples"),
        (("--replays-per-sample", "0"), "positive replay"),
        (("--changed-input-replays", "1"), "at least two"),
        (("--changed-input-delta", "0"), "finite and non-zero"),
        (("--layer-count", "0"), "layer-count must be positive"),
    ),
)
def test_deferred_chain_diagnostic_rejects_weak_qualification(extra, match):
    with pytest.raises(ValueError, match=match):
        diagnostic._validate_args(_args(*extra))


def test_deferred_chain_percentile_interpolates_sorted_samples():
    values = [9.0, 1.0, 5.0]

    assert diagnostic._percentile(values, 50.0) == 5.0
    assert diagnostic._percentile(values, 95.0) == pytest.approx(8.6)


def test_deferred_chain_records_physical_tp_order(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,2,4")
    monkeypatch.setattr(diagnostic.dist, "get_world_size", lambda: 3)

    def gather(output, _local):
        output[:] = ["1", "2", "4"]

    monkeypatch.setattr(diagnostic.dist, "all_gather_object", gather)

    assert diagnostic._device_mapping(1) == ("1", "2", "4")
