# SPDX-License-Identifier: Apache-2.0
"""CPU checks for MC2 qualification profile runtime identity."""

from types import SimpleNamespace

import pytest

import vllm_ascend.spec_decode.pearl.mc2 as mc2
from examples import measure_specslo_mc2 as measurement


def _measurement_args(*extra: str):
    return measurement._parser().parse_args(
        ["--m", "64", "--k", "3072", "--n", "5120", "--output", "/tmp/mc2.json", *extra]
    )


def test_mc2_measurement_defaults_to_changed_input_qualification():
    args = _measurement_args()

    measurement._validate_args(args)

    assert args.changed_input_replays == 8
    assert args.changed_input_delta != 0.0


@pytest.mark.parametrize(
    "extra",
    (
        ("--changed-input-replays", "1"),
        ("--changed-input-replays", "8", "--changed-input-delta", "0"),
    ),
)
def test_mc2_measurement_rejects_ineffective_changed_input_qualification(extra):
    with pytest.raises(ValueError, match="at least two replays"):
        measurement._validate_args(_measurement_args(*extra))


def test_changed_input_liveness_offsets_are_bounded_and_change_each_replay():
    offsets = [measurement._changed_input_offset_units(index) for index in range(1024)]

    assert offsets[:10] == [0, 1, -1, 2, -2, 0, 1, -1, 2, -2]
    assert max(abs(offset) for offset in offsets) == 2
    assert all(left != right for left, right in zip(offsets, offsets[1:]))


def test_changed_input_liveness_offset_rejects_negative_replay():
    with pytest.raises(ValueError, match="non-negative"):
        measurement._changed_input_offset_units(-1)


def test_standalone_tp3_visible_devices_record_physical_rank_order(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "5,6,7")

    assert tuple(measurement._physical_device_id_for_local_rank(rank, 3) for rank in range(3)) == (
        "5",
        "6",
        "7",
    )


def test_four_card_process_mapping_resolves_target_subgroup(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")

    assert tuple(measurement._physical_device_id_for_local_rank(rank, 4) for rank in (1, 2, 3)) == (
        "5",
        "6",
        "7",
    )


def test_visible_device_mapping_rejects_missing_local_rank(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "5,6")

    with pytest.raises(ValueError, match="does not provide a physical NPU"):
        measurement._physical_device_id_for_local_rank(2, 3)


def test_tp_mapping_is_collected_in_process_group_rank_order(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "5,6,7")
    monkeypatch.setattr(measurement.dist, "get_world_size", lambda: 3)

    def gather(output, _local):
        output[:] = ["5", "6", "7"]

    monkeypatch.setattr(measurement.dist, "all_gather_object", gather)

    assert measurement._tp_rank_device_mapping(1) == ("5", "6", "7")


def test_standalone_profile_binding_matches_four_card_target_subgroup(monkeypatch):
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "5,6,7")
    monkeypatch.setattr(mc2, "active_mc2_vendor_payload_sha256", lambda: "a" * 64)
    monkeypatch.setattr(mc2, "active_mc2_adapter_binary_sha256", lambda: "c" * 64)
    monkeypatch.setattr(
        mc2,
        "active_mc2_opapi_symbol_provider_sha256",
        lambda: {
            symbol: ("e" * 64 if index < 2 else None)
            for index, symbol in enumerate(mc2._MC2_OPAPI_SYMBOLS)
        },
    )
    standalone_mapping = tuple(
        measurement._physical_device_id_for_local_rank(rank, 3) for rank in range(3)
    )
    binding = mc2.current_mc2_runtime_binding(
        tp_rank_device_mapping=standalone_mapping,
    )

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    metadata = {"runtime_binding": binding}
    for tp_rank, logical_device in enumerate((1, 2, 3)):
        device = SimpleNamespace(type="npu", index=logical_device)
        assert mc2._runtime_binding_mismatch(metadata, device, tp_rank_id=tp_rank) is None
