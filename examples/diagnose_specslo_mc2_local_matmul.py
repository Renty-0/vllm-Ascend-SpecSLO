# SPDX-License-Identifier: Apache-2.0
"""Isolate TP3 MC2 local MatMul accuracy with one active source rank.

This standalone-world3 diagnostic consumes real-layer captures written as
``rank-{0,1,2}.pt``.  It does not reproduce the draft rank or a four-rank
PEARL/SpecSLO process-group creation order; use the subgroup topology probe for
that separate question.  For each TP rank it keeps that rank's captured
activation and replaces the other two activations with zero.  It then compares
projection-only MC2 with::

    torch.nn.functional.linear(local_activation, local_weight)
    + the same deterministic target-group HCCL all-reduce

Each graph replays bounded A/B activation states through the same captured
static input address.  An all-zero sentinel follows the rotated source-rank
sequence and rejects stale MC2 window contents.  All ordinary HCCL references
are materialized before the first MC2 launch.
Diagnostic tensor exchange uses a separate Gloo group, so evidence collection
cannot overwrite or otherwise prime MC2's HCCL windows between invocations.

With only one non-zero local projection, an exact ordinary-HCCL result and an
identical MC2 result on all three ranks separate a source-rank local MatMul
difference from an all-reduce publication/rank-consensus failure.

Example for the captured Qwen3-32B down projection at layer 31::

    HCCL_DETERMINISTIC=true HCCL_OP_EXPANSION_MODE=AIV \
    ASCEND_RT_VISIBLE_DEVICES=1,2,4 torchrun --standalone --nproc-per-node 3 \
      examples/diagnose_specslo_mc2_local_matmul.py \
      --input-dir /path/to/captures/down/layer-31/m100 \
      --m 100 --k 8576 --n 5120 --output local-matmul-isolation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # Registers torch.npu and Ascend operators.

from vllm_ascend.spec_decode.pearl.mc2 import (
    current_mc2_runtime_binding,
    resolve_hccl_comm_name,
)

TP_SIZE = 3
EXECUTION_MODES = ("eager", "graph")
MISMATCH_COORDINATE_LIMIT = 256
ZERO_SENTINEL = "all_zero_sentinel"
INPUT_STATE_LABELS = ("A", "B")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing real-layer rank-{0,1,2}.pt captures.",
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--execution-modes",
        choices=EXECUTION_MODES,
        nargs="+",
        default=list(EXECUTION_MODES),
    )
    parser.add_argument(
        "--replays",
        type=int,
        default=4,
        help=(
            "Number of invocations per case; at least four are required to "
            "repeat both bounded A/B states for bit-exact stability checks."
        ),
    )
    parser.add_argument(
        "--state-delta",
        type=float,
        default=0.015625,
        help="Bounded BF16 delta between captured static input states A and B.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.m, args.k, args.n) <= 0:
        raise ValueError("M, K and N must all be positive")
    if args.replays < 4:
        raise ValueError("local MatMul isolation requires at least four replays")
    if not torch.isfinite(torch.tensor(args.state_delta)) or args.state_delta == 0:
        raise ValueError("state delta must be finite and non-zero")
    if len(set(args.execution_modes)) != len(args.execution_modes):
        raise ValueError("execution modes must not contain duplicates")


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _tensor_comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Return compact bit-exact evidence for two same-shaped CPU tensors."""

    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    if actual_cpu.shape != expected_cpu.shape:
        raise ValueError(
            f"comparison shape mismatch: {tuple(actual_cpu.shape)} != "
            f"{tuple(expected_cpu.shape)}"
        )
    mismatch = actual_cpu.ne(expected_cpu).reshape(-1)
    error = (actual_cpu.float() - expected_cpu.float()).abs().reshape(-1)
    mismatch_count = int(mismatch.sum().item())
    first_mismatch = None
    worst_mismatch = None
    mismatch_coordinates: list[list[int]] = []
    if mismatch_count:
        width = int(actual_cpu.shape[-1])
        mismatch_indices = mismatch.nonzero().reshape(-1)
        first_flat = int(mismatch_indices[0].item())
        worst_flat = int(error.argmax().item())
        mismatch_coordinates = [
            list(divmod(int(flat_index), width))
            for flat_index in mismatch_indices[:MISMATCH_COORDINATE_LIMIT].tolist()
        ]
        first_mismatch = {
            "flat_index": first_flat,
            "coordinate": list(divmod(first_flat, width)),
            "actual": float(actual_cpu.reshape(-1)[first_flat].float().item()),
            "expected": float(expected_cpu.reshape(-1)[first_flat].float().item()),
        }
        worst_mismatch = {
            "flat_index": worst_flat,
            "coordinate": list(divmod(worst_flat, width)),
            "actual": float(actual_cpu.reshape(-1)[worst_flat].float().item()),
            "expected": float(expected_cpu.reshape(-1)[worst_flat].float().item()),
        }
    return {
        "actual_sha256": _tensor_sha256(actual_cpu),
        "expected_sha256": _tensor_sha256(expected_cpu),
        "exact_match": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "element_count": int(mismatch.numel()),
        "mismatch_fraction": mismatch_count / int(mismatch.numel()),
        "max_abs_error": float(error.max().item()),
        "first_mismatch": first_mismatch,
        "worst_mismatch": worst_mismatch,
        "mismatch_coordinates_sample": mismatch_coordinates,
        "mismatch_coordinates_truncated": (
            mismatch_count > MISMATCH_COORDINATE_LIMIT
        ),
    }


def _all_tensors_exact(values: Sequence[torch.Tensor]) -> bool:
    if len(values) != TP_SIZE:
        raise ValueError(f"expected {TP_SIZE} TP-rank tensors, got {len(values)}")
    return all(torch.equal(values[0], value) for value in values[1:])


def _state_label(replay: int) -> str:
    if replay < 0:
        raise ValueError("replay index must be non-negative")
    return INPUT_STATE_LABELS[replay % len(INPUT_STATE_LABELS)]


def _source_order(replay: int) -> tuple[int | str, ...]:
    """Rotate the source preceding the zero sentinel across measured replays."""

    start = replay % TP_SIZE
    return tuple((start + offset) % TP_SIZE for offset in range(TP_SIZE)) + (
        ZERO_SENTINEL,
    )


def _cross_replay_stability(
    outputs_by_replay: Sequence[Sequence[torch.Tensor]],
    state_labels: Sequence[str],
    *,
    require_state_transition: bool,
) -> dict[str, Any]:
    if len(outputs_by_replay) != len(state_labels):
        raise ValueError("output and state-label replay counts differ")
    if len(outputs_by_replay) < 2:
        raise ValueError("cross-replay stability requires at least two replays")
    if any(len(outputs) != TP_SIZE for outputs in outputs_by_replay):
        raise ValueError(f"each replay must contain {TP_SIZE} rank outputs")

    hashes_by_state: dict[str, list[list[str]]] = {}
    for outputs, label in zip(outputs_by_replay, state_labels, strict=True):
        hashes_by_state.setdefault(label, []).append(
            [_tensor_sha256(output) for output in outputs]
        )
    same_state_bit_exact = all(
        all(replay_hashes == state_hashes[0] for replay_hashes in state_hashes[1:])
        for state_hashes in hashes_by_state.values()
    )
    state_transition_observed = True
    if require_state_transition:
        state_transition_observed = (
            set(INPUT_STATE_LABELS).issubset(hashes_by_state)
            and all(
                hashes_by_state[INPUT_STATE_LABELS[0]][0][rank]
                != hashes_by_state[INPUT_STATE_LABELS[1]][0][rank]
                for rank in range(TP_SIZE)
            )
        )
    return {
        "same_state_bit_exact": same_state_bit_exact,
        "state_transition_observed": state_transition_observed,
        "passed": same_state_bit_exact and state_transition_observed,
        "sha256_by_state_and_replay": hashes_by_state,
    }


def _classify_single_source_case(
    *,
    source_local_projection: torch.Tensor,
    split_outputs_by_rank: Sequence[torch.Tensor],
    mc2_outputs_by_rank: Sequence[torch.Tensor],
) -> tuple[str, dict[str, Any]]:
    """Classify a one-active-source projection without tolerance masking.

    ``local_matmul_or_prepublication_candidate`` is intentionally
    evidence-driven rather than an unconditional claim: the ordinary HCCL
    path transported the source projection exactly, every MC2 rank published
    the same value, and that common value differs from standard
    ``torch.linear``.  Under the one-active-source construction no
    floating-point reduction is required, which isolates the remaining
    arithmetic difference to MC2's source-local projection or another uniform
    transform before publication.
    """

    if len(split_outputs_by_rank) != TP_SIZE or len(mc2_outputs_by_rank) != TP_SIZE:
        raise ValueError(f"classification requires exactly {TP_SIZE} rank outputs")
    split_consensus = _all_tensors_exact(split_outputs_by_rank)
    mc2_consensus = _all_tensors_exact(mc2_outputs_by_rank)
    split_vs_source = [
        _tensor_comparison(value, source_local_projection)
        for value in split_outputs_by_rank
    ]
    mc2_vs_source = [
        _tensor_comparison(value, source_local_projection)
        for value in mc2_outputs_by_rank
    ]
    split_exact = split_consensus and all(item["exact_match"] for item in split_vs_source)
    mc2_exact = mc2_consensus and all(item["exact_match"] for item in mc2_vs_source)

    if not split_exact:
        classification = "ordinary_hccl_reference_failure"
    elif not mc2_consensus:
        classification = "mc2_rank_publication_or_reduction_failure"
    elif not mc2_exact:
        classification = "local_matmul_or_prepublication_candidate"
    else:
        classification = "exact"
    return classification, {
        "ordinary_hccl_rank_consensus": split_consensus,
        "ordinary_hccl_exact_to_source_local": split_exact,
        "mc2_rank_consensus": mc2_consensus,
        "mc2_exact_to_source_local": mc2_exact,
        "ordinary_hccl_vs_source_local_by_rank": split_vs_source,
        "mc2_vs_source_local_by_rank": mc2_vs_source,
    }


def _classify_zero_sentinel(
    *,
    split_outputs_by_rank: Sequence[torch.Tensor],
    mc2_outputs_by_rank: Sequence[torch.Tensor],
) -> tuple[str, dict[str, Any]]:
    if len(split_outputs_by_rank) != TP_SIZE or len(mc2_outputs_by_rank) != TP_SIZE:
        raise ValueError(f"zero sentinel requires exactly {TP_SIZE} rank outputs")
    expected = torch.zeros_like(split_outputs_by_rank[0])
    split_vs_zero = [
        _tensor_comparison(value, expected) for value in split_outputs_by_rank
    ]
    mc2_vs_zero = [_tensor_comparison(value, expected) for value in mc2_outputs_by_rank]
    split_consensus = _all_tensors_exact(split_outputs_by_rank)
    mc2_consensus = _all_tensors_exact(mc2_outputs_by_rank)
    split_zero = split_consensus and all(item["exact_match"] for item in split_vs_zero)
    mc2_zero = mc2_consensus and all(item["exact_match"] for item in mc2_vs_zero)
    if not split_zero:
        classification = "ordinary_hccl_zero_control_failure"
    elif not mc2_consensus:
        classification = "mc2_zero_rank_publication_failure"
    elif not mc2_zero:
        classification = "mc2_stale_window_or_nonzero_zero_control"
    else:
        classification = "exact_zero_control"
    return classification, {
        "ordinary_hccl_rank_consensus": split_consensus,
        "ordinary_hccl_all_zero": split_zero,
        "mc2_rank_consensus": mc2_consensus,
        "mc2_all_zero": mc2_zero,
        "ordinary_hccl_vs_zero_by_rank": split_vs_zero,
        "mc2_vs_zero_by_rank": mc2_vs_zero,
    }


def _physical_device_id(local_rank: int) -> str:
    visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES") or os.getenv("ASCEND_VISIBLE_DEVICES")
    if visible:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if local_rank >= len(devices):
            raise ValueError(
                f"visible device mapping {devices} has no local rank {local_rank}"
            )
        return devices[local_rank]
    return str(local_rank)


def _validate_payload(
    payload: Any,
    *,
    source: str,
    expected_shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(f"{source} is not a tensor dictionary")
    required = ("activation", "weight", "residual", "gamma")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"{source} is missing tensors {missing}")
    non_tensors = [name for name in required if not isinstance(payload[name], torch.Tensor)]
    if non_tensors:
        raise ValueError(f"{source} contains non-tensor values {non_tensors}")
    actual_shapes = {name: tuple(payload[name].shape) for name in required}
    if actual_shapes != expected_shapes:
        raise ValueError(f"real MC2 payload shapes {actual_shapes} != {expected_shapes}")
    bad_dtypes = {
        name: str(payload[name].dtype)
        for name in required
        if payload[name].dtype != torch.bfloat16
    }
    if bad_dtypes:
        raise ValueError(f"real MC2 payload requires BF16 tensors, got {bad_dtypes}")
    non_finite = [name for name in required if not bool(torch.isfinite(payload[name]).all())]
    if non_finite:
        raise ValueError(f"real MC2 payload contains non-finite tensors {non_finite}")
    return {name: payload[name] for name in required}


def _load_payload(
    input_dir: Path,
    tp_rank: int,
    expected_shapes: dict[str, tuple[int, ...]],
) -> tuple[dict[str, torch.Tensor], str]:
    path = input_dir / f"rank-{tp_rank}.pt"
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    payload = _validate_payload(loaded, source=str(path), expected_shapes=expected_shapes)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {name: value.to(device="npu") for name, value in payload.items()}, file_hash


def _gather_cpu_tensor(
    value: torch.Tensor,
    *,
    group: dist.ProcessGroup,
) -> list[torch.Tensor]:
    gathered: list[torch.Tensor | None] = [None] * TP_SIZE
    dist.all_gather_object(gathered, value.detach().cpu().contiguous(), group=group)
    if any(item is None for item in gathered):
        raise RuntimeError("Gloo evidence gather returned an empty TP-rank tensor")
    return [item for item in gathered if item is not None]


def _invoke_projection(
    operator: Any,
    activation: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    comm_name: str,
    tp_rank: int,
) -> torch.Tensor:
    projected, _ = operator(
        activation,
        weight,
        residual,
        gamma,
        comm_name,
        TP_SIZE,
        tp_rank,
        1e-6,
        True,
        False,
        True,
    )
    return projected


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    if dist.get_world_size() != TP_SIZE:
        raise ValueError(f"diagnostic requires exactly {TP_SIZE} ranks")
    tp_rank = rank
    evidence_group = dist.new_group(list(range(TP_SIZE)), backend="gloo")
    torch.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    operator = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm
    comm_name = resolve_hccl_comm_name(dist.group.WORLD, device="npu", rank=tp_rank)
    if not comm_name:
        raise RuntimeError("Could not resolve the TP3 HCCL communicator")
    runtime_binding = current_mc2_runtime_binding(
        tp_rank_device_mapping=tuple(
            _physical_device_id(index) for index in range(TP_SIZE)
        )
    )
    bindings: list[dict[str, Any] | None] = [None] * TP_SIZE
    dist.all_gather_object(bindings, runtime_binding, group=evidence_group)
    if any(binding != bindings[0] for binding in bindings[1:]):
        raise RuntimeError("TP3 ranks resolved different MC2 runtime bindings")

    expected_shapes = {
        "activation": (args.m, args.k),
        "weight": (args.n, args.k),
        "residual": (args.m, args.n),
        "gamma": (args.n,),
    }
    payload, payload_hash = _load_payload(args.input_dir, tp_rank, expected_shapes)
    activation = payload["activation"]
    weight = payload["weight"]
    residual = payload["residual"]
    gamma = payload["gamma"]
    case_ids: tuple[int | str, ...] = (*range(TP_SIZE), ZERO_SENTINEL)
    state_labels = [_state_label(replay) for replay in range(args.replays)]
    input_states: dict[int | str, list[torch.Tensor]] = {
        case_id: [] for case_id in case_ids
    }
    for case_id in case_ids:
        for replay in range(args.replays):
            if case_id == ZERO_SENTINEL or case_id != tp_rank:
                state = torch.zeros_like(activation)
            else:
                offset = args.state_delta if _state_label(replay) == "B" else 0.0
                state = (activation.float() + offset).to(torch.bfloat16).contiguous()
            input_states[case_id].append(state)

    # Materialize every reference before the first MC2 launch.  The standard
    # reference deliberately uses the production HCCL process group.
    source_local_projections: dict[int | str, list[torch.Tensor]] = {
        case_id: [] for case_id in case_ids
    }
    split_outputs: dict[int | str, list[list[torch.Tensor]]] = {
        case_id: [] for case_id in case_ids
    }
    for case_id in case_ids:
        for replay in range(args.replays):
            local_projection = torch.nn.functional.linear(
                input_states[case_id][replay], weight
            )
            gathered_local = _gather_cpu_tensor(local_projection, group=evidence_group)
            source_rank = case_id if isinstance(case_id, int) else 0
            source_local_projections[case_id].append(gathered_local[source_rank])
            split_projection = local_projection.clone()
            dist.all_reduce(split_projection)
            torch.npu.synchronize()
            split_outputs[case_id].append(
                _gather_cpu_tensor(split_projection, group=evidence_group)
            )

    # No ordinary HCCL collective occurs from here until all MC2 launches and
    # device-to-host copies are complete.
    measured: dict[str, dict[int | str, list[torch.Tensor]]] = {}
    if "eager" in args.execution_modes:
        eager_outputs = {case_id: [] for case_id in case_ids}
        for replay in range(args.replays):
            for case_id in _source_order(replay):
                output = _invoke_projection(
                    operator,
                    input_states[case_id][replay],
                    weight,
                    residual,
                    gamma,
                    comm_name,
                    tp_rank,
                )
                torch.npu.synchronize()
                eager_outputs[case_id].append(output.detach().cpu().contiguous())
        measured["eager"] = eager_outputs

    retained_graphs: dict[int | str, tuple[Any, torch.Tensor, torch.Tensor]] = {}
    if "graph" in args.execution_modes:
        graph_outputs = {case_id: [] for case_id in case_ids}
        for case_id in case_ids:
            static_input = input_states[case_id][0].clone()
            # Match production: eager warm-up, capture, then measured replays.
            _invoke_projection(
                operator,
                static_input,
                weight,
                residual,
                gamma,
                comm_name,
                tp_rank,
            )
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                static_output = _invoke_projection(
                    operator,
                    static_input,
                    weight,
                    residual,
                    gamma,
                    comm_name,
                    tp_rank,
                )
            retained_graphs[case_id] = (graph, static_input, static_output)
        for replay in range(args.replays):
            for case_id in _source_order(replay):
                graph, static_input, static_output = retained_graphs[case_id]
                static_input.copy_(input_states[case_id][replay])
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                graph_outputs[case_id].append(
                    static_output.detach().cpu().contiguous()
                )
        measured["graph"] = graph_outputs

    cases: list[dict[str, Any]] = []
    diagnostic_valid = True
    for case_id in case_ids:
        is_zero_control = case_id == ZERO_SENTINEL
        case: dict[str, Any] = {
            "case": case_id,
            "source_tp_rank": None if is_zero_control else case_id,
            "control": is_zero_control,
            "executions": {},
        }
        for mode, outputs_by_case in measured.items():
            invocations = []
            outputs_by_replay: list[list[torch.Tensor]] = []
            for replay, local_output in enumerate(outputs_by_case[case_id]):
                outputs_by_rank = _gather_cpu_tensor(local_output, group=evidence_group)
                outputs_by_replay.append(outputs_by_rank)
                if is_zero_control:
                    classification, evidence = _classify_zero_sentinel(
                        split_outputs_by_rank=split_outputs[case_id][replay],
                        mc2_outputs_by_rank=outputs_by_rank,
                    )
                else:
                    classification, evidence = _classify_single_source_case(
                        source_local_projection=(
                            source_local_projections[case_id][replay]
                        ),
                        split_outputs_by_rank=split_outputs[case_id][replay],
                        mc2_outputs_by_rank=outputs_by_rank,
                    )
                invocations.append(
                    {
                        "replay": replay,
                        "input_state": (
                            "ZERO" if is_zero_control else state_labels[replay]
                        ),
                        "preceding_source_tp_rank": (
                            _source_order(replay)[-2] if is_zero_control else None
                        ),
                        "classification": classification,
                        **evidence,
                    }
                )
            replay_labels = (
                ["ZERO"] * args.replays if is_zero_control else state_labels
            )
            mc2_stability = _cross_replay_stability(
                outputs_by_replay,
                replay_labels,
                require_state_transition=not is_zero_control,
            )
            reference_stability = _cross_replay_stability(
                split_outputs[case_id],
                replay_labels,
                require_state_transition=not is_zero_control,
            )
            expected_classifications = (
                {"exact_zero_control"}
                if is_zero_control
                else {"exact", "local_matmul_or_prepublication_candidate"}
            )
            execution_valid = (
                mc2_stability["passed"]
                and reference_stability["passed"]
                and all(
                    invocation["classification"] in expected_classifications
                    for invocation in invocations
                )
            )
            diagnostic_valid = diagnostic_valid and execution_valid
            case["executions"][mode] = {
                "valid": execution_valid,
                "mc2_cross_replay_gate": mc2_stability,
                "ordinary_hccl_cross_replay_gate": reference_stability,
                "invocations": invocations,
            }
        cases.append(case)

    payload_hashes: list[str | None] = [None] * TP_SIZE
    dist.all_gather_object(payload_hashes, payload_hash, group=evidence_group)
    if rank == 0:
        classifications = [
            invocation["classification"]
            for case in cases
            for execution in case["executions"].values()
            for invocation in execution["invocations"]
        ]
        document = {
            "schema_version": 2,
            "diagnostic": "tp3_mc2_single_source_local_matmul_isolation",
            "topology_scope": (
                "standalone world3 only; does not establish world4 target-subgroup "
                "process-group equivalence"
            ),
            "input_dir": str(args.input_dir.resolve()),
            "shape": {"m": args.m, "k": args.k, "n": args.n},
            "replays": args.replays,
            "state_delta": args.state_delta,
            "source_order_by_replay": [
                list(_source_order(replay)) for replay in range(args.replays)
            ],
            "execution_modes": list(args.execution_modes),
            "graph_input_protocol": (
                "one retained static input buffer per source/sentinel graph; "
                "copy bounded A/B state into that same buffer, synchronize, replay"
            ),
            "payload_validation": {
                "required_dtype": "torch.bfloat16",
                "all_tensors_finite": True,
            },
            "payload_sha256_by_rank": payload_hashes,
            "runtime_binding_by_rank": bindings,
            "runtime_binding_rank_consensus": all(
                binding == bindings[0] for binding in bindings[1:]
            ),
            "cases": cases,
            "summary": {
                "diagnostic_valid": diagnostic_valid,
                "all_noncontrol_exact": all(
                    value in {"exact", "exact_zero_control"}
                    for value in classifications
                ),
                "classification_counts": {
                    value: classifications.count(value)
                    for value in sorted(set(classifications))
                },
                "interpretation": {
                    "local_matmul_or_prepublication_candidate": (
                        "ordinary HCCL transported the sole source projection exactly "
                        "and all MC2 ranks agreed, but MC2 differed from torch.linear; "
                        "standalone world3 cannot separate Cube MatMul from another "
                        "uniform transform before MC2 publication"
                    ),
                    "mc2_rank_publication_or_reduction_failure": (
                        "MC2 ranks did not publish an identical projection"
                    ),
                    "ordinary_hccl_reference_failure": (
                        "the control HCCL path did not preserve the sole source projection"
                    ),
                    "exact_zero_control": (
                        "ordinary HCCL and MC2 both returned zero after a non-zero source"
                    ),
                },
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(document["summary"], sort_keys=True), flush=True)

    dist.barrier(group=evidence_group)
    dist.destroy_process_group()
    if not diagnostic_valid:
        raise RuntimeError(
            "TP3 MC2 local MatMul isolation controls or replay-stability gates failed; "
            f"inspect {args.output}"
        )


if __name__ == "__main__":
    main()
