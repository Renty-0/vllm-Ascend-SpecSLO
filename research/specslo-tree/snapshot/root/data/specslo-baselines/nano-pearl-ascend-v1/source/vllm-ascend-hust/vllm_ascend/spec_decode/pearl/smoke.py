# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-model-group HCCL protocol smoke test for experimental PEARL support.

Run with:
    torchrun --standalone --nproc_per_node=2 -m vllm_ascend.spec_decode.pearl.smoke
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from vllm_ascend.spec_decode.pearl.protocol import PearlProposalBatch, PearlVerificationBatch
from vllm_ascend.spec_decode.pearl.runtime import PearlRoundExecutor
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups, PearlTopology
from vllm_ascend.spec_decode.pearl.verifier import PearlTargetVerifier
from vllm_ascend.spec_decode.pearl.vocab import PearlVocabProjection


def parse_args() -> argparse.Namespace:
    """Parse the model-group sizes used by the protocol smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Exchange a proposal batch and correction batch across HCCL groups."""
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    try:
        topology = PearlTopology.from_tensor_parallel_sizes(args.draft_tp_size, args.target_tp_size)
        groups = PearlProcessGroups.create(topology, backend="hccl")
        device = torch.device(f"npu:{local_rank}")
        expected_proposals = PearlProposalBatch(
            request_slots=(101, 102),
            candidate_token_ids=((11, 12, 13), (21, 22, 23)),
        )

        expected_verifications = PearlVerificationBatch(
            request_slots=(101, 102),
            accepted_prefix_lengths=(3, 1),
            correction_token_ids=(None, 99),
            finished=(False, False),
        )
        if groups.rank == topology.target_leader_rank:
            # A target-only suffix is deliberately the raw argmax. Successful
            # verification proves the target logits were cropped to draft vocab.
            target_logits = torch.full((6, 160), -1.0, device=device)
            target_logits[:, 159] = 100.0
            target_logits[0, 11] = 10.0
            target_logits[1, 12] = 10.0
            target_logits[2, 13] = 10.0
            target_logits[3, 21] = 10.0
            target_logits[4, 99] = 10.0
            target_logits[5, 23] = 10.0
            target_logits_input = target_logits
        else:
            target_logits_input = None
        expected_verifications.validate_against(expected_proposals)
        executor = PearlRoundExecutor(
            groups=groups,
            device=device,
            verifier=PearlTargetVerifier.with_eos_tokens(
                PearlVocabProjection(draft_vocab_size=128, target_vocab_size=160),
                None,
            ),
        )
        received_proposals, received_verifications = executor.execute(
            expected_proposals if groups.rank == topology.draft_leader_rank else None,
            target_logits_input,
        )
        if groups.is_verification_worker:
            assert received_proposals == expected_proposals
        else:
            assert received_proposals is None
        assert received_verifications == expected_verifications
        dist.barrier()
        if groups.rank == 0:
            print("PEARL HCCL proposal and verification protocol passed.", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
