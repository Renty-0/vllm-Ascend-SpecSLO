# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic full-block prefix cache for the native PEARL runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeCacheAllocation:
    """Physical page tables and reusable prompt lengths for one static batch."""

    block_tables: list[list[int]]
    num_cached_tokens: list[int]


@dataclass
class _CacheBlock:
    block_id: int
    prefix_key: tuple[int, ...] | None = None
    ref_count: int = 0
    last_used: int = 0


class NativePrefixCache:
    """Allocate paged KV slots and retain unreferenced full prompt blocks."""

    def __init__(self, num_blocks: int, blocks_per_sequence: int, block_size: int) -> None:
        if num_blocks <= 0 or blocks_per_sequence <= 0 or block_size <= 0:
            raise ValueError("Native prefix-cache dimensions must be positive.")
        self.num_blocks = num_blocks
        self.blocks_per_sequence = blocks_per_sequence
        self.block_size = block_size
        self._blocks = [_CacheBlock(block_id) for block_id in range(num_blocks)]
        self._prefix_to_block: dict[tuple[int, ...], int] = {}
        self._clock = 0
        self._active_tables: list[list[int]] | None = None

    def allocate(
        self,
        prompts: list[list[int]],
        enable_prefix_caching: bool = True,
        *,
        sequence_capacity: int | None = None,
    ) -> NativeCacheAllocation:
        if self._active_tables is not None:
            raise RuntimeError("Release the active PEARL cache allocation before allocating another batch.")
        capacity = len(prompts) if sequence_capacity is None else int(sequence_capacity)
        if capacity < len(prompts) or capacity <= 0:
            raise ValueError("Native PEARL cache capacity must fit every initial prompt.")

        block_tables: list[list[int]] = []
        cached_token_counts: list[int] = []
        acquired_block_ids: list[int] = []
        try:
            for prompt in prompts:
                if len(prompt) > self.blocks_per_sequence * self.block_size:
                    raise ValueError("A PEARL prompt exceeds its KV cache page table.")
                table = [-1] * self.blocks_per_sequence
                cached_tokens = 0
                max_reusable_tokens = max(0, len(prompt) - 1)
                num_full_blocks = max_reusable_tokens // self.block_size
                num_prompt_blocks = (len(prompt) + self.block_size - 1) // self.block_size
                for logical_block in range(num_prompt_blocks):
                    prefix_end = (logical_block + 1) * self.block_size
                    prefix_key = tuple(prompt[:prefix_end]) if logical_block < num_full_blocks else None
                    block_id, cache_hit = self._acquire_block(
                        prefix_key if enable_prefix_caching else None,
                    )
                    acquired_block_ids.append(block_id)
                    table[logical_block] = block_id
                    if cache_hit:
                        cached_tokens += self.block_size
                block_tables.append(table)
                cached_token_counts.append(cached_tokens)
            block_tables.extend([[-1] * self.blocks_per_sequence for _ in range(capacity - len(prompts))])
            cached_token_counts.extend([0] * (capacity - len(prompts)))
        except Exception:
            for block_id in acquired_block_ids:
                self._blocks[block_id].ref_count -= 1
            raise

        self._active_tables = block_tables
        return NativeCacheAllocation(block_tables, cached_token_counts)

    def activate_sequence(
        self,
        sequence_id: int,
        prompt: list[int],
        *,
        enable_prefix_caching: bool = True,
    ) -> int:
        """Populate one reserved page-table row for a live request."""
        if self._active_tables is None:
            raise RuntimeError("Allocate a PEARL batch before activating a live sequence.")
        if not 0 <= sequence_id < len(self._active_tables):
            raise ValueError("PEARL cache sequence ID is outside the reserved batch.")
        table = self._active_tables[sequence_id]
        if any(block_id != -1 for block_id in table):
            raise RuntimeError("A live PEARL cache sequence cannot be activated twice.")
        if not prompt or len(prompt) > self.blocks_per_sequence * self.block_size:
            raise ValueError("A live PEARL prompt must fit its KV cache page table.")

        acquired_block_ids: list[int] = []
        cached_tokens = 0
        try:
            max_reusable_tokens = max(0, len(prompt) - 1)
            num_full_blocks = max_reusable_tokens // self.block_size
            num_prompt_blocks = (len(prompt) + self.block_size - 1) // self.block_size
            for logical_block in range(num_prompt_blocks):
                prefix_end = (logical_block + 1) * self.block_size
                prefix_key = tuple(prompt[:prefix_end]) if logical_block < num_full_blocks else None
                block_id, cache_hit = self._acquire_block(
                    prefix_key if enable_prefix_caching else None,
                )
                acquired_block_ids.append(block_id)
                table[logical_block] = block_id
                if cache_hit:
                    cached_tokens += self.block_size
        except Exception:
            for logical_block, block_id in enumerate(acquired_block_ids):
                self._blocks[block_id].ref_count -= 1
                table[logical_block] = -1
            raise
        return cached_tokens

    def ensure_capacity(
        self,
        sequence_ids: list[int],
        positions: list[int],
    ) -> list[tuple[int, int, int]]:
        """Allocate pages and return device page-table updates."""
        if self._active_tables is None:
            raise RuntimeError("Allocate a PEARL batch before extending its KV cache.")
        if len(sequence_ids) != len(positions):
            raise ValueError("Every PEARL cache position needs a sequence ID.")
        updates: list[tuple[int, int, int]] = []
        for sequence_id, position in zip(sequence_ids, positions):
            if sequence_id < 0 or sequence_id >= len(self._active_tables):
                raise ValueError("PEARL cache sequence ID is outside the active batch.")
            logical_block = position // self.block_size
            if logical_block >= self.blocks_per_sequence:
                raise ValueError("A PEARL sequence exceeded its configured max_model_len.")
            if self._active_tables[sequence_id][logical_block] == -1:
                block_id, _ = self._acquire_block(None)
                self._active_tables[sequence_id][logical_block] = block_id
                updates.append((sequence_id, logical_block, block_id))
        return updates

    def release(self) -> None:
        if self._active_tables is None:
            return
        for table in self._active_tables:
            self._release_table(table)
        self._active_tables = None

    def _release_table(self, table: list[int]) -> None:
        for block_id in table:
            if block_id == -1:
                continue
            block = self._blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError("PEARL KV cache block reference count became negative.")

    def _acquire_block(self, prefix_key: tuple[int, ...] | None) -> tuple[int, bool]:
        self._clock += 1
        if prefix_key is not None and prefix_key in self._prefix_to_block:
            block = self._blocks[self._prefix_to_block[prefix_key]]
            block.ref_count += 1
            block.last_used = self._clock
            return block.block_id, True

        free_blocks = (block for block in self._blocks if block.ref_count == 0)
        try:
            block = min(
                free_blocks,
                key=lambda candidate: (
                    candidate.prefix_key is not None,
                    candidate.last_used,
                    candidate.block_id,
                ),
            )
        except ValueError as error:
            raise RuntimeError("The native PEARL KV cache has no free physical blocks.") from error

        if block.prefix_key is not None:
            self._prefix_to_block.pop(block.prefix_key, None)
        block.prefix_key = prefix_key
        block.ref_count = 1
        block.last_used = self._clock
        if prefix_key is not None:
            self._prefix_to_block[prefix_key] = block.block_id
        return block.block_id, False
