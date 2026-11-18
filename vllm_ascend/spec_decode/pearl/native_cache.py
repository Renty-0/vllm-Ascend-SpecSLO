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
        max_cached_tokens: int | None = None,
        defer_prefix_registration: bool = False,
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
            if max_cached_tokens is not None:
                if max_cached_tokens < 0 or max_cached_tokens % self.block_size:
                    raise ValueError("Prefix reuse cap must be a non-negative full-block length.")
                num_full_blocks = min(num_full_blocks, max_cached_tokens // self.block_size)
            num_prompt_blocks = (len(prompt) + self.block_size - 1) // self.block_size
            for logical_block in range(num_prompt_blocks):
                prefix_end = (logical_block + 1) * self.block_size
                prefix_key = tuple(prompt[:prefix_end]) if logical_block < num_full_blocks else None
                block_id, cache_hit = self._acquire_block(
                    prefix_key if enable_prefix_caching else None,
                    register_new=not defer_prefix_registration,
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

    def available_cached_tokens(self, prompt: list[int]) -> int:
        """Count the contiguous, already published full prompt blocks."""

        if not prompt:
            return 0
        full_blocks = max(0, len(prompt) - 1) // self.block_size
        count = 0
        for logical_block in range(full_blocks):
            prefix_end = (logical_block + 1) * self.block_size
            if tuple(prompt[:prefix_end]) not in self._prefix_to_block:
                break
            count += 1
        return count * self.block_size

    def clear_reusable_prefixes(self) -> None:
        """Start a new benchmark generate call without warmup KV carryover."""

        if any(block.ref_count for block in self._blocks):
            raise RuntimeError("Reusable prefix keys cannot be cleared with live KV pages.")
        self._prefix_to_block.clear()
        for block in self._blocks:
            block.prefix_key = None

    def publish_prompt_blocks(self, sequence_id: int, prompt: list[int]) -> int:
        """Expose completed prompt KV to later requests, never in-flight pages."""

        if self._active_tables is None or not 0 <= sequence_id < len(self._active_tables):
            raise RuntimeError("A live cache row is required to publish its prompt blocks.")
        table = self._active_tables[sequence_id]
        full_blocks = max(0, len(prompt) - 1) // self.block_size
        published = 0
        for logical_block in range(full_blocks):
            block_id = table[logical_block]
            if block_id < 0:
                raise RuntimeError("A completed prompt is missing a full KV page.")
            block = self._blocks[block_id]
            prefix_end = (logical_block + 1) * self.block_size
            key = tuple(prompt[:prefix_end])
            if block.prefix_key == key:
                continue
            if block.prefix_key is not None:
                raise RuntimeError("A KV page cannot be published for two prompt prefixes.")
            if key in self._prefix_to_block:
                # Another completed request already owns the reusable copy.
                continue
            block.prefix_key = key
            self._prefix_to_block[key] = block_id
            self._clock += 1
            block.last_used = self._clock
            published += 1
        return published

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

    def release_sequence(self, sequence_id: int) -> int:
        """Release one live row while retaining the surrounding allocation.

        Online serving reserves logical page-table rows for the complete
        request trace, but physical pages only need to follow the resident
        service batch. Clearing a completed row lets a later request reuse
        those pages without changing either request's stable sequence ID.
        """

        if self._active_tables is None:
            raise RuntimeError("Allocate a PEARL batch before releasing a live sequence.")
        if not 0 <= sequence_id < len(self._active_tables):
            raise ValueError("PEARL cache sequence ID is outside the reserved batch.")
        table = self._active_tables[sequence_id]
        released = sum(block_id != -1 for block_id in table)
        self._release_table(table)
        table[:] = [-1] * len(table)
        return released

    def _release_table(self, table: list[int]) -> None:
        for block_id in table:
            if block_id == -1:
                continue
            block = self._blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError("PEARL KV cache block reference count became negative.")

    def _acquire_block(
        self,
        prefix_key: tuple[int, ...] | None,
        *,
        register_new: bool = True,
    ) -> tuple[int, bool]:
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
            tables = self._active_tables or []
            live_rows = sum(any(block_id >= 0 for block_id in table) for table in tables)
            peak_row_blocks = max((sum(block_id >= 0 for block_id in table) for table in tables), default=0)
            raise RuntimeError(
                "The native PEARL KV cache has no free physical blocks: "
                f"capacity={self.num_blocks}, live_rows={live_rows}, "
                f"peak_row_blocks={peak_row_blocks}, "
                f"selective_pressure_estimate={getattr(self, 'selective_pressure_estimate', -1)}, "
                f"selective_cycles={getattr(self, 'selective_cycles', -1)}."
            ) from error

        if block.prefix_key is not None:
            self._prefix_to_block.pop(block.prefix_key, None)
        block.prefix_key = prefix_key if register_new else None
        block.ref_count = 1
        block.last_used = self._clock
        if block.prefix_key is not None:
            self._prefix_to_block[prefix_key] = block.block_id
        return block.block_id, False
