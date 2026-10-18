# SPDX-License-Identifier: Apache-2.0
"""Checked target metadata from the current host cache allocation.

The allocator remains authoritative. The maintained device table is gathered
after capacity updates on the same current stream; it is never read to decide
whether a physical page is valid. Model operations and graph inputs are unchanged.
"""
from dataclasses import replace
import inspect
import textwrap

import torch

from vllm_ascend.spec_decode.pearl.native_model import (
    NativeAttentionMetadata,
    _is_contiguous_linear_tree_plan,
    make_tree_fia_mask,
)


def install(native):
    """Change only the checked metadata builder in the original target call."""
    cls = native.NativePearlEngine
    source = textwrap.dedent(inspect.getsource(cls.target_tree_forward))
    before = ('self.model.make_tree_attention_metadata(\n'
              '        plan_list,\n        roots,\n        padded_candidates,\n'
              '        self.cache_allocation.block_tables,')
    after = ('_dynamic_tree_make_host_metadata(\n        self,\n'
             '        plan_list,\n        roots,\n        padded_candidates,')
    assert source.count(before) == 1
    source = source.replace(before, after)
    def counted(engine, *args, **kwargs):
        result = make_host_tree_attention_metadata(engine, *args, **kwargs)
        engine._dynamic_tree_host_target_calls = getattr(engine, '_dynamic_tree_host_target_calls', 0) + 1
        return result
    namespace = vars(native)
    namespace['_dynamic_tree_make_host_metadata'] = counted
    exec(compile(source, __file__ + ':target_forward', 'exec'), namespace)
    cls.target_tree_forward = namespace['target_tree_forward']


def checked_host_tree_slots(block_tables, sequence_ids, positions, *, block_size, max_model_len, cache_capacity):
    """Validate every packed query before transferring its physical slot."""
    if (type(block_size) is not int or block_size <= 0 or type(cache_capacity) is not int
            or cache_capacity <= 0 or type(max_model_len) is not int or max_model_len <= 0
            or not isinstance(block_tables, list) or not block_tables
            or len(sequence_ids) != len(positions) or not positions):
        raise ValueError("Target host slot dimensions must be positive and aligned.")
    slots = []
    for sequence_id, position in zip(sequence_ids, positions, strict=True):
        if type(sequence_id) is not int or not 0 <= sequence_id < len(block_tables):
            raise ValueError("Target host block tables must cover every non-negative request ID.")
        if type(position) is not int or not 0 <= position < max_model_len:
            raise ValueError("Target host cache position must fit max_model_len.")
        table = block_tables[sequence_id]
        logical_block = position // block_size
        if logical_block >= len(table):
            raise RuntimeError("Target host query exceeds its physical block table.")
        page = table[logical_block]
        if type(page) is not int or page < 0:
            raise RuntimeError("Target host forward referenced an unallocated KV cache page.")
        slot = page * block_size + position % block_size
        if slot >= cache_capacity or (page + 1) * block_size > cache_capacity:
            raise RuntimeError("Target host KV mapping contains an out-of-bounds physical slot.")
        slots.append(slot)
    if len(set(slots)) != len(slots):
        raise RuntimeError("Target host KV mapping aliases distinct tree queries.")
    return slots


def _cpu_values(tensor, label):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.ndim != 1:
        raise ValueError(f"Target host {label} must be a one-dimensional CPU tensor.")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Target host {label} must contain integer positions.")
    return tensor.tolist()


def _uses_fused_attention(device, attention):
    return device.type == "npu" and attention.uses_paged_attention


def _make_host_causal_metadata(model, packed_ids, logical_positions, physical_tables, slots):
    positions, metadata = model.make_attention_metadata(
        packed_ids, logical_positions, physical_tables, slot_mapping=slots,
        use_fused_infer_attention=True,
    )
    # The original implicit mapping combines int32 physical pages with the
    # long position tensor's remainder, so promotion produces int64 slots.
    # The explicit host-slot API creates int32; restore the original input
    # dtype without changing any checked value or reading device data.
    return positions, replace(metadata, slot_mapping=metadata.slot_mapping.to(dtype=torch.long))


def make_host_tree_attention_metadata(engine, plans, roots, draft_rows, *, sequence_ids=None,
                                      allow_causal_fast_path=True):
    """Pack the original FULL/causal envelope with host-checked cache slots."""
    if not plans or len(plans) != len(roots) or len(plans) != len(draft_rows):
        raise ValueError("tree plans, root tokens and draft rows must have equal non-zero length")
    allocation = engine.cache_allocation
    if allocation is None or engine.cache_block_tables is None:
        raise RuntimeError("Allocate the native target cache before host metadata packing.")
    tables = allocation.block_tables
    if not isinstance(tables, list) or not tables or not tables[0]:
        raise ValueError("Target host metadata requires the allocator's host block tables.")
    table_width = len(tables[0])
    if any(len(row) != table_width for row in tables):
        raise ValueError("Target host block table rows must have equal lengths.")
    model = engine.model
    attention = model.layers[0].self_attn
    device = model.embed_tokens.weight.device
    physical_tables = engine.cache_block_tables
    if (not isinstance(physical_tables, torch.Tensor) or physical_tables.ndim != 2
            or tuple(physical_tables.shape) != (len(tables), table_width)
            or physical_tables.dtype != torch.int32 or physical_tables.device != device):
        raise ValueError("Target cached device block tables must match the current host table shape and device.")
    request_ids = list(range(len(plans))) if sequence_ids is None else list(sequence_ids)
    if (len(request_ids) != len(plans) or len(set(request_ids)) != len(request_ids)
            or any(type(index) is not int or not 0 <= index < len(tables) for index in request_ids)):
        raise ValueError("tree sequence IDs must be unique, non-negative and row-aligned")
    inputs, logical_positions, physical_positions, packed_ids = [], [], [], []
    masks, query_lengths, sequence_lens = [], [], []
    all_linear = bool(allow_causal_fast_path)
    for request_id, plan, root, candidates in zip(request_ids, plans, roots, draft_rows, strict=True):
        expected_nodes = int(plan.parent_indices.numel())
        if len(candidates) != expected_nodes:
            raise ValueError("draft row length does not match its tree plan")
        cache_positions = getattr(plan, "cache_positions", None)
        cache_positions = plan.positions if cache_positions is None else cache_positions
        physical = _cpu_values(cache_positions, "cache positions")
        logical = _cpu_values(plan.positions, "logical positions")
        if len(physical) != expected_nodes + 1 or len(set(physical)) != len(physical):
            raise ValueError("tree cache positions must be unique and include the root")
        if len(logical) != len(physical):
            raise ValueError("tree logical/cache positions must have equal length")
        if any(not 0 <= position < model.max_model_len for position in logical):
            raise ValueError("Target host logical position must fit max_model_len.")
        mask = plan.attention_mask
        if tuple(mask.shape) != (expected_nodes + 1, model.max_model_len):
            raise ValueError("tree attention mask shape does not match its plan")
        if mask.device.type != "cpu" or mask.dtype != torch.bool:
            raise ValueError("Target host tree masks must remain boolean CPU tensors.")
        if plan.parent_indices.device.type != "cpu":
            raise ValueError("Target host tree parents must remain CPU tensors.")
        inputs.extend([int(root), *map(int, candidates)])
        logical_positions.extend(logical)
        physical_positions.extend(physical)
        packed_ids.extend([request_id] * len(physical))
        masks.append(mask)
        query_lengths.append(len(physical))
        sequence_lens.append(max(physical) + 1)
        all_linear = all_linear and _is_contiguous_linear_tree_plan(plan)
    slots = checked_host_tree_slots(
        tables, packed_ids, physical_positions, block_size=int(attention.block_size),
        max_model_len=int(model.max_model_len), cache_capacity=int(engine._tree_cache_capacity),
    )
    input_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    use_fused_tree_attention = _uses_fused_attention(device, attention)
    if use_fused_tree_attention and all_linear:
        positions, metadata = _make_host_causal_metadata(
            model, packed_ids, logical_positions, physical_tables, slots)
        return input_tensor, positions, metadata
    positions = torch.tensor(logical_positions, dtype=torch.long, device=device)
    request_tensor = torch.tensor(request_ids, dtype=torch.long, device=device)
    request_tables = physical_tables.index_select(0, request_tensor)
    if use_fused_tree_attention:
        dense_mask = None
        token_tables = request_tables
    else:
        dense_mask = torch.cat(masks, dim=0).to(device=device, dtype=torch.bool)
        token_tensor = torch.tensor(packed_ids, dtype=torch.long, device=device)
        token_tables = physical_tables.index_select(0, token_tensor)
    cumulative = []
    for length in query_lengths:
        cumulative.append((cumulative[-1] if cumulative else 0) + length)
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
        context_lens=torch.tensor([position + 1 for position in physical_positions], dtype=torch.int32),
        block_tables=token_tables, actual_seq_lengths_q=tuple(cumulative),
        sequence_lens=tuple(sequence_lens), request_block_tables=request_tables,
        attention_mask=dense_mask, use_fused_infer_attention=use_fused_tree_attention,
        tree_attention=True, tree_attention_mask=make_tree_fia_mask(masks).to(device=device, dtype=torch.bool),
    )
    return input_tensor, positions, metadata
