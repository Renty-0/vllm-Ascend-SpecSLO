"""Normal-request composite mode: packed known suffix/root + explicit head.

The native hidden graph still returns hidden; the native three-step graph
still returns encoded top2 pairs. Root gathering and selection are outside
ACLGraph and never claim native qualification by assignment.
"""
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

MODE = 'normal-packed-root-seed-top2-v1'
SCALE = 1_000_000


@dataclass(frozen=True)
class RootLayout:
    sequence_ids: tuple
    positions: tuple
    tokens: tuple
    root_indices: tuple
    query_ends: tuple
    sequence_ends: tuple
    repair_queries: int


def layout(entries, max_model_len, *, query_limit=128):
    ids, positions, tokens, roots, ends, contexts = [], [], [], [], [], []
    seen = set()
    repairs = 0
    for seq, prefix, suffix_start, suffix, root in entries:
        suffix = tuple(suffix)
        if (type(seq) is not int or seq < 0 or seq in seen
                or type(prefix) is not int or not 0 <= prefix < max_model_len
                or type(suffix_start) is not int or suffix_start < 0
                or suffix_start+len(suffix) != prefix
                or type(root) is not int or root < 0
                or any(type(t) is not int or t < 0 for t in suffix)):
            raise ValueError('Root seed must contain unique requests and a known contiguous suffix ending at its root')
        seen.add(seq)
        values = suffix+(root,)
        ids.extend([seq]*len(values))
        positions.extend(range(suffix_start, prefix+1))
        tokens.extend(values)
        roots.append(len(tokens)-1)
        ends.append(len(tokens))
        contexts.append(prefix+1)
        repairs += len(suffix)
    if not entries or len(tokens) > query_limit:
        raise ValueError('Packed root seed exceeds its independently bounded physical-query scope')
    return RootLayout(tuple(ids), tuple(positions), tuple(tokens), tuple(roots),
                      tuple(ends), tuple(contexts), repairs)


def prepare(engine, entries, native):
    import torch
    if engine.config.enable_prefix_caching:
        raise RuntimeError('Root seed prototype requires the original uncached draft allocation')
    result = layout(entries, engine.config.max_model_len,
                    query_limit=min(128, int(engine.graph_runner.max_graph_tokens)))
    capacity = min(128, int(engine.graph_runner.max_graph_tokens))
    physical = native._next_linear_draft_graph_bucket(len(result.tokens), capacity)
    inputs = torch.tensor(result.tokens, dtype=torch.long, device=engine.device)
    positions, metadata = engine._prepare_attention_metadata(
        list(result.sequence_ids), list(result.positions), False)
    if (metadata.use_fused_infer_attention or metadata.request_block_tables is not None
            or metadata.attention_mask is not None
            or tuple(metadata.context_lens.tolist()) != tuple(p+1 for p in result.positions)
            or tuple(metadata.actual_seq_lengths_q) != result.query_ends
            or tuple(metadata.sequence_lens) != result.sequence_ends):
        raise RuntimeError('Original metadata did not preserve grouped root-seed PA causality')
    inputs, positions, metadata = native.NativeACLGraphRunner._pad_inputs(
        inputs, positions, metadata, physical)
    return result, inputs, positions, metadata, physical


def explicit_root_head(engine, hidden, root_indices, head_rows):
    """Original logits/FP32 softmax/top2/encoding, device-only real-row gather."""
    import torch
    index = torch.tensor(root_indices, dtype=torch.long, device=hidden.device)
    # Keep the original normal-batch LM projection M. Finite dummy hidden rows
    # have no logical request and never enter the returned real-row prefix.
    selected = hidden.new_zeros((head_rows, hidden.shape[-1]))
    selected[:len(root_indices)].copy_(hidden.index_select(0, index))
    logits = engine.model.compute_logits(selected)[:, :engine.draft_vocab_size]
    probabilities = torch.softmax(logits.float(), dim=-1)
    confidence, tokens = probabilities.topk(2, dim=-1)
    encoded = tokens.to(torch.long)*(SCALE+1)+torch.round(
        confidence.clamp(0, 1)*SCALE).to(torch.long)
    return encoded, tokens[:, 0]


def install(native):
    import torch
    import fast_tree_draft_v5 as fast
    fast.install(native)
    cls = native.NativePearlEngine
    original = cls.draft_tree_forward

    def draft(engine, plans, roots, ids, eager_parent_sources=None, **kwargs):
        if (not engine.is_draft or eager_parent_sources or not plans
                or any(p.width != 2 or int(p.depth) != 4 for p in plans)
                or min(4, max(int(p.candidate_budget) for p in plans)) != 4):
            return original(engine, plans, roots, ids,
                            eager_parent_sources=eager_parent_sources, **kwargs)
        if (not len(plans) == len(roots) == len(ids) or len(ids) != len(set(ids))
                or any(type(seq) is not int or seq < 0 for seq in ids)):
            raise ValueError('Root seed requires aligned, unique normal request rows')
        if any(float(v) > 0 for v in (kwargs.get('draft_temperatures') or ())):
            raise RuntimeError('Canonical eager draft is qualified only for greedy sampling')
        if len(ids) > engine.config.max_num_seqs:
            raise RuntimeError('Root seed normal request count exceeds the original capacity')
        catchups = kwargs.get('committed_catchup_token_ids') or [()]*len(ids)
        if len(catchups) != len(ids):
            raise ValueError('Root seed committed catchup rows must align with requests')
        known = getattr(engine, '_dynamic_tree_known_KV', {})
        entries = []
        for plan, root, seq, values in zip(plans, roots, ids, catchups):
            prefix = int(plan.prefix_len)
            if values and int(values[-1]) != int(root):
                raise RuntimeError('Root seed received a stale committed current root')
            state = known.get(seq)
            start, suffix = (fast.missing_committed_suffix(state, prefix, values)
                             if state is not None else (prefix, []))
            entries.append((seq, prefix, start, suffix, int(root)))
        if not any(entry[3] for entry in entries):
            return original(engine, plans, roots, ids,
                            eager_parent_sources=eager_parent_sources, **kwargs)
        packed_limit = min(128, int(engine.graph_runner.max_graph_tokens))
        if sum(len(entry[3])+1 for entry in entries) > packed_limit:
            # The composite mode is optional. Preserve the complete committed
            # suffix on the original qualified stepwise path when it cannot fit
            # one packed call; never truncate known tokens to force admission.
            counts = getattr(engine, '_dynamic_tree_draft_counts', None)
            if counts is None:
                counts = engine._dynamic_tree_draft_counts = Counter()
            counts['root_seed_capacity_fallback_calls'] += 1
            return original(engine, plans, roots, ids,
                            eager_parent_sources=eager_parent_sources, **kwargs)

        # No hidden/selector readback occurs before the original final encoded
        # result readback. Native model/LM-head sticky finite flags are retained.
        fast.reserve_query_pages(engine, plans, ids)
        packed, inputs, positions, metadata, physical = prepare(engine, entries, native)
        hidden = engine.graph_runner.run_tree_hidden(inputs, positions, metadata)
        execution = engine.graph_runner.last_target_execution
        if execution.mode not in ('capture_replay', 'replay') or not execution.used_aclgraph:
            raise RuntimeError('Root seed requires the original native hidden ACLGraph gate')
        hidden_key = ('target-hidden:0|steps:1|paged', (physical,))
        if hidden_key not in engine.graph_runner.target_entries:
            raise RuntimeError('Root seed has no authoritative original hidden graph entry')
        first_hidden_diagnostic = (hidden[:len(packed.tokens)].clone()
            if getattr(engine, '_v27_retain_diagnostics', False) else None)
        head_rows = native._next_linear_draft_graph_bucket(len(ids), engine.config.max_num_seqs)
        seed, first_tokens = explicit_root_head(engine, hidden, packed.root_indices, head_rows)
        from chain_metadata import build as chain_metadata
        _, tail_positions, tail_metadata = chain_metadata(engine,
            [SimpleNamespace(prefix_len=int(p.prefix_len)+1) for p in plans],
            ids, [0]*len(ids), 3, head_rows)
        engine.graph_runner._dynamic_tree_top2 = True
        tail = engine.graph_runner.run_draft_greedy(first_tokens, tail_positions, tail_metadata,
            engine.draft_vocab_size, valid_row_count=len(ids), final_kv_only=False,
            return_confidence=True).clone()
        tail_execution = engine.graph_runner.last_draft_execution
        if tail_execution.mode not in ('capture_replay', 'replay') or not tail_execution.used_aclgraph:
            raise RuntimeError('Root seed requires the original native three-step draft graph gate')
        encoded = torch.cat((seed[:, None, :], tail), dim=1)[:len(ids)]
        if getattr(engine, '_v27_retain_diagnostics', False):
            engine._v27_last_diagnostics = dict(mode=MODE, layout=packed,
                first_hidden=first_hidden_diagnostic,
                encoded=encoded.clone(), hidden_key=hidden_key,
                tail_key=engine.graph_runner.last_draft_entry_key,
                physical_rows=physical, head_rows=head_rows)
        values = encoded.cpu().tolist()  # Same single final readback as FastV5.
        rows, scores = [], []
        for seq, root, plan, row in zip(ids, roots, plans, values):
            tokens = [[int(v)//(SCALE+1) for v in pair] for pair in row]
            confidence = [[int(v) % (SCALE+1)/SCALE for v in pair] for pair in row]
            rows.append([pair[0] for pair in tokens]+[pair[1] for pair in tokens])
            scores.append([pair[0] for pair in confidence]+[pair[1] for pair in confidence])
            # The last predicted m4 is still deferred, exactly as in FastV5.
            known[seq] = dict(safe_before=int(plan.prefix_len),
                tokens={int(plan.prefix_len)+j:token for j, token in
                        enumerate([int(root)]+[pair[0] for pair in tokens[:-1]])})
        engine._dynamic_tree_known_KV = known
        counts = getattr(engine, '_dynamic_tree_draft_counts', None)
        if counts is None:
            counts = engine._dynamic_tree_draft_counts = Counter()
        counts['chain_calls'] += 1
        counts['chain_real_queries'] += 3*len(ids)
        counts['chain_padded_queries'] += 3*head_rows
        counts['generated_candidates'] += 8*len(ids)
        counts['root_seed_calls'] += 1
        counts['root_seed_PA_real_queries'] += len(packed.tokens)
        counts['root_seed_PA_padded_queries'] += physical
        counts['root_seed_committed_repair_queries'] += packed.repair_queries
        counts['root_seed_root_queries'] += len(ids)
        counts['root_seed_head_outside_graph_calls'] += 1
        counts['root_seed_head_padded_rows'] += head_rows
        counts['root_seed_model_stages'] += 4
        return dict(draft_token_ids=rows, root_token_ids=list(map(int, roots)),
            eager_frontier_tokens={}, parent_indices=torch.cat([p.parent_indices for p in plans]),
            num_draft_tokens=[p.candidate_budget for p in plans],
            draft_confidence=torch.tensor([sum(row)/8 for row in scores]),
            node_confidences=[torch.tensor(row) for row in scores], cache_slot_mapping=None,
            query_count=physical+3*head_rows,
            graph_padding_query_count=physical-len(packed.tokens)+3*(head_rows-len(ids)),
            model_calls=2, graph_calls=2, model_stages=4, root_selection_outside_graph=True,
            execution_mode=MODE, profile_seconds={}, materialization_deferred=True,
            linear_full_chain=False, canonical_deferred_KV=True)

    cls.draft_tree_forward = draft
    return original
