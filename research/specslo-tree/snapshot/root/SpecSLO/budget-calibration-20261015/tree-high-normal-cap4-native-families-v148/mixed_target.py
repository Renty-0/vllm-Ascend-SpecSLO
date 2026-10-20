"""Independent true-tree + exact causal-prompt target forward prototype.

Not installed in any service loop. No sampling/state publication, KV
compaction, distributed rendezvous or graph sealing is performed here.
Native graph methods own capture and changed-input numerical validation.
"""
from dataclasses import dataclass, replace

import torch
from vllm_ascend.spec_decode.pearl.roofline import attention_backend_identity

from mixed_layout import MixedLayout


@dataclass(frozen=True)
class PreparedMixedCall:
    layout: MixedLayout
    input_ids: torch.Tensor
    positions: torch.Tensor
    metadata: object


def record_hidden_fault(engine, hidden):
    """Sticky device flag; never exit one TP rank before its projection peers."""
    failed=~torch.isfinite(hidden).all()
    flag=getattr(engine,'_spec_slo_mixed_fault',None)
    if flag is None:flag=engine._spec_slo_mixed_fault=torch.zeros((),dtype=torch.bool,device=hidden.device)
    flag.logical_or_(failed)
    original=getattr(engine.model,'output_nonfinite',None)
    if isinstance(original,torch.Tensor):original.logical_or_(failed)
    return flag


def target_failure_consensus(engine, failed, message):
    """Target-only diagnostic/qualification gate, never a WORLD collective."""
    import torch.distributed as dist
    context=getattr(engine.model,'context',None)
    topology=getattr(engine,'topology',None)
    size=int(getattr(context,'size',len(getattr(topology,'target_ranks',(0,)))))
    device=getattr(engine,'device',engine.model.embed_tokens.weight.device)
    flag=torch.as_tensor(failed,device=device,dtype=torch.int64).reshape(1)
    if size>1:
        group=getattr(getattr(engine,'groups',None),'target_group',None)
        if group is None or not dist.is_initialized():
            raise RuntimeError('Mixed diagnostic failure gate requires the initialized target-only TP group')
        dist.all_reduce(flag,op=dist.ReduceOp.MAX,group=group)
    if bool(flag.item()):raise RuntimeError(message)


def install_native_fault_flag(native):
    """Extend the ORIGINAL all-rank finite vote input with the sticky flag."""
    cls=native.NativePearlEngine
    if getattr(cls,'_spec_slo_mixed_fault_hook_installed',False):return
    original=cls._spec_rhythm_nonfinite_flag
    def flag(engine,*,include_layer_cache=True):
        value=original(engine,include_layer_cache=include_layer_cache)
        mixed=getattr(engine,'_spec_slo_mixed_fault',None)
        return value if mixed is None else value.logical_or(mixed)
    cls._spec_rhythm_nonfinite_flag=flag
    cls._spec_slo_mixed_fault_hook_installed=True


def pad_full_query_extent(metadata, capacity):
    """Pad unused FULL query-mask rows only, preserving exact Q/KV lists."""
    if not getattr(metadata, 'tree_attention', False) or not metadata.use_fused_infer_attention:
        return metadata
    mask = metadata.tree_attention_mask
    if type(capacity) is not int or capacity < 2 or mask is None or mask.ndim != 4:
        raise ValueError('Invalid mixed FULL query-mask capacity')
    if mask.shape[2] > capacity:
        raise ValueError('Mixed FULL query extent exceeds separately qualified capacity')
    if mask.shape[2] == capacity:
        return metadata
    padded = mask.new_ones((*mask.shape[:2], capacity, mask.shape[3]))
    padded[:, :, :mask.shape[2], :].copy_(mask)
    return replace(metadata, tree_attention_mask=padded)


def prepare(engine, layout, *, full_query_capacity=128):
    """Use the ORIGINAL model metadata builder, with actual physical slots."""
    if engine.is_draft:
        raise ValueError('Mixed target forward belongs to target TP ranks only')
    if not isinstance(layout, MixedLayout) or not layout.prefill_spans:
        raise ValueError('An exact mixed verification/prompt layout is required')
    if engine.cache_allocation is None or engine.cache_block_tables is None:
        raise RuntimeError('Allocate native cache before mixed target execution')
    if layout.total_queries > engine.config.max_num_batched_tokens:
        raise ValueError('Mixed physical queries exceed native batching capacity')
    packed_ids, physical_positions = [], []
    for seq, plan, row in zip(layout.sequence_ids, layout.plans, layout.rows, strict=True):
        if len(row) != int(plan.parent_indices.numel()) or len(row) != plan.candidate_budget:
            raise ValueError('Mixed plans must contain only actual selected candidates')
        cache_positions = plan.cache_positions if plan.cache_positions is not None else plan.positions
        packed_ids.extend([seq] * len(cache_positions))
        physical_positions.extend(map(int, cache_positions.tolist()))
    engine._ensure_cache_capacity(packed_ids, physical_positions)
    tokens, positions, metadata = engine.model.make_tree_attention_metadata(
        list(layout.plans), list(layout.roots), [list(r) for r in layout.rows],
        engine.cache_allocation.block_tables, list(layout.sequence_ids),
        allow_causal_fast_path=True,
    )
    if int(tokens.numel()) != layout.total_queries:
        raise RuntimeError('Original metadata did not preserve mixed physical query count')
    if metadata.slot_mapping.unique().numel() != layout.total_queries:
        raise RuntimeError('Mixed target KV slots alias another request or scratch row')
    metadata = pad_full_query_extent(metadata, full_query_capacity)
    return PreparedMixedCall(layout, tokens, positions, metadata)


def graph_hidden(engine, call):
    """No fallback substitution: an eager/capacity/mismatch result fails."""
    hidden = engine.graph_runner.run_tree_hidden(call.input_ids, call.positions, call.metadata)
    execution = engine.graph_runner.last_target_execution
    bad=execution.mode not in ('capture_replay','replay') or not execution.used_aclgraph
    pending=getattr(engine,'_mixed_world_finite_vote_pending',False)
    if not pending or not engine.graph_runner.graph_cache_sealed:
        target_failure_consensus(engine,bad,'Mixed target did not pass the native ACLGraph execution gate')
    elif bad:
        raise RuntimeError('Mixed target did not pass the native ACLGraph execution gate')
    return hidden


def split_outputs(engine, call, hidden):
    """Original tree verifier format plus target-owned completed-prompt tokens."""
    layout = call.layout
    if hidden.ndim != 2 or hidden.shape[0] != layout.total_queries:
        raise ValueError('Mixed hidden states must cover every actual and scratch query')
    fault=record_hidden_fault(engine,hidden)
    if not getattr(engine,'_mixed_world_finite_vote_pending',False):
        target_failure_consensus(engine,fault,'Mixed target produced non-finite hidden states')
    heads = torch.tensor(layout.head_indices, dtype=torch.long, device=hidden.device)
    selected = engine.model.compute_greedy_tokens(hidden.index_select(0, heads), engine.draft_vocab_size)
    q = layout.verification_queries
    if selected.numel() != len(layout.head_indices):
        raise RuntimeError('Mixed vocabulary projection changed selected output count')
    target, query, bonus, counts = [], [], [], []
    cursor = 0
    for plan in layout.plans[:layout.verification_requests]:
        n = int(plan.candidate_budget)
        row = selected[cursor:cursor+n+1]
        target.append(row[:n]); query.append(row); bonus.append(row[n]); counts.append(n)
        cursor += n+1
    if cursor != q:
        raise RuntimeError('Mixed verification output boundary is inconsistent')
    completed = tuple(seq for seq, start, end, final in layout.prefill_spans if final)
    if len(completed) != selected.numel()-q:
        raise RuntimeError('Partial prompt chunks must not produce first tokens')
    tree = dict(target_token_ids=torch.cat(target), target_query_token_ids=torch.cat(query),
        bonus_token_ids=torch.stack(bonus), num_draft_tokens=counts, target_logits=None,
        used_aclgraph=True, used_tree_attention=bool(getattr(call.metadata, 'tree_attention', False)),
        attention_backend=attention_backend_identity(call.metadata),
        cache_slot_mapping=call.metadata.slot_mapping[:q],
        # query_count is TREE work, preserving the original controller's B accounting.
        # Physical mixed work is explicitly reported separately below.
        query_count=q, logical_query_count=q, graph_padding_query_count=0,
        tree_count=layout.verification_requests, profile_seconds={},
        mixed_prefill_query_count=layout.prefill_queries,
        mixed_scratch_query_count=layout.padding_queries,
        mixed_total_physical_query_count=layout.total_queries)
    return dict(tree=tree, prefill_request_ids=completed, prefill_token_ids=selected[q:])


def forward(engine, layout, *, full_query_capacity=128):
    call = prepare(engine, layout, full_query_capacity=full_query_capacity)
    return split_outputs(engine, call, graph_hidden(engine, call))


def independent_eager_hidden(engine, layout):
    """Reference: original native builder/model, one real request at a time.

    Call after saving graph outputs. Each request rewrites exactly the same
    private query slots and sees the same prior committed prefix. No ledger,
    graph or mixed metadata builder constructs this independent reference.
    """
    rows = []
    for plan, root, tokens, seq in zip(layout.plans, layout.roots, layout.rows,
                                     layout.sequence_ids, strict=True):
        ids, pos, meta = engine.model.make_tree_attention_metadata(
            [plan], [root], [list(tokens)], engine.cache_allocation.block_tables, [seq],
            allow_causal_fast_path=True,
        )
        rows.append(engine.model(ids, pos, meta))
    return torch.cat(rows)


def qualify(engine, call, *, synchronize):
    """Untimed all-hidden and exact-token oracle plus ORIGINAL changed-input gate.

    The caller must populate all committed prompt prefixes, including chunk
    positions before start. This helper cannot run after graph-cache sealing.
    It leaves the ORIGINAL actual-input KV contents restored at return.
    """
    runner = engine.graph_runner
    if runner.graph_cache_sealed:
        raise RuntimeError('Mixed numerical qualification cannot occur after sealing')
    actual = graph_hidden(engine, call)
    synchronize(); actual = actual.clone()
    target_failure_consensus(engine,record_hidden_fault(engine,actual),
        'Mixed qualification produced non-finite hidden states')
    reference = independent_eager_hidden(engine, call.layout)
    synchronize()
    target_failure_consensus(engine,not runner._target_outputs_match((actual,), (reference,)),
        'Mixed all-hidden output differs from independent per-request eager')
    graph_tokens = split_outputs(engine, call, actual)
    eager_tokens = split_outputs(engine, call, reference)
    for key in ('target_token_ids', 'target_query_token_ids', 'bonus_token_ids'):
        target_failure_consensus(engine,not torch.equal(graph_tokens['tree'][key],eager_tokens['tree'][key]),
            'Mixed exact tree token oracle failed: '+key)
    target_failure_consensus(engine,not torch.equal(graph_tokens['prefill_token_ids'],eager_tokens['prefill_token_ids']),
        'Mixed exact completed-prompt token oracle failed')
    changed = call.input_ids.clone()
    changed[0] = (changed[0]+1) % engine.draft_vocab_size
    graph_hidden(engine, replace(call, input_ids=changed))
    restored = graph_hidden(engine, call)
    synchronize()
    target_failure_consensus(engine,not runner._target_outputs_match((restored,), (reference,)),
        'Mixed actual-input restoration changed hidden states')
    # The graph runner—not this helper—sets runtime_validated. Formal replay
    # still requires the original runner seal and trace fixed-point audit.
    return dict(passed=True, all_hidden_equal=True, exact_verifier_tokens_equal=True,
        exact_completed_prompt_tokens_equal=True, actual_KV_restored=True,
        native_changed_input_attempted=True, formal_fixed_point_qualified=False,
        actual_tree_predictions=graph_tokens['tree']['target_query_token_ids'].cpu().tolist(),
        actual_completed_prompt_tokens=graph_tokens['prefill_token_ids'].cpu().tolist())
