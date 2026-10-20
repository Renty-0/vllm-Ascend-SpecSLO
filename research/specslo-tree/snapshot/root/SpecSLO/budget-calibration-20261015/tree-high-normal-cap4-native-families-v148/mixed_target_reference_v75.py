"""Untimed genuine FULL197/S35 reference and original Native restore gates.

All public orchestration functions are called on all four ranks. Target-only
model/TP work is contained in the supplied original stage_all4 coordinator.
The 35 reference owners have independently rebuilt histories; candidate KV
never initializes a reference. This module supplies no online/cost receipt.
"""
from dataclasses import fields, replace
import hashlib

import torch

import mixed_target


M = 197
S = 35
MASK = (35, 1, 128, 4096)
COUNTERS = ('total_calls','capture_replay_calls','replay_calls','eager_fallback_calls',
    'disabled_entry_calls','unclassified_calls','runtime_validation_calls',
    'runtime_validation_failures','changed_input_validation_calls',
    'logical_row_expansion_validation_calls')
MC2_COUNTERS = ('fused_attempt','fused_success','fallback','exception','full_fused_attempt',
    'full_fused_success','native_epilogue_attempt','native_epilogue_success',
    'native_epilogue_chained_attempt','native_epilogue_chained_success','native_epilogue_chain_flush')


class ReferenceContractError(RuntimeError):
    pass


def _gate(engine, failed, message):
    mixed_target.target_failure_consensus(engine, failed, message)


def span_bounds(layout):
    S=len(layout.plans); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    lengths = [int(plan.candidate_budget)+1 for plan in layout.plans]
    if len(lengths) != S or sum(lengths) != M:
        raise ReferenceContractError('V78 reference requires all35 spans/all197 physical queries')
    result=[];cursor=0
    for length in lengths:
        result.append((cursor,cursor+length));cursor+=length
    return tuple(result)


def independent_plan(plan):
    """Re-derive exact ancestor visibility, physical slots and logical RoPE."""
    parents=[int(v) for v in plan.parent_indices.tolist()]
    prefix=int(plan.prefix_len);levels=[]
    for index,parent in enumerate(parents):
        if not -1 <= parent < index:
            raise ReferenceContractError('Independent ancestor order is invalid')
        levels.append(1 if parent < 0 else levels[parent]+1)
    positions=torch.tensor([prefix,*[prefix+level for level in levels]],dtype=plan.positions.dtype)
    slots=torch.arange(prefix,prefix+len(parents)+1,dtype=plan.cache_positions.dtype)
    mask=torch.ones((len(parents)+1,int(plan.max_model_len)),dtype=torch.bool)
    mask[:,:prefix]=False;mask[:,prefix]=False
    for index in range(len(parents)):
        ancestor=index
        while ancestor >= 0:
            mask[index+1,prefix+ancestor+1]=False
            ancestor=parents[ancestor]
    if (not torch.equal(positions,plan.positions) or not torch.equal(slots,plan.cache_positions)
            or not torch.equal(mask,plan.attention_mask)):
        raise ReferenceContractError('Native plan differs from independently derived ancestry/positions/slots')
    if not parents:
        from flexible_layout import RootOnlyPromptPlan
        if (type(plan) is not RootOnlyPromptPlan or plan.candidate_budget != 0
                or plan.width != 1 or plan.depth != 0):
            raise ReferenceContractError('Only the typed real prompt plan may own one root-only query')
        # Its tensor fields are init=False.  A fresh typed constructor owns
        # new buffers; dataclasses.replace(tensor_field=...) is invalid here.
        rebuilt=RootOnlyPromptPlan(prefix,int(plan.max_model_len))
        if (not torch.equal(rebuilt.positions,positions)
                or not torch.equal(rebuilt.cache_positions,slots)
                or not torch.equal(rebuilt.attention_mask,mask)
                or rebuilt.parent_indices.dtype != plan.parent_indices.dtype
                or rebuilt.positions.dtype != plan.positions.dtype
                or rebuilt.cache_positions.dtype != plan.cache_positions.dtype):
            raise ReferenceContractError('Fresh typed root-only prompt differs from independent ancestry/slots')
        return rebuilt
    return replace(plan,positions=positions,cache_positions=slots,attention_mask=mask,
        parent_indices=torch.tensor(parents,dtype=plan.parent_indices.dtype))


def clone_call(call):
    """Own all metadata buffers before another builder or replay can reuse them."""
    copies={field.name:getattr(call.metadata,field.name).clone()
        for field in fields(call.metadata) if isinstance(getattr(call.metadata,field.name),torch.Tensor)}
    return replace(call,input_ids=call.input_ids.clone(),positions=call.positions.clone(),
        metadata=replace(call.metadata,**copies))


def _prefixes(epoch, key):
    S=len(epoch['candidate_span_ids']); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    values=epoch[key]
    if len(values) != S or any(not isinstance(row,(list,tuple)) or
            any(type(token) is not int or token < 0 for token in row) for row in values):
        raise ReferenceContractError('Every physical owner needs its actual immutable-prefix tokens: '+key)
    return values


def phase_for_layout(layout,epoch):
    """Use the actual variable-length history phase, never a start/else guess."""
    S=len(layout.plans); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    index=epoch['next_prompt_phase'];scope=epoch['scope'];phases=scope.phases()
    if (type(index) is not int or not 0<=index<len(phases)
            or scope.target_requests!=layout.verification_requests):
        raise ReferenceContractError('Mixed reference has no current declared history phase')
    phase=phases[index]
    if (phase['phase_index']!=index or phase['physical_M']!=M or phase['physical_S']!=S
            or tuple(phase['mask_shape'])!=MASK
            or len(layout.prefill_spans)!=1
            or tuple(layout.prefill_spans[0])!=(epoch['prompt_id'],phase['prompt_start'],
                phase['prompt_end'],phase['final'])):
        raise ReferenceContractError('Mixed reference phase/cursor/final ownership differs from the actual history')
    return index,phase


def validate_layout(layout,epoch):
    S=len(layout.plans); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    from vllm_ascend.spec_decode.pearl.native_model import _is_contiguous_linear_tree_plan
    bounds=span_bounds(layout);n=layout.verification_requests
    if not 1<=n<=32 or len(layout.prefill_spans) != 1:
        raise ReferenceContractError('V78 target admits only N17B5/N32B3 trees plus one real prompt')
    _,phase=phase_for_layout(layout,epoch)
    seq,start,end,final=layout.prefill_spans[0]
    if (seq != epoch['prompt_id'] or seq != n
            or tuple(layout.sequence_ids) != tuple(epoch['candidate_span_ids'])
            or list(epoch['candidate_tree_ids']) != list(range(n))
            or layout.verification_queries != phase['tree_queries'] or layout.prefill_queries != end-start
            or len(layout.head_indices) != phase['tree_queries']+int(final)):
        raise ReferenceContractError('Mixed layout left the exact variable-phase real source/ownership/head scope')
    expected_padding=M-phase['tree_queries']-(end-start)
    actual_pads=tuple(epoch['actual_padding_ids']);reference_pads=tuple(epoch['reference_padding_ids'])
    counts=tuple(int(plan.candidate_budget)+1 for plan in layout.plans)
    if (layout.padding_queries != expected_padding or len(layout.plans)!=S
            or len(actual_pads)!=S-1-n or len(reference_pads)!=S-1-n
            or tuple(layout.sequence_ids[n+1:])!=actual_pads
            or counts!=tuple(phase['query_partition'])
            or sum(counts[n+1:])!=expected_padding
            or any(not 2<=count<=128 for count in counts[n+1:])):
        raise ReferenceContractError('Mixed scratch queries cannot extend real trees or prompt chunks')
    if tuple(layout.head_indices) != (*range(phase['tree_queries']),*((bounds[n][1]-1,) if final else ())):
        raise ReferenceContractError('Only real tree successors and final prompt head own predictions')
    actual=list(layout.sequence_ids);refs=list(epoch['reference_span_ids'])
    if (len(refs)!=S or len(set(refs))!=S or set(actual)&set(refs)
            or epoch['reference_prompt_id'] != refs[n]
            or list(epoch['reference_tree_ids']) != refs[:n]
            or tuple(refs[n+1:])!=reference_pads):
        raise ReferenceContractError('Reference ownership must be19 disjoint owners, preserving every scratch span')
    for key in ('candidate_prefixes','reference_prefixes'):
        prefixes=_prefixes(epoch,key)
        if any(len(tokens)!=int(plan.prefix_len) for tokens,plan in zip(prefixes,layout.plans,strict=True)):
            raise ReferenceContractError('Current history cursor/prefix differs from actual per-span plan: '+key)
    for index,plan in enumerate(layout.plans):
        independent_plan(plan)
        if index<n and (plan.candidate_budget!=phase['tree_budgets'][index] or (phase['tree_budgets'][index]==5 and _is_contiguous_linear_tree_plan(plan))):
            raise ReferenceContractError('Real B5 trees must retain a genuine fork')
    return bounds


def _table_rows(engine,ids,lengths):
    size=int(engine.config.kvcache_block_size);tables=engine.cache_allocation.block_tables
    pages=[];result=[]
    for seq,length in zip(ids,lengths,strict=True):
        row=[int(v) for v in tables[seq][:(length+size-1)//size]]
        if any(v<0 or v>=544 for v in row) or len(set(row))!=len(row):
            raise ReferenceContractError('Every used logical page must be allocated and unique')
        pages.extend(row);result.append(row)
    if len(set(pages))!=len(pages):
        raise ReferenceContractError('Independent candidate/reference live pages alias')
    return result


def protect_prefixes(engine,ids,prefixes):
    """All64 used-prefix K/V finite and byte hashes, never unallocated slots."""
    if len(engine.model.layers)!=64 or len(ids)!=len(prefixes):
        raise ReferenceContractError('Prefix protection must cover all64 target layers and aligned owners')
    size=int(engine.config.kvcache_block_size);tables=engine.cache_allocation.block_tables
    slots=[]
    for seq,tokens in zip(ids,prefixes,strict=True):
        indices=[]
        for position in range(len(tokens)):
            block=int(tables[seq][position//size])
            if not 0<=block<544:
                raise ReferenceContractError('Protected prefix references an unallocated page')
            indices.append(block*size+position%size)
        slots.append(indices)
    hashes={};finite=torch.ones((),dtype=torch.bool,device=engine.device)
    for layer_index,layer in enumerate(engine.model.layers):
        for name in ('key_cache','value_cache'):
            storage=getattr(layer.self_attn,name).flatten(0,1)
            for seq,indices in zip(ids,slots,strict=True):
                values=storage.index_select(0,torch.tensor(indices,dtype=torch.long,device=storage.device))
                finite.logical_and_(torch.isfinite(values).all())
                hashes[f'{layer_index}:{name}:{seq}']=hashlib.sha256(
                    values.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    _gate(engine,~finite,'V75 protected source prefix contains non-finite target KV')
    return dict(hashes=hashes,all_used_prefix_finite=True,layers=64,owner_ids=list(ids),
        prefix_lengths=[len(tokens) for tokens in prefixes])


def query_KV(engine,metadata):
    if len(engine.model.layers)!=64:
        raise ReferenceContractError('Mixed oracle requires every actual64 target layer')
    slots=metadata.slot_mapping.to(torch.long)
    if (slots.numel()!=M or bool((slots<0).any()) or slots.unique().numel()!=M):
        raise ReferenceContractError('FULL scratch is real legal query work, never slot=-1 or aliased')
    result=[];finite=torch.ones((),dtype=torch.bool,device=engine.device)
    for layer in engine.model.layers:
        for name in ('key_cache','value_cache'):
            values=getattr(layer.self_attn,name).flatten(0,1).index_select(0,slots).clone()
            result.append(values);finite.logical_and_(torch.isfinite(values).all())
    _gate(engine,~finite,'V75 full197 all64-layer query KV contains non-finite values')
    return tuple(result)


def _check_metadata(call, *, expected=None):
    S=len(call.layout.plans); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    metadata=call.metadata
    if (call.input_ids.ndim!=1 or call.input_ids.numel()!=M or tuple(call.positions.shape)!=(M,)
            or not metadata.tree_attention or not metadata.use_fused_infer_attention
            or tuple(metadata.tree_attention_mask.shape)!=MASK
            or tuple(metadata.request_block_tables.shape)!=(S,32)
            or tuple(metadata.block_tables.shape)!=(S,32)
            or metadata.attention_mask is not None
            or len(metadata.actual_seq_lengths_q)!=S or metadata.actual_seq_lengths_q[-1]!=M
            or len(metadata.sequence_lens)!=S or metadata.slot_mapping.unique().numel()!=M
            or bool((metadata.slot_mapping<0).any())):
        raise ReferenceContractError('Actual FULL197/S35/mask128 metadata ABI is required')
    if expected is not None and (metadata.actual_seq_lengths_q!=expected.actual_seq_lengths_q
            or metadata.sequence_lens!=expected.sequence_lens
            or not torch.equal(metadata.context_lens,expected.context_lens)):
        raise ReferenceContractError('Independent reference changed actual full Q/KV length vectors')
    return call


def check_call_ownership(engine,call):
    """Independently map every real/scratch token to its actual private slot."""
    if int(engine.config.kvcache_block_size)!=128:
        raise ReferenceContractError('The actual Native FULL contract uses128-token pages')
    tokens=[];positions=[];slots=[];contexts=[];bounds=span_bounds(call.layout)
    tables=engine.cache_allocation.block_tables
    for index,(owner,plan,root,row) in enumerate(zip(call.layout.sequence_ids,call.layout.plans,
            call.layout.roots,call.layout.rows,strict=True)):
        checked=independent_plan(plan);tokens.extend([root,*row])
        positions.extend(int(v) for v in checked.positions.tolist())
        for position in checked.cache_positions.tolist():
            page=int(tables[owner][int(position)//128])
            if not 0<=page<544:raise ReferenceContractError('A logical mixed query page is unallocated')
            slots.append(page*128+int(position)%128);contexts.append(int(position)+1)
        expected_table=torch.as_tensor(tables[owner],device=call.metadata.request_block_tables.device,
            dtype=call.metadata.request_block_tables.dtype)
        mask=call.metadata.tree_attention_mask[index,0];q=bounds[index][1]-bounds[index][0]
        if (not torch.equal(call.metadata.request_block_tables[index],expected_table)
                or not torch.equal(mask[:q],checked.attention_mask.to(mask.device))
                or not bool(mask[q:].all())):
            raise ReferenceContractError('FULL mask or private request block table changed logical owner')
    metadata=call.metadata
    if (call.input_ids.tolist()!=tokens or call.positions.tolist()!=positions
            or metadata.slot_mapping.tolist()!=slots or metadata.context_lens.tolist()!=contexts
            or not torch.equal(metadata.block_tables,metadata.request_block_tables)):
        raise ReferenceContractError('Mixed exact tokens/RoPE/cache-slot/context ownership is inconsistent')
    return True


def prepare_reference(engine,actual,epoch,span=None):
    """One retained span, or the independent all-real mirror, at identical ABI."""
    S=len(actual.layout.plans); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    bounds=span_bounds(actual.layout);refs=tuple(epoch['reference_span_ids'])
    if span is not None and (type(span) is not int or not 0<=span<S):
        raise ReferenceContractError('Invalid retained reference span')
    plans=[];roots=[];rows=[]
    for index,(plan,root,row) in enumerate(zip(actual.layout.plans,actual.layout.roots,actual.layout.rows,strict=True)):
        if span is None or index==span:
            plans.append(independent_plan(plan));roots.append(root);rows.append(tuple(row))
        else:
            count=int(plan.candidate_budget)
            if count==0:
                from flexible_layout import RootOnlyPromptPlan
                # The non-retained final-Q1 prompt is still one real causal
                # query on a separate owner, never a budget0 verification tree.
                dummy=RootOnlyPromptPlan(int(plan.prefix_len),int(plan.max_model_len))
                plans.append(independent_plan(dummy))
            else:
                plans.append(cached_cpu_tree_speculation_plan(1,count,int(plan.prefix_len),4096,candidate_budget=count))
            roots.append(0);rows.append((0,)*count)
    owners=[];physical=[]
    for seq,plan in zip(refs,plans,strict=True):
        values=[int(v) for v in plan.cache_positions.tolist()]
        owners.extend([seq]*len(values));physical.extend(values)
    engine._ensure_cache_capacity(owners,physical)
    tokens,positions,metadata=engine.model.make_tree_attention_metadata(plans,roots,[list(r) for r in rows],
        engine.cache_allocation.block_tables,list(refs),allow_causal_fast_path=False)
    metadata=mixed_target.pad_full_query_extent(metadata,128)
    prompt=actual.layout.prefill_spans[0]
    reference_layout=replace(actual.layout,plans=tuple(plans),roots=tuple(roots),rows=tuple(rows),
        sequence_ids=refs,prefill_spans=((refs[actual.layout.verification_requests],*prompt[1:]),))
    call=clone_call(mixed_target.PreparedMixedCall(reference_layout,tokens,positions,metadata))
    _check_metadata(call,expected=actual.metadata)
    check_call_ownership(engine,call)
    if bool(torch.isin(call.metadata.slot_mapping,actual.metadata.slot_mapping).any()):
        raise ReferenceContractError('Reference query slots alias the candidate cache')
    indices=range(S) if span is None else (span,)
    for index in indices:
        lo,hi=bounds[index]
        if (not torch.equal(call.input_ids[lo:hi],actual.input_ids[lo:hi])
                or not torch.equal(call.positions[lo:hi],actual.positions[lo:hi])
                or not torch.equal(call.metadata.tree_attention_mask[index],actual.metadata.tree_attention_mask[index])):
            raise ReferenceContractError('Reference retained real tokens/logical positions/ancestor mask changed')
    return call


def check_selected_compaction(engine,layout,accepted_paths,reference_query_KV):
    """Compare original candidate compaction against independent sparse KV."""
    if engine.is_draft:return None
    n=layout.verification_requests;bounds=span_bounds(layout)
    if len(engine.model.layers)!=64 or len(accepted_paths)!=n or len(reference_query_KV)!=128:
        raise ReferenceContractError('All real paths and64-layer independent query KV are required')
    matcher=engine.graph_runner._target_outputs_match;size=int(engine.config.kvcache_block_size)
    actual=[];reference=[]
    for layer_index,layer in enumerate(engine.model.layers):
        for kind,name in enumerate(('key_cache','value_cache')):
            cache=getattr(layer.self_attn,name).flatten(0,1)
            values=reference_query_KV[2*layer_index+kind]
            if values.shape[0]!=M:
                raise ReferenceContractError('Compaction oracle lost full physical197 row identity')
            for row,path in enumerate(accepted_paths):
                chosen=[int(v) for v in path if v>=0]
                if (list(path)!=chosen+[-1]*(len(path)-len(chosen))
                        or any(v>=layout.plans[row].candidate_budget for v in chosen)):
                    raise ReferenceContractError('Accepted paths must retain original packed indices')
                parent=-1
                for node in chosen:
                    if int(layout.plans[row].parent_indices[node])!=parent:
                        raise ReferenceContractError('Independent compaction path is not ancestor closed')
                    parent=node
                seq=layout.sequence_ids[row];prefix=int(layout.plans[row].prefix_len)
                table=engine.cache_allocation.block_tables[seq]
                slots=[int(table[p//size])*size+p%size for p in range(prefix,prefix+len(chosen)+1)]
                if any(value<0 for value in slots):
                    raise ReferenceContractError('Compaction destination references an unallocated page')
                actual.append(cache.index_select(0,torch.tensor(slots,dtype=torch.long,device=cache.device)))
                indices=[bounds[row][0],*[bounds[row][0]+node+1 for node in chosen]]
                reference.append(values[indices])
    failed=any(not bool(torch.isfinite(value).all()) for value in (*actual,*reference))
    _gate(engine,failed or not matcher(tuple(actual),tuple(reference)),
        'Original mixed tree compaction differs from independently recomputed same-sparse source KV')
    return dict(passed=True,all64_target_layers=True,all_real_rows=n,
        independent_source_query_KV=True,perrow_original_query_offset=True,
        prompt_and_scratch_not_treated_as_tree_rows=True,
        scope='same sparse physical layout local observed commit only',dense_AR_qualified=False)


def _counters(engine):
    values=engine.graph_runner.execution_counters['target']
    if set(values)!=set(COUNTERS) or any(type(v) is not int or v<0 for v in values.values()):
        raise ReferenceContractError('Every original target Native execution counter must be present')
    return dict(values)


def _dispatch():
    from vllm_ascend.spec_decode.pearl.mc2 import snapshot_mc2_dispatch_counters
    values=snapshot_mc2_dispatch_counters()
    if set(values)!=set(MC2_COUNTERS) or any(type(v) is not int or v<0 for v in values.values()):
        raise ReferenceContractError('Every original11 MC2 counter must be present')
    return dict(values)


def _delta(before,after):
    if set(before)!=set(after) or any(after[k]<before[k] for k in before):
        raise ReferenceContractError('Original counters lost fields or decreased')
    return {key:after[key]-value for key,value in before.items()}


def _execution(engine):
    actual=engine.graph_runner.last_target_execution
    value=dict(mode=actual.mode,used_aclgraph=actual.used_aclgraph)
    if actual.mode not in ('capture_replay','replay') or actual.used_aclgraph is not True:
        raise ReferenceContractError('Mixed hidden forward must use original Native graph execution')
    return value


def _entry(engine,call):
    S=len(call.layout.plans); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    runner=engine.graph_runner;key=runner._frontier_qualification_last_key
    signature=((tuple(call.metadata.tree_attention_mask.shape),tuple(call.metadata.request_block_tables.shape)),)
    expected=(f'target-hidden:0|steps:1|tree-fia-full:exact-length-task-update|buffers:{signature}',(M,))
    entry=runner.target_entries.get(key)
    if (key!=expected or entry is None or len(entry.input_ids)!=1
            or tuple(entry.input_ids[0].shape)!=(M,) or entry.tree_attention_modes!=(True,)
            or tuple(entry.request_block_tables[0].shape)!=(S,32)
            or tuple(entry.tree_attention_masks[0].shape)!=MASK
            or len(entry.tasks)!=64 or entry.tasks_per_step!=64
            or getattr(entry,'validated_real_row_count',None) is not None
            or type(entry.runtime_validated) is not bool):
        raise ReferenceContractError('Actual Native hidden entry/key must describe FULL197/S35/mask128/64tasks')
    if (len({id(task.handle) for task in entry.tasks})!=64
            or len({id(task.event) for task in entry.tasks})!=64):
        raise ReferenceContractError('Every original target layer retains its own attention handle/event')
    for task in entry.tasks:
        if (task.handle is None or task.event is None or not task.tree_attention
                or tuple(task.query.shape[:1])!=(M,) or tuple(task.output.shape[:1])!=(M,)
                or tuple(task.block_table.shape)!=(S,32) or tuple(task.attention_mask.shape)!=MASK):
            raise ReferenceContractError('Original64 attention task geometries differ from actual mixed input')
    return dict(key=repr(key),object_id=id(entry),runtime_validated=entry.runtime_validated,
        validated_real_row_count=getattr(entry,'validated_real_row_count',None),
        native_real_row_field_present=hasattr(entry,'validated_real_row_count'),physical_M=M,segments=S,
        mask_shape=list(MASK),original_task_count=len(entry.tasks))


def _graph(engine,call):
    hidden=engine.graph_runner.run_tree_hidden(call.input_ids,call.positions,call.metadata)
    saved=hidden.clone()  # First device operation after the actual original API.
    _gate(engine,tuple(saved.shape[:1])!=(M,) or saved.ndim!=2 or ~torch.isfinite(saved).all(),
        'Actual mixed graph hidden must be finite and cover all197 rows')
    global_validation=engine.graph_runner.runtime_validation_replay_count
    if type(global_validation) is not int or global_validation<0:
        raise ReferenceContractError('Original global Native validation count is unavailable')
    return dict(returned=hidden,saved=saved,KV=query_KV(engine,call.metadata),
        entry=_entry(engine,call),execution=_execution(engine),counters=_counters(engine),
        runtime_validation_replays=global_validation)


def _copy_output(output):
    result=dict(output);result['tree']={key:value.clone() if isinstance(value,torch.Tensor) else value
        for key,value in output['tree'].items()}
    result['prefill_token_ids']=output['prefill_token_ids'].clone()
    return result


def _heads_match(engine,a,b):
    same=(a['prefill_token_ids'].shape==b['prefill_token_ids'].shape
        and torch.equal(a['prefill_token_ids'],b['prefill_token_ids']))
    for key in ('target_token_ids','target_query_token_ids','bonus_token_ids'):
        same=same and torch.equal(a['tree'][key],b['tree'][key])
    _gate(engine,not same,'Mixed exact tree/final-prompt head predictions differ')
    return True


def _match(engine,a,b,message):
    matcher=engine.graph_runner._target_outputs_match
    _gate(engine,not matcher(tuple(a),tuple(b)),message)
    return True


def _reference_round(engine,actual,baseline,original_output,epoch,phase_index,stage_all4):
    """Callback for the independent-history actor; all four ranks participate."""
    S=len(epoch['reference_span_ids']); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    target=lambda label,fn:stage_all4(label,lambda:fn() if not engine.is_draft else None)
    bounds=span_bounds(actual.layout) if actual is not None else None
    prefixes=tuple(tuple(row) for row in epoch['reference_prefixes'])
    before=target('V75-independent-reference-prefix-before',lambda:
        protect_prefixes(engine,epoch['reference_span_ids'],prefixes))
    hidden_parts=[];KV_parts=[];calls=[]
    isolated_required=epoch['scope'].normal_prefix_tokens is None
    for span in (range(S) if isolated_required else ()):
        call=target(f'V75-independent-isolation-{span}-metadata',lambda span=span:
            prepare_reference(engine,actual,epoch,span))
        def evaluate(call=call,span=span):
            dispatch_before=_dispatch()
            hidden=engine.model(call.input_ids,call.positions,call.metadata)
            saved=hidden.clone()
            _gate(engine,~torch.isfinite(saved).all(),'Independent isolated FULL197 hidden is non-finite')
            KV=query_KV(engine,call.metadata);lo,hi=bounds[span]
            dispatch_after=_dispatch()
            return dict(hidden=saved[lo:hi].clone(),KV=tuple(value[lo:hi].clone() for value in KV),
                record=dict(span=span,retained_rows=[lo,hi],physical_M=M,segments=S,
                    actual_seq_lengths_q=list(call.metadata.actual_seq_lengths_q),
                    sequence_lens=list(call.metadata.sequence_lens),mask_shape=list(MASK),
                    original_builder_forced_FULL=True,before=dispatch_before,after=dispatch_after,
                    delta=_delta(dispatch_before,dispatch_after)))
        value=target(f'V75-independent-isolation-{span}-eager',evaluate)
        if not engine.is_draft:
            hidden_parts.append(value['hidden']);KV_parts.append(value['KV']);calls.append(value['record'])
    mirror=target('V75-independent-all-real-mirror-metadata',lambda:
        prepare_reference(engine,actual,epoch,None))
    def mirror_forward():
        isolated_hidden=torch.cat(hidden_parts,dim=0) if isolated_required else None
        isolated_KV=tuple(torch.cat([span[index] for span in KV_parts],dim=0) for index in range(128)) if isolated_required else None
        dispatch_before=_dispatch()
        hidden=engine.model(mirror.input_ids,mirror.positions,mirror.metadata)
        saved=hidden.clone()
        _gate(engine,~torch.isfinite(saved).all(),'Independent all-real FULL197 mirror is non-finite')
        KV=query_KV(engine,mirror.metadata);dispatch_after=_dispatch()
        if isolated_required:
            _match(engine,(saved,),(isolated_hidden,),'Independent mirror differs from isolated full197 hidden')
            _match(engine,KV,isolated_KV,'Independent mirror differs from isolated all64 query KV')
        output=_copy_output(mixed_target.split_outputs(engine,mirror,hidden))
        _match(engine,(baseline['saved'],),(saved,),'Actual mixed hidden differs from independent mirror')
        if isolated_required:_match(engine,(baseline['saved'],),(isolated_hidden,),'Actual mixed hidden differs from independent retained spans')
        _match(engine,baseline['KV'],KV,'Actual all64 query KV differs from independent mirror')
        if isolated_required:_match(engine,baseline['KV'],isolated_KV,'Actual all64 query KV differs from independent retained spans')
        _heads_match(engine,original_output,output)
        if isolated_required:
            isolated_output=mixed_target.split_outputs(engine,mirror,isolated_hidden)
            _heads_match(engine,original_output,isolated_output)
        return dict(hidden=saved,KV=KV,isolated_hidden=isolated_hidden,isolated_KV=isolated_KV,output=output,
            dispatch=dict(before=dispatch_before,after=dispatch_after,delta=_delta(dispatch_before,dispatch_after)))
    result=target('V75-independent-all-real-mirror-forward',mirror_forward)
    after=target('V75-independent-reference-prefix-after',lambda:
        protect_prefixes(engine,epoch['reference_span_ids'],prefixes))
    target('V75-independent-reference-prefix-unchanged',lambda:_gate(engine,before!=after,
        'Independent sameM isolation/mirror changed an already-valid reference prefix'))
    def record():
        seq,start,end,final=actual.layout.prefill_spans[0];n=actual.layout.verification_requests
        first_token=int(result['output']['prefill_token_ids'][0].item()) if final else None
        prompt_tokens=[int(v) for v in actual.input_ids[bounds[n][0]:bounds[n][1]].cpu().tolist()]
        return dict(passed=True,source_prefix_finite=True,all_physical_hidden_original_matcher=True,
            all64_query_KV_original_matcher=True,exact_target_heads=True,mirror_recomputed=True,
            copied_candidate_KV=False,phase_index=phase_index,prompt_start=start,prompt_end=end,
            reference_span_ids=list(epoch['reference_span_ids']),reference_prompt_id=epoch['reference_prompt_id'],
            physical_M=M,segments=S,mask_capacity=128,source_input_sha256=epoch['source_input_sha256'],
            input_prompt_tokens=prompt_tokens,first_token=first_token,reference_query_KV=result['KV'],
            reference_hidden=result['hidden'],isolated_hidden=result['isolated_hidden'],
            isolated_query_KV=result['isolated_KV'],reference_output=result['output'],
            prefix_before=before,prefix_after=after,span_bounds=bounds,independent_eager_calls=calls,
            independent_mirror_dispatch=result['dispatch'],query_partition=list(mirror.metadata.actual_seq_lengths_q),
            KV_lengths=list(mirror.metadata.sequence_lens))
    return target('V75-independent-reference-callback-record',record)


def qualify_mixed_round(engine,layout,epoch,*,stage_all4,changed=False):
    """Actual Native full197 call, independent reference and actual-input restore.

    Independent prompt history advances through its own actor. Original tree
    commit/publication remains outside this function and must succeed later.
    Returned GPU snapshots are private comparison evidence, never a receipt.
    """
    S=len(layout.plans); MASK=(S,1,128,4096)
    if S not in (19,35):raise ReferenceContractError('Only genuine S19/S35 layouts are supported')
    from independent_history_v75 import advance_prompt_reference
    stage_all4('V75-mixed-round-layout-contract',lambda:validate_layout(layout,epoch))
    target=lambda label,fn:stage_all4(label,lambda:fn() if not engine.is_draft else None)
    def prepare():
        if engine.graph_runner.graph_cache_sealed:
            raise ReferenceContractError('New mixed qualification cannot run after Native seal')
        from flexible_target_v78 import prepare_mixed_call
        call=_check_metadata(clone_call(prepare_mixed_call(engine,layout,epoch)))
        check_call_ownership(engine,call)
        lengths=[int(plan.prefix_len)+int(plan.candidate_budget)+1 for plan in layout.plans]
        _table_rows(engine,[*epoch['candidate_span_ids'],*epoch['reference_span_ids']],lengths*2)
        return call
    call=target('V75-actual-mixed-call-metadata',prepare)
    before=target('V75-actual-mixed-prefix-before',lambda:
        protect_prefixes(engine,epoch['candidate_span_ids'],epoch['candidate_prefixes']))
    dispatch_before=target('V75-actual-mixed-MC2-before',_dispatch)
    actual=target('V75-actual-original-mixed-hidden',lambda:_graph(engine,call))
    original_output=target('V75-actual-original-mixed-heads',lambda:
        _copy_output(mixed_target.split_outputs(engine,call,actual['returned'])))
    alternate=target('V75-unpublished-original-input-change',lambda:replace(call,
        input_ids=(call.input_ids+1)%int(engine.draft_vocab_size)))
    changed_call=target('V75-original-native-changed-input',lambda:_graph(engine,alternate))
    restored=target('V75-original-native-input-restore',lambda:_graph(engine,call))
    def native_gate():
        a,b,c=actual,changed_call,restored
        if (not b['entry']['runtime_validated'] or not c['entry']['runtime_validated']
                or a['entry']['key']!=b['entry']['key'] or a['entry']['key']!=c['entry']['key']
                or len({a['entry']['object_id'],b['entry']['object_id'],c['entry']['object_id']})!=1
                or b['execution']!={'mode':'replay','used_aclgraph':True}
                or c['execution']!={'mode':'replay','used_aclgraph':True}):
            raise ReferenceContractError('Original same-key changed/restore Native runtime gate remains open')
        change=_delta(a['counters'],b['counters']);restore=_delta(b['counters'],c['counters'])
        cold=a['entry']['runtime_validated'] is False
        if (b['runtime_validation_replays']-a['runtime_validation_replays']!=int(cold)
                or c['runtime_validation_replays']!=b['runtime_validation_replays']):
            raise ReferenceContractError('Original global Native changed/restore validation delta differs')
        for key in COUNTERS:
            wanted=1 if key in ('total_calls','replay_calls') or (cold and key in
                ('runtime_validation_calls','changed_input_validation_calls')) else 0
            if change[key]!=wanted or restore[key]!=(1 if key in ('total_calls','replay_calls') else 0):
                raise ReferenceContractError('Actual Native changed/restore counters differ: '+key)
        _match(engine,(a['saved'],),(c['saved'],),'Mixed input restoration changed original full197 hidden')
        _match(engine,a['KV'],c['KV'],'Mixed input restoration changed original all64 query KV')
        output=mixed_target.split_outputs(engine,call,c['returned']);_heads_match(engine,original_output,output)
        return dict(actual_entry=a['entry'],changed_entry=b['entry'],restored_entry=c['entry'],
            actual_execution=a['execution'],changed_execution=b['execution'],restored_execution=c['execution'],
            counters_before_changed=a['counters'],counters_after_changed=b['counters'],counters_after_restore=c['counters'],
            changed_delta=change,restore_delta=restore,cold_runtime_gate_closed=cold,
            global_runtime_validation_replays=[v['runtime_validation_replays'] for v in (a,b,c)],
            original_native_flags_assigned=False,hidden_validated_real_row_count=c['entry']['validated_real_row_count'])
    native=target('V75-original-native-changed-and-restore-gate',native_gate)
    dispatch_after=target('V75-actual-mixed-MC2-after',_dispatch)
    stage_all4('V75-independent-history-callback-ready',lambda:None)
    phase_index,_=phase_for_layout(layout,epoch)
    reference=advance_prompt_reference(engine,epoch,phase_index,
        recompute_same_m=lambda e,ep,phase:_reference_round(e,call,actual,original_output,ep,phase,stage_all4),stage_all4=stage_all4)
    # The history actor may return a proof wrapper; the callback's exact result
    # is retained locally, without relying on a boolean-only qualification.
    def reference_contract():
        if not engine.is_draft and (not isinstance(reference,dict) or 'reference_query_KV' not in reference):
            raise ReferenceContractError('Independent history actor must return its actual sameM reference callback result')
    stage_all4('V75-independent-reference-result-contract',reference_contract)
    def compare():
        _match(engine,(actual['saved'],),(reference['reference_hidden'],),'Actual mixed hidden differs from fresh independent all-real mirror')
        if epoch['scope'].normal_prefix_tokens is None:_match(engine,(actual['saved'],),(reference['isolated_hidden'],),'Actual mixed hidden differs from independent retained-span oracle')
        _match(engine,actual['KV'],reference['reference_query_KV'],'Actual all64 physical query KV differs from independent mirror')
        if epoch['scope'].normal_prefix_tokens is None:_match(engine,actual['KV'],reference['isolated_query_KV'],'Actual all64 physical query KV differs from isolated-span reference')
        _heads_match(engine,original_output,reference['reference_output'])
        return True
    target('V75-independent-actual-fullphysical-and-exactheads-gate',compare)
    final=target('V75-actual-whole-mixed-call-restoration',lambda:_graph(engine,call))
    def finish():
        _match(engine,(actual['saved'],),(final['saved'],),'Final actual mixed hidden restoration failed')
        _match(engine,actual['KV'],final['KV'],'Final actual mixed query KV restoration failed')
        output=mixed_target.split_outputs(engine,call,final['returned']);_heads_match(engine,original_output,output)
        after=protect_prefixes(engine,epoch['candidate_span_ids'],epoch['candidate_prefixes'])
        _gate(engine,before!=after,'Mixed qualification changed an already-valid candidate source prefix')
        evidence=dict(passed=True,changed_input_variant=bool(changed),phase_index=phase_index,
            physical_M=M,segments=S,mask_shape=list(MASK),tree_queries=layout.verification_queries,
            prompt_queries=layout.prefill_queries,scratch_queries=layout.padding_queries,
            exact_head_rows=len(layout.head_indices),all197_hidden_original_matcher=True,
            all64_target_query_KV_original_matcher=True,exact_tree_and_final_prompt_heads=True,
            all_candidate_and_reference_prefixes_finite_and_unchanged=True,actual_query_KV_restored=True,
            reference_never_initialized_from_candidate_KV=True,original_native=native,
            isolated_span_reference_executed=epoch['scope'].normal_prefix_tokens is None,
            independent_full_mirror_executed=True,
            independent_mirror_metadata=dict(physical_M=M,segments=S,mask_shape=list(MASK),
                actual_seq_lengths_q=reference['query_partition'],sequence_lens=reference['KV_lengths']),
            candidate_MC2=dict(before=dispatch_before,after=dispatch_after,delta=_delta(dispatch_before,dispatch_after)),
            independent_eager_calls=reference['independent_eager_calls'],
            independent_mirror_dispatch=reference['independent_mirror_dispatch'],
            reference_scope=f'independent history, same sparse M197/S{S}/FULL128 local qualification only',
            source_input_sha256=epoch['source_input_sha256'],dense_AR_qualified=False,
            online_qualified=False,cost_or_Goodput=False)
        return dict(output=output,call=call,hidden_snapshot=final['saved'],query_KV_snapshot=final['KV'],
            reference_query_KV=reference['reference_query_KV'],reference_hidden=reference['reference_hidden'],
            reference_mirror_evidence=reference,evidence=evidence)
    result=target('V75-actual-mixed-final-state-gate',finish)
    if engine.is_draft:
        return dict(output=None,call=None,hidden_snapshot=None,query_KV_snapshot=None,
            reference_query_KV=None,reference_hidden=None,reference_mirror_evidence=None,
            reference_record=reference,evidence=dict(role='draft',target_model_called=False,
                phase_index=phase_index,independent_draft_reference_chunk_executed=True,
                additional_actual_forward_for_qualification=False,online_qualified=False,cost_or_Goodput=False))
    result['reference_record']=reference
    result['evidence']['additional_actual_forward_for_qualification']=True
    return result
