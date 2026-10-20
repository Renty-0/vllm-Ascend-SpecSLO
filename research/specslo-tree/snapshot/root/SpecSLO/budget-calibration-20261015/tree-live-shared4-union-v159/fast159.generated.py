"""Optional spine-first branching draft captured as one dependent graph.

Up to eight sequential top-2 decisions produce a spine with terminal siblings.
The selected candidate budget bounds how many dependent steps are explored. This changes the draft implementation only; target verification,
ancestor masks and selected-branch KV commits remain authoritative.
"""
from collections import Counter
import time

SCALE=1_000_000


def reserve_query_pages(engine,plans,ids):
    sequence_ids=[]
    positions=[]
    for plan,seq in zip(plans,ids):
        for step in range(int(getattr(plan,'depth',4))):
            sequence_ids.append(seq)
            positions.append(int(plan.prefix_len)+step)
    # Branch repair includes the current root. It may be the first token
    # on a new page, before the ordinary PA metadata builder is reached.
    engine._ensure_cache_capacity(sequence_ids,positions)


def missing_committed_suffix(state,prefix,tokens):
    """The current root is evaluated by the next normal chain, not repair."""
    start=prefix-len(tokens)+1
    for offset,token in enumerate(tokens[:-1]):
        pos=start+offset
        if pos>=state['safe_before'] and state['tokens'].get(pos)!=int(token):
            return pos,list(map(int,tokens[offset:-1]))
    return prefix,[]


def execute_top2(runner, inputs, positions, metadata, vocabulary_size,
                 *, final_kv_only=False, return_confidence=False):
    import torch
    outputs=[]
    for step,(pos,attn) in enumerate(zip(positions,metadata)):
        hidden=runner.model(inputs,pos,attn)
        if final_kv_only and step==len(positions)-1:
            continue
        logits=runner.model.compute_logits(hidden)[:,:vocabulary_size]
        probabilities=torch.softmax(logits.float(),dim=-1)
        confidence,tokens=probabilities.topk(2,dim=-1)
        inputs=tokens[:,0]
        encoded=tokens.to(torch.long)*(SCALE+1)+torch.round(
            confidence.clamp(0,1)*SCALE).to(torch.long)
        outputs.append(encoded)
    return torch.stack(outputs,dim=1)


def install(native):
    import torch
    runner_cls=native.NativeACLGraphRunner
    original_execute=runner_cls._execute_draft
    def execute(runner,*args,**kwargs):
        if getattr(runner,'_dynamic_tree_top2',False) and kwargs.get('return_confidence'):
            return execute_top2(runner,*args,**kwargs)
        return original_execute(runner,*args,**kwargs)
    runner_cls._execute_draft=execute
    cls=native.NativePearlEngine
    original=cls.draft_tree_forward
    allocate=cls._allocate_cache
    def fresh_cache(engine,*args,**kwargs):
        result=allocate(engine,*args,**kwargs)
        engine._dynamic_tree_known_KV={}
        engine._dynamic_tree_fast_profile_calls=0
        return result
    cls._allocate_cache=fresh_cache

    def draft(engine,plans,roots,ids,eager_parent_sources=None,**kwargs):
        if not engine.is_draft:
            return None
        depth=int(plans[0].depth) if plans else 0
        if not plans or not all(p.width==2 and p.depth==depth for p in plans):
            if eager_parent_sources:
                raise RuntimeError('Canonical eager received an unsupported tree topology')
            return original(engine,plans,roots,ids,eager_parent_sources=eager_parent_sources,**kwargs)
        if any(float(v)>0 for v in (kwargs.get('draft_temperatures') or ())):
            raise RuntimeError('Canonical eager draft is qualified only for greedy sampling')
        if len(ids)!=len(set(ids)):
            raise RuntimeError('Canonical eager draft requests must be unique')
        sources=eager_parent_sources or {}
        if not set(sources).issubset(ids):
            raise RuntimeError('Canonical eager source is outside this draft batch')
        steps=min(depth,max(int(p.candidate_budget) for p in plans))
        reserve_query_pages(engine,plans,ids)
        known=getattr(engine,'_dynamic_tree_known_KV',None)
        if known is None:
            known=engine._dynamic_tree_known_KV={}
        counts=getattr(engine,'_dynamic_tree_draft_counts',None)
        if counts is None:
            counts=engine._dynamic_tree_draft_counts=Counter()
        profiled=getattr(engine,'_dynamic_tree_fast_profile_calls',0)
        profiling=profiled<getattr(engine.config,'profile_host_decode_steps',0)
        ticks=[time.perf_counter_ns()] if profiling else None
        catchups=kwargs.get('committed_catchup_token_ids') or [()]*len(plans)
        repairs=[]
        frontiers=[]
        effective_roots=list(map(int,roots))
        eager_frontier_tokens={}
        for plan,root,seq,tokens in zip(plans,roots,ids,catchups):
            prefix=int(plan.prefix_len)
            if seq in sources:
                from vllm_ascend.spec_decode.pearl.tree import tree_primary_path
                parent,row=sources[seq]
                path=tree_primary_path(parent)
                state=known.get(seq)
                if not path or state is None or tokens:
                    raise RuntimeError('Canonical eager needs a tracked parent and no committed catchup')
                parent_prefix=int(parent.prefix_len)
                if prefix!=parent_prefix+len(path)+1:
                    raise RuntimeError('Canonical eager prefix does not follow its parent spine')
                # Target packed slots may contain gaps from unrelated sibling
                # nodes. Draft PA owns an independent canonical logical KV
                # prefix, checked below from the actual known-token ledger.
                positions=parent.positions.detach().cpu().tolist()
                if [int(positions[j+1]) for j in path]!=list(range(parent_prefix+1,prefix)):
                    raise RuntimeError('Canonical eager parent spine positions are not contiguous')
                if state['safe_before']>parent_prefix or state['tokens'].get(parent_prefix)!=int(root):
                    raise RuntimeError('Canonical eager parent root or committed boundary is stale')
                for offset,node in enumerate(path[:-1],1):
                    if state['tokens'].get(parent_prefix+offset)!=int(row[node]):
                        raise RuntimeError('Canonical eager parent spine KV is not materialized')
                frontiers.append((seq,prefix-1,int(row[path[-1]])))
                continue
            if seq not in known:
                # The native service prefills every committed prompt token
                # before publishing this first root.
                known[seq]={'safe_before':prefix,'tokens':{}}
            state=known[seq]
            pos,repair=missing_committed_suffix(state,prefix,tokens)
            if repair:
                repairs.append((seq,pos,repair,int(root)))
            state['safe_before']=prefix
        # Same normal4/parent3/child3 branch; known-token checks above and
        # all original ledger/publication code below remain authoritative.
        from live159 import fused_prelude
        fused_plan,predicted=fused_prelude(engine,native,runner_cls,
            plans,ids,sources,repairs,frontiers,steps)
        repair_calls=0
        repair_total=repair_padding=0
        if fused_plan is None and repairs and all(len(tokens)==1 for _,_,tokens,_ in repairs):
            repair_ids=[seq for seq,_,_,_ in repairs]
            repair_positions=[pos for _,pos,_,_ in repairs]
            inp=torch.tensor([tokens[0] for _,_,tokens,_ in repairs],dtype=torch.long,device=engine.device)
            pos,meta=engine._prepare_attention_metadata(repair_ids,repair_positions,
                use_fused_infer_attention=False)
            repair_bucket=native._next_linear_draft_graph_bucket(len(repairs),engine.config.max_num_seqs)
            inp,pos,meta=runner_cls._pad_inputs(inp,pos,meta,repair_bucket)
            engine.graph_runner.run_tree_hidden(inp,pos,meta)
            repair_total=int(inp.numel())
            repair_padding=repair_total-len(repairs)
            repair_calls=1
            counts['repair_real_queries']+=len(repairs)
            counts['repair_padded_queries']+=repair_total
            counts['repair_calls']+=1
            counts['repair_PA_calls']+=1
        elif fused_plan is None and repairs:
            from pa_repair import run as repair_pa
            repair_total,repair_padding,repair_calls=repair_pa(engine,repairs,native)
            counts['repair_real_queries']+=sum(len(tokens) for _,_,tokens,_ in repairs)
            counts['repair_padded_queries']+=repair_total
            counts['repair_calls']+=repair_calls
            counts['repair_PA_calls']+=repair_calls
            counts['repair_teacher_forced_chains']+=1
        frontier_total=frontier_padding=0
        if frontiers:
            if fused_plan is None:
                fids=[seq for seq,_,_ in frontiers]
                fpositions=[pos for _,pos,_ in frontiers]
                ftokens=[token for _,_,token in frontiers]
                engine._ensure_cache_capacity(fids,fpositions)
                inp=torch.tensor(ftokens,dtype=torch.long,device=engine.device)
                pos,meta=engine._prepare_attention_metadata(fids,fpositions,False)
                fbucket=native._next_linear_draft_graph_bucket(len(frontiers),engine.config.max_num_seqs)
                inp,pos,meta=runner_cls._pad_inputs(inp,pos,meta,fbucket)
                predicted=engine.graph_runner.run_target_greedy([inp],[pos],[meta],
                    engine.draft_vocab_size)[0][:len(frontiers)].cpu().tolist()
            for (seq,position,token),frontier in zip(frontiers,predicted):
                effective_roots[ids.index(seq)]=int(frontier)
                eager_frontier_tokens[seq]=int(frontier)
                from kv_ledger import record_overwrite
                record_overwrite(known[seq],position,[token])
            if fused_plan is None:
                frontier_total=fbucket
                frontier_padding=fbucket-len(frontiers)
                counts['eager_frontier_calls']+=1
                counts['eager_frontier_real_queries']+=len(frontiers)
                counts['eager_frontier_padded_queries']+=fbucket
            else:
                # Old per-group PA call/padded histograms now describe only
                # unfused calls. Keep logical real work, report one union call.
                counts['repair_real_queries']+=fused_plan.R
                counts['eager_frontier_real_queries']+=fused_plan.E
                counts['repair_frontier_fused_PA_calls']+=1
                counts['repair_frontier_fused_physical_queries']+=fused_plan.union_M
                counts['repair_frontier_fused_head_rows']+=fused_plan.frontier_M
                counts['repair_frontier_fused_repair_work_rows']+=fused_plan.R
                counts['repair_frontier_fused_frontier_work_rows']+=fused_plan.E
        prelude_queries=(fused_plan.union_M if fused_plan is not None else repair_total+frontier_total)
        prelude_padding=(fused_plan.union_M-fused_plan.E-fused_plan.R if fused_plan is not None else repair_padding+frontier_padding)
        prelude_calls=(1 if fused_plan is not None else repair_calls+int(bool(frontiers)))
        if profiling:ticks.append(time.perf_counter_ns())
        bucket=native._next_linear_draft_graph_bucket(len(ids),engine.config.max_num_seqs)
        from chain_metadata import build as batch_chain_metadata
        padded_inputs,positions,metadatas=batch_chain_metadata(engine,plans,ids,effective_roots,steps,bucket)
        if profiling:ticks.append(time.perf_counter_ns())
        engine.graph_runner._dynamic_tree_top2=True
        encoded=engine.graph_runner.run_draft_greedy(padded_inputs,positions,metadatas,
            engine.draft_vocab_size,valid_row_count=len(ids),final_kv_only=False,
            return_confidence=True)[:len(ids)].clone()
        values=encoded.cpu().tolist()
        if profiling:ticks.append(time.perf_counter_ns())
        rows=[];scores=[]
        for seq,root,plan,row in zip(ids,effective_roots,plans,values):
            tokens=[[int(v)//(SCALE+1) for v in pair] for pair in row]
            confidence=[[int(v)%(SCALE+1)/SCALE for v in pair] for pair in row]
            rows.append([t[0] for t in tokens]+[0]*(depth-steps)+[t[1] for t in tokens]+[0]*(depth-steps))
            scores.append([v[0] for v in confidence]+[0.]*(depth-steps)+[v[1] for v in confidence]+[0.]*(depth-steps))
            # The last predicted leaf is materialized only if accepted, as
            # one PA query before the next chain consumes its committed root.
            written=[int(root)]+[t[0] for t in tokens[:-1]]
            written_tokens={int(plan.prefix_len)+j:t for j,t in enumerate(written)}
            if seq in sources:
                # Keep the parent's still-uncommitted root/spine for rejection
                # repair. Ahead-of-turn execution never advances safe_before.
                from kv_ledger import record_overwrite
                record_overwrite(known[seq],int(plan.prefix_len),written)
            else:
                known[seq]['tokens']=written_tokens
        counts['chain_calls']+=1
        counts['chain_real_queries']+=steps*len(ids)
        counts['chain_padded_queries']+=steps*bucket
        counts['generated_candidates']+=2*steps*len(ids)
        if profiling:
            ticks.append(time.perf_counter_ns())
            for j,name in enumerate(('repair_submit','main_metadata','chain_and_D2H','result_pack')):
                counts['profile_'+name+'_ns']+=ticks[j+1]-ticks[j]
            counts['profiled_calls']+=1
            engine._dynamic_tree_fast_profile_calls=profiled+1
        return {'draft_token_ids':rows,'root_token_ids':effective_roots,
            'eager_frontier_tokens':eager_frontier_tokens,'parent_indices':torch.cat([p.parent_indices for p in plans]),
            'num_draft_tokens':[p.candidate_budget for p in plans],
            'draft_confidence':torch.tensor([sum(v)/(2*steps) for v in scores]),
            'node_confidences':[torch.tensor(v) for v in scores],
            'cache_slot_mapping':None,'query_count':steps*bucket+prelude_queries,
            'graph_padding_query_count':steps*(bucket-len(ids))+prelude_padding,
            'model_calls':1+prelude_calls,'graph_calls':1+prelude_calls,
            'profile_seconds':{},'materialization_deferred':True,
            'linear_full_chain':False,'canonical_deferred_KV':True}
    cls.draft_tree_forward=draft
