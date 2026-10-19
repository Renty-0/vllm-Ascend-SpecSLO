"""Diagnostic eager intermediates only; no relaxed model matchers or timing."""
from contextlib import contextmanager
import json
from pathlib import Path


def install(engine,output):
    if engine.is_draft:
        return
    import torch
    import oracle317
    from reference_schedule317 import schedule

    active={}
    captured={'candidate':{},'reference':{}}
    handles=[]

    @contextmanager
    def recording(phase,physical,labels):
        if active:
            raise RuntimeError('ROOT321 nested recording')
        active.update(phase=phase,index=torch.tensor(physical,dtype=torch.long,device=engine.device),labels=labels)
        try:
            yield
        finally:
            active.clear()

    def collect(name,value):
        if not active:
            return
        values=(value,)if isinstance(value,torch.Tensor)else value if isinstance(value,(tuple,list))else ()
        for index,tensor in enumerate(values):
            if not isinstance(tensor,torch.Tensor)or tensor.ndim!=2 or tensor.shape[0]!=165:
                continue
            key=name+'.'+str(index)
            rows=tensor.detach().index_select(0,active['index'])
            entries=captured[active['phase']].setdefault(key,{})
            for j,label in enumerate(active['labels']):
                if label in entries:
                    raise RuntimeError('ROOT321 duplicate intermediate '+key+str(label))
                entries[label]=rows[j]

    for i,layer in enumerate(engine.model.layers[:4]):
        for name,module in [('qkv',layer.self_attn.qkv_proj),('attention',layer.self_attn),('gate_up',layer.mlp.gate_up_proj),('layer',layer)]:
            key=f'layer{i}.{name}'
            def before(module,args,*,key=key):
                collect(key+'.input',args)
            def after(module,args,result,*,key=key):
                collect(key+'.output',result)
            handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])

    candidate=oracle317.candidate
    reference=oracle317.reference
    alias_proof=None

    def actual(e,case,bank,plans,rows):
        nonlocal alias_proof
        state=e._branch_tail_state
        if case['name']!='N1-fork5-anchor0'or state.prepare.__name__=='original_layout':
            return candidate(e,case,bank,plans,rows)
        eager=state.eager_last
        def observed():
            mapping=state.static_current['physical'][4]['original_to_physical']
            with recording('candidate',list(mapping[:case['query_count']]),list(range(case['query_count']))):
                return eager()
        state.eager_last=observed
        try:
            result=candidate(e,case,bank,plans,rows)
        finally:
            state.eager_last=eager
        # Direct device proof of the prefix/alias bytes after current writes.
        from dual_bank_v146 import query_KV
        desc=state.copy313_layout_HOST
        count=state.last_record['real_copy_slots']
        src=torch.tensor(list(desc['source'][:count]),dtype=torch.long,device=e.device)
        dst=torch.tensor(list(desc['dest'][:count]),dtype=torch.long,device=e.device)
        left,right=query_KV(e,src),query_KV(e,dst)
        equal=[torch.equal(a.view(torch.int16),b.view(torch.int16))for a,b in zip(left,right,strict=True)]
        alias_proof=dict(compared_slots=count,all128_tensors_raw16_equal=all(equal),different_tensor_indices=[i for i,v in enumerate(equal)if not v])
        return result

    def own(e,case,bank,plans,rows,stage):
        if case['name']!='N1-fork5-anchor0':
            return reference(e,case,bank,plans,rows,stage)
        frames=schedule(plans,bank['roots'][:case['N']],rows)
        ordinal=0
        def staged(label,fn):
            nonlocal ordinal
            if not label.startswith('independent-reference/frame'):
                return stage(label,fn)
            frame=frames[ordinal];ordinal+=1
            physical=[];labels=[]
            for row in frame['real_queries']:
                logical=row['candidate_query']
                if logical is not None:
                    if logical[0]!=0:
                        raise RuntimeError('ROOT321 explicit one-owner diagnostic')
                    physical.append(row['head_row']);labels.append(logical[1])
                elif frame['depth']==0:
                    physical.append(row['head_row']);labels.append('shadow-root')
            def observed():
                with recording('reference',physical,labels):
                    return fn()
            return stage(label,observed)
        try:
            result=reference(e,case,bank,plans,rows,staged)
            torch.npu.synchronize()
            compared=[]
            for key,left in captured['candidate'].items():
                right=captured['reference'][key]
                rows=[]
                for label,a in left.items():
                    b=right[label]
                    delta=(a.float()-b.float()).abs()
                    rows.append(dict(query=label,raw16_equal=torch.equal(a.view(torch.int16),b.view(torch.int16)),different_elements=int((a.view(torch.int16)!=b.view(torch.int16)).sum().item()),max_abs=float(delta.max().item())))
                shadow=None
                if 'shadow-root'in right:
                    a,b=left[0],right['shadow-root']
                    shadow=dict(candidate_root_vs_independent_side_root_raw16_equal=torch.equal(a.view(torch.int16),b.view(torch.int16)),max_abs=float((a.float()-b.float()).abs().max().item()))
                compared.append(dict(module=key,rows=rows,independent_shadow_root=shadow))
            layer=e.model.layers[0]
            info=dict(rank=e.rank,case=case,NZ_threshold_actual=layer.mlp.gate_up_proj.large_m_nz_min_rows,
                NZ_alternate_present=layer.mlp.gate_up_proj.large_m_nz_weight is not None,
                row_count165_uses_NZ=(layer.mlp.gate_up_proj.large_m_nz_weight is not None and 165>=layer.mlp.gate_up_proj.large_m_nz_min_rows),
                chained165_admitted=165 in e.model._mc2_chained_row_counts,
                actual_attention_MC2_qualified165=layer.mc2_attention_route.decisions[165].qualified,
                actual_down_MC2_qualified165=layer.mc2_down_route.decisions[165].qualified,
                prefix_alias_raw16=alias_proof,intermediates=compared,model_weights_or_KV_copied_from_candidate_to_reference=False,
                diagnostic_only=True,performance=False,Goodput=None)
            with(Path(output)/f'rank{e.rank}-first-divergence321.json').open('x')as f:
                json.dump(info,f,indent=2)
            return result
        finally:
            for handle in handles:
                handle.remove()
            oracle317.candidate=candidate
            oracle317.reference=reference

    oracle317.candidate=actual
    oracle317.reference=own
