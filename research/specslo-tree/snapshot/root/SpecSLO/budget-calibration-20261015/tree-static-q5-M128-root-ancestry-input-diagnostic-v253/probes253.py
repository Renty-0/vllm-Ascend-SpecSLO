"""Read actual eager Q/visible KV/attention to localize long branch drift.

No graph validation flags, sampling, model precision or weights are changed.
This is a three-case diagnostic, never a performance qualification.
"""
def run(engine,native,stage,output,fixture):
    import hashlib,json,torch
    from pathlib import Path
    from dataclasses import replace
    from static_state251 import install
    from bank251 import build
    from oracle251 import cases,prepare,candidate,reference
    from baseline251 import measurement
    from reference_schedule251 import schedule
    import metadata251
    target=not engine.is_draft
    activation=stage('ROOT253/actual-static128-model-hooks',lambda:install(engine,native))
    bank=build(engine,native,stage,fixture['prefixes'])
    records=[];traces={};capture={'mode':None}
    def retain(name,value):
        with(Path(output)/f'rank{engine.rank}-{name}.json').open('x')as f:json.dump(value,f,indent=2)
    def fingerprint(value):
        v=value.contiguous().cpu()
        return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest())
    def traced(attention,old,layer):
        def execute(query,metadata):
            attended=old(query,metadata)
            mode=capture['mode']
            if mode and layer<4:
                # Both consumed side nodes are the first query in group1.
                row,group=(0,0)if mode=='candidate-root'else(5,1)
                visible=(~metadata.tree_attention_mask[group,0,0,:2560]).nonzero().flatten()
                table=metadata.request_block_tables[group]
                slots=table.index_select(0,visible.div(128,rounding_mode='floor')).long()*128+visible.remainder(128)
                full=torch.arange(2560,device=engine.device)
                allslots=table.index_select(0,full.div(128,rounding_mode='floor')).long()*128+full.remainder(128)
                values=dict(Q=query[row].clone().cpu(),K=attention.key_cache.flatten(0,1).index_select(0,slots).cpu(),
                    V=attention.value_cache.flatten(0,1).index_select(0,slots).cpu(),A=attended[row].clone().cpu())
                traces.setdefault(mode,{})[layer]=dict(values=values,visible_positions=visible.cpu().tolist(),
                    table=table.cpu().tolist(),mask_shape=list(metadata.tree_attention_mask.shape),
                    masked_tail_K=fingerprint(attention.key_cache.flatten(0,1).index_select(0,allslots)),
                    masked_tail_V=fingerprint(attention.value_cache.flatten(0,1).index_select(0,allslots)))
            return attended
        return execute
    # First create all needed real graphs using the unchanged model callbacks.
    selected=cases()[:3]
    for case in selected:
        plans,rows=prepare(case,bank['prefixes'])
        actual=stage('ROOT253/'+case['name']+'/actual',lambda:candidate(engine,case,bank,plans,rows)if target else None)
        if target:own=reference(engine,case,bank,plans,rows,lambda label,fn:stage('ROOT253/'+case['name']+'/'+label,fn))
        else:
            own=None
            for k,_ in enumerate(schedule(plans,[0]*case['N'],rows)):stage('ROOT253/'+case['name']+'/independent-reference/frame'+str(k),lambda:None)
        result=stage('ROOT253/'+case['name']+'/measure',lambda:measurement(engine,case,actual,own)if target else None)
        records.append(result)
        stage('ROOT253/'+case['name']+'/retain',lambda:retain('case'+str(len(records)),result))
    case=selected[-1];plans,rows=prepare(case,bank['prefixes'])
    # Repeat exact candidate before the eager-only inspection. Installation
    # occurs after graph capture, so no host inspection runs during capture.
    actual=stage('ROOT253/repeat-actual-before-inspection',lambda:candidate(engine,case,bank,plans,rows)if target else None)
    saved=[]
    if target:
        for layer,block in enumerate(engine.model.layers):
            attention=block.self_attn;old=attention._fused_infer_attention
            saved.append((attention,old));attention._fused_infer_attention=traced(attention,old,layer)
    def inspect_actual():
        if not target:return None
        capture['mode']='candidate-cap11'
        try:return engine._branch_tail_state.eager_last().cpu()
        finally:capture['mode']=None
    stage('ROOT253/eager-actual-attention-inputs',inspect_actual)
    def inspect_root():
        if not target:return None
        capture['mode']='candidate-root'
        try:return engine._branch_tail_state.eager_last().cpu()
        finally:capture['mode']=None
    stage('ROOT253/eager-actual-root-attention-inputs',inspect_root)
    original_pack=metadata251.pack_q5
    original_checked=metadata251.checked
    inspected=[]
    try:
        for capacity in (10,11):
            def pack(e,frame,*,full,capacity=capacity):
                x,p,m=original_pack(e,frame,full=full)
                if capacity==11:
                    mask=torch.ones((26,1,11,4096),dtype=torch.bool,device=e.device)
                    mask[:,:,:10].copy_(m.tree_attention_mask);m=replace(m,tree_attention_mask=mask)
                return x,p,m
            metadata251.pack_q5=pack
            def checked(meta,capacity=capacity):
                if tuple(meta.tree_attention_mask.shape)!=(26,1,capacity,4096)or tuple(meta.sequence_lens)!=(2560,)*26:
                    raise RuntimeError('exact declared diagnostic reference capacity')
                return meta
            metadata251.checked=checked
            def labeled(label,fn,capacity=capacity):
                def call():
                    if target:capture['mode']=('reference-cap'+str(capacity)if label=='independent-reference/frame1'else 'reference-root-cap'+str(capacity)if label=='independent-reference/frame0'else None)
                    try:return fn()
                    finally:capture['mode']=None
                return stage('ROOT253/inspect-cap'+str(capacity)+'/'+label,call)
            if target:own=reference(engine,case,bank,plans,rows,labeled)
            else:
                own=None
                for k,_ in enumerate(schedule(plans,[0]*case['N'],rows)):labeled('independent-reference/frame'+str(k),lambda:None)
            inspected.append(stage('ROOT253/measure-inspected-cap'+str(capacity),lambda:measurement(engine,case,actual,own)if target else None))
    finally:
        metadata251.pack_q5=original_pack
        metadata251.checked=original_checked
        for attention,old in saved:attention._fused_infer_attention=old
    def summarize():
        if not target:return None
        result={}
        base=traces['candidate-cap11']
        for mode,layers in traces.items():
            entries=[]
            for layer,item in sorted(layers.items()):
                other=traces['candidate-root'][layer]if 'root'in mode else base[layer]
                differences={}
                for key,v in item['values'].items():
                    a=other['values'][key]
                    differences[key]=dict(bitwise_equal=torch.equal(a,v),max_abs=float((a.float()-v.float()).abs().max()),
                        candidate=fingerprint(a),reference=fingerprint(v),different_positions=(a!=v).reshape(a.shape[0],-1).any(-1).nonzero().flatten().tolist())
                entries.append(dict(layer=layer,comparisons=differences,visible_positions_equal=item['visible_positions']==other['visible_positions'],
                    visible_count=len(item['visible_positions']),mask_shape=item['mask_shape'],table=item['table'],
                    masked_tail_K=item['masked_tail_K'],masked_tail_V=item['masked_tail_V']))
            result[mode]=entries
        return result
    trace=stage('ROOT253/summarize-captured-attention-inputs',summarize)
    stage('ROOT253/retain-captured-attention-summary',lambda:retain('attention-inputs',dict(trace=trace,inspected=inspected)))
    stage('ROOT253/release-bounded-bank',lambda:engine._release_cache()if target else None)
    return dict(rank=engine.rank,activation=activation,cases=records,input_trace=trace,inspected=inspected,
        actual_diagnostic_cases=3,protocol_120_label_is_not_node_count=True,
        whole_workload_AR_qualified=False,performance=False,adopted=False)
