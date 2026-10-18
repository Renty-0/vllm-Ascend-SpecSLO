"""Actual first-layer model values; identical-query FIA geometry sweep.

This duplicates only operator inputs for diagnostic comparison. It does not
change any workload, sampler or production model, and measures no Goodput.
"""
def run(engine,native,stage,output,fixture):
    import hashlib,json,torch
    from pathlib import Path
    from static_state251 import install
    from bank251 import build
    from oracle251 import cases,prepare,candidate
    from vllm_ascend.spec_decode.pearl.native_graph import run_native_fused_infer_attention
    target=not engine.is_draft
    activation=stage('ROOT254/static128-model-hooks',lambda:install(engine,native))
    bank=build(engine,native,stage,fixture['prefixes'])
    case=cases()[2];plans,rows=prepare(case,bank['prefixes'])
    stage('ROOT254/unchanged-real-model-case',lambda:candidate(engine,case,bank,plans,rows)if target else None)
    saved={}
    def inspect():
        if not target:return None
        attention=engine.model.layers[0].self_attn;old=attention._fused_infer_attention
        def observe(query,metadata):
            result=old(query,metadata)
            saved.update(query=query[0].clone(),attention=attention,table=metadata.request_block_tables[0].clone(),
                         visible_count=len(fixture['prefixes'][0])+1,original_A=result[0].clone())
            return result
        attention._fused_infer_attention=observe
        try:engine._branch_tail_state.eager_last().cpu()
        finally:attention._fused_infer_attention=old
    stage('ROOT254/retain-actual-first-layer-root-inputs',inspect)
    tests=[]
    def sweep():
        if not target:return None
        attention=saved['attention'];prefix=saved['visible_count']
        reference=None;results=[]
        for m in (80,100,120,125,128,160,192):
            sizes=[5]*(m//5)+([m%5]if m%5 else[])
            for pattern in ('identical_all_queries','q0_visible_future_key0'):
                q=saved['query'].unsqueeze(0).expand(m,-1,-1).contiguous()
                table=saved['table'].unsqueeze(0).expand(len(sizes),-1).contiguous()
                mask=torch.ones((len(sizes),1,10,4096),dtype=torch.bool,device=engine.device)
                ends=[];cursor=0;checked=[]
                for i,size in enumerate(sizes):
                    if pattern=='identical_all_queries':mask[i,0,:size,:prefix]=False;checked.extend(range(cursor,cursor+size))
                    else:mask[i,0,0,:prefix]=False;mask[i,0,1:size,0]=False;checked.append(cursor)
                    cursor+=size;ends.append(cursor)
                out=torch.empty_like(q)
                run_native_fused_infer_attention(query=q,key_cache=attention.key_cache.view(544,128,-1),
                    value_cache=attention.value_cache.view(544,128,-1),num_kv_heads=attention.num_kv_heads,
                    num_heads=attention.num_heads,scale=attention.scale,block_table=table,attention_mask=mask,
                    actual_seq_lengths_q=ends,actual_seq_lengths_kv=[2560]*len(sizes),block_size=128,output=out,tree_attention=True)
                cpu=out.cpu();index=torch.tensor(checked,dtype=torch.long,device='cpu');real=cpu.index_select(0,index)
                equal=(real==real[0]).reshape(len(checked),-1).all(-1)
                maximum=(real.float()-real[0].float()).abs().reshape(len(checked),-1).amax(-1)
                if reference is None:reference=real[0].clone()
                results.append(dict(M=m,query_sizes=sizes,pattern=pattern,checked_rows=checked,
                    all_identical_queries_bitwise_equal=bool(equal.all()),different_physical_rows=[checked[j]for j in(~equal).nonzero().flatten().tolist()],
                    maximum_abs_difference_by_checked_row=maximum.tolist(),
                    first_row_equals_original_model_A=torch.equal(real[0],saved['original_A'].cpu()),
                    first_row_equals_first_M80_reference=torch.equal(real[0],reference),
                    output_sha256=hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
        return results
    tests=stage('ROOT254/actual-FIA-identical-query-geometry-sweep',sweep)
    def retain():
        with(Path(output)/f'rank{engine.rank}-operator-geometry.json').open('x')as f:
            json.dump(dict(rank=engine.rank,tests=tests,counterfactual_operator_inputs=True,performance=False),f,indent=2)
    stage('ROOT254/retain-operator-sweep',retain)
    stage('ROOT254/release-bank',lambda:engine._release_cache()if target else None)
    return dict(rank=engine.rank,activation=activation,actual_model_case=case,operator_tests=tests,
        protocol_120_label_is_not_node_count=True,actual_model_cases=1,actual_model_nodes=6,
        counterfactual_operator_inputs=True,whole_workload_AR_qualified=False,performance=False,adopted=False)
