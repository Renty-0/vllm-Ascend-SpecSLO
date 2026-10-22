"""Independent own AR with actual per-position geometry; no candidate inputs."""
import ast,copy
import reference218 as original

def load_geometry(artifact,trace,target_leader_rank):
    from geometry_history460 import build
    leader_path=original.Path(artifact).with_name(f'rank{target_leader_rank}-full463-actual-trace.json.gz')
    raw=leader_path.read_bytes();leader=original.json.loads(original.gzip.decompress(raw))
    original.need(leader['rank']==target_leader_rank,'actual output leader geometry witness')
    for key in('requests','tokenized_inputs','manifest_sha256','prefix_sha256','source_generation','standalone_prefill_calls'):
        original.need(trace[key]==leader[key],'same actual per-rank geometry source inputs '+key)
    result=build(leader,adaptive_required=True)
    result.update(source_artifact=str(leader_path),source_artifact_sha256=original.hashlib.sha256(raw).hexdigest())
    return result

def frame(prompts,outputs,indices,*,physical_M,future=0):
    from reference425 import frame as large
    if physical_M not in(80,120):raise ValueError('ROOT460 bounded recorded physical family')
    data=(original.frame if physical_M==80 else large)(prompts,outputs,indices,future=future)
    data['physical_M']=physical_M
    return data

def target_step(engine,data):
    import hashlib,torch
    from metadata459 import pack_q5
    from scope460 import current
    current(engine)
    original.need(engine._branch_tail_state.enabled is False,'independent reference never copies candidate branches')
    rows=data['physical_M'];groups=rows//5
    engine._prepare_attention_metadata(data['ids'],data['positions'],True)
    inputs,positions,meta=pack_q5(engine,dict(data,selected=data['ids'][::5]),full=True)
    signature=((tuple(meta.tree_attention_mask.shape),tuple(meta.request_block_tables.shape)),)
    key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{signature}',(rows,))
    runner=engine.graph_runner;entry=runner.target_entries.get(key)
    original.need(runner.graph_cache_sealed is True and entry is not None and entry.runtime_validated is True
                  and key not in runner.disabled_entry_keys,'actual naturally validated sealed own-geometry reference graph')
    original.need(meta.tree_attention and meta.use_fused_infer_attention
                  and tuple(meta.actual_seq_lengths_q)==tuple(data['actual_seq_lengths_q'])
                  and tuple(meta.sequence_lens)==(2560,)*groups
                  and tuple(meta.request_block_tables.shape)==(groups,32)
                  and tuple(meta.tree_attention_mask.shape)==(groups,1,10,4096),'complete actual own-prefix q5 metadata')
    output=runner.run_target_greedy([inputs],[positions],[meta],engine.draft_vocab_size)[0]
    values=output.cpu().tolist()
    original.need(len(values)==rows and all(type(v)is int and 0<=v<engine.draft_vocab_size for v in values)
                  and runner.target_entries[key]is entry,'complete own greedy head and same sealed entry')
    mask=meta.tree_attention_mask.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(heads80=values,real_tokens=[values[i]for i in data['real_head_rows']],
                metadata=dict(positions=positions.cpu().tolist(),slots=meta.slot_mapping.cpu().tolist(),
                              contexts=meta.context_lens.cpu().tolist(),block_tables=meta.request_block_tables.cpu().tolist(),
                              actual_seq_lengths_q=list(meta.actual_seq_lengths_q),sequence_lens=list(meta.sequence_lens),
                              backend='independent_static_FULL_q5_actual_M80_or_M120_KV2560',physical_M=rows,
                              tree_mask_shape=list(meta.tree_attention_mask.shape),zero_page=engine._branch_tail_state.static_zero_page,
                              exact_mask_sha256=hashlib.sha256(mask).hexdigest(),
                              future_query_visibility='nonreal q1..q4 attend prefix key0 only'))

def generate_steps(engine,lane,prompts,actual,local_ids,owners,schedules):
    steps=[];canaries=[]
    for begin in range(0,len(local_ids),16):
        chunk=local_ids[begin:begin+16]
        for step in range(255):
            for rows in(80,120):
                selected=[i for i in chunk if schedules[owners[i]][step+1]==rows]
                if not selected:continue
                label=f'AR{begin}-step{step}-M{rows}'
                data=None if engine.is_draft else frame(prompts,actual,selected,physical_M=rows)
                row=lane(label,lambda:None if engine.is_draft else target_step(engine,data))
                if step in(0,127,254):
                    alternate=None if engine.is_draft else frame(prompts,actual,selected,physical_M=rows,future=151643)
                    other=lane('future-canary-'+label,lambda:None if engine.is_draft else target_step(engine,alternate))
                    if not engine.is_draft:
                        original.need(row['real_tokens']==other['real_tokens'],'future dummy inputs cannot change independent first-query outputs')
                        canaries.append(dict(segment_begin=begin,step=step,physical_M=rows,real_owners=selected,
                                             zeros=row['real_tokens'],alternate=other['real_tokens'],equal=True))
                if not engine.is_draft:
                    for i,token in zip(selected,row['real_tokens'],strict=True):actual[i].append(token)
                    steps.append(dict(segment_begin=begin,step=step,physical_M=rows,real_owners=selected,
                                      heads80=row['heads80'],metadata=row['metadata']if step in(0,127,254)else None))
    return steps,canaries

def constructed_run():
    source=original.Path(original.__file__).read_text()
    start=source.index('        for begin in range(0,len(local_ids),16):')
    stop=source.index("        lane('original-all-rank-sticky-finiteness-vote'",start)
    old_loop=source[start:stop]
    new_loop='        steps,canaries=generate_steps(engine,lane,source,actual,local_ids,owners,geometry[\'schedules\'])\n'
    changes=((old_loop,new_loop),
             ("    prompts=trace['tokenized_inputs'];requests=trace['requests']",
              "    prompts=trace['tokenized_inputs'];requests=trace['requests']\n    geometry=load_geometry(artifact,trace,engine.topology.target_leader_rank)"),
             ('all(0<len(group)<=32 for group in groups)','all(0<len(group)<=64 for group in groups)'),
             ('reference_only_prompt_plus_own_outputs=True,candidate_KV_or_tokens_seeded=False,',
              "reference_only_prompt_plus_own_outputs=True,candidate_KV_or_tokens_seeded=False,\n        reference_geometry_source_artifact=geometry['source_artifact'],reference_geometry_source_sha256=geometry['source_artifact_sha256'],"),
             ("backend='original target TP3 BF16 causal FIA Q80/S16; only first query per owner consumed'",
              "backend='independent FULL q5, actual per-token M80/S16 or M120/S24, KV2560'"))
    projected=source
    for old,new in changes:
        if projected.count(old)!=1:raise RuntimeError('ROOT460 exact original independent AR source anchor')
        projected=projected.replace(old,new)
    inverse=projected
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=source:raise RuntimeError('ROOT460 entire independent AR source inverse')
    nodes=[copy.deepcopy(n)for n in ast.parse(projected).body if isinstance(n,ast.FunctionDef)and n.name=='run']
    ns=dict(vars(original),target_step=target_step,frame=frame,generate_steps=generate_steps,load_geometry=load_geometry)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::original-independent-AR-with-actual-geometry','exec'),ns)
    return ns['run']

def run(*args,**kwargs):
    return constructed_run()(*args,**kwargs)
