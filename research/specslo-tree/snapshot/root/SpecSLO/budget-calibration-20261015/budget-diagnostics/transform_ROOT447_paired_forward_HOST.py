"""Project446 diagnostics to bounded graph pairs, retaining every NN case."""
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-M120-initial-and-bulk-prefill-math-diagnostic-v446'
P=B/'tree-M120-paired-memory-prefill-math-diagnostic-v447'
t=(OLD/'forward446.py').read_text().replace('446','447')
edits=[]
def change(old,new):
    global t
    if t.count(old)!=1:raise RuntimeError('ROOT447 exact446 anchor '+old)
    t=t.replace(old,new);edits.append((old,new))
change('import torch\n    import torch.distributed as dist','import torch,gc\n    import torch.distributed as dist')
change('from forward445 import measure as original','from parent447 import measure as original')
change('    tokens=list(engine._root445_fixture_prefixes[16][:40])', '''    memory_records=[]
    def release_unused(stage):
        torch.npu.synchronize()
        before=dict(allocated=torch.npu.memory_allocated(),reserved=torch.npu.memory_reserved())
        gc.collect();torch.npu.empty_cache();torch.npu.synchronize()
        free,total=torch.npu.mem_get_info()
        local=dict(rank=engine.rank,stage=stage,before=before,
            allocated=torch.npu.memory_allocated(),reserved=torch.npu.memory_reserved(),
            driver_free=free,driver_total=total)
        rows=[None]*3;dist.all_gather_object(rows,local,group=group)
        memory_records.append(rows)
        with(Path(output)/f'rank{engine.rank}-paired-memory447.json').open('w')as stream:
            json.dump(memory_records,stream,indent=2);stream.write('\\n')
        if any(v['driver_free']<3*1024**3 for v in rows):
            raise RuntimeError('ROOT447 all3 target roles require3GiB free before another capture; no queued model work')
    release_unused('after-fresh445-release-unreferenced-graph-pools')
    tokens=list(engine._root445_fixture_prefixes[16][:40])''')
change("    graphs={'baseline':g0};frames={'baseline':base};outputs={'baseline':h0};cases=[]",'''    graphs={'baseline':g0};frames={'baseline':base};outputs={'baseline':h0};cases=[]
    times={'baseline':[]}
    for _ in range(9):
        dist.barrier(group=group);begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(10):g0.replay()
        end.record();end.synchronize();times['baseline'].append(float(begin.elapsed_time(end))/10)''')
change('            g,h=capture(f);graphs[name]=g;frames[name]=f;outputs[name]=h',
    "            release_unused('before-'+name)\n            g,h=capture(f);graphs[name]=g;frames[name]=f;outputs[name]=h")
change('            cases.append(case)', '''            if set(graphs)!=set(frames)or list(graphs)!=['baseline',name]or len(graphs)!=2:
                raise RuntimeError('ROOT447 exactly the current two owned graph/input frames')
            case['actual_graph_pair']=['baseline',name]
            with(Path(output)/f'rank{engine.rank}-bulk-prefill447-{name}-before-timing.json').open('x')as stream:
                json.dump(dict(rank=engine.rank,case=case,actual_graph_pair=list(frames)),stream,indent=2);stream.write('\\n')
            pair={'baseline':[],name:[]}
            for repeat in range(9):
                for variant in(('baseline',name)if repeat%2==0 else(name,'baseline')):
                    dist.barrier(group=group);begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    begin.record()
                    for _ in range(10):graphs[variant].replay()
                    end.record();end.synchronize();value=float(begin.elapsed_time(end))/10
                    if not math.isfinite(value)or value<=0:raise RuntimeError('ROOT447 actual positive paired device event')
                    pair[variant].append(value)
            times[name]=pair[name];case['paired_baseline_event_ms']=pair['baseline']
            case['post_pair_prefill_hidden_stable']=bool(torch.equal(h[80:80+length].detach().cpu(),actual))
            case['post_pair_prefill128_KV_exact']=all(torch.equal(a,b)for a,b in zip(kv(torch.tensor([r['slot']for r in f[3]],dtype=torch.long,device=engine.device)),wanted_KV,strict=True))
            cases.append(case)
            dist.barrier(group=group);torch.npu.synchronize()
            g.reset()
            del graphs[name],frames[name],outputs[name],g,h,f
            release_unused('after-release-'+name)''')
start=t.index('    names=list(graphs)\n')
stop=t.index('    if params!=',start)
old=t[start:stop]
change(old,'''    names=['baseline',*[c['name']for c in cases]]
    if list(graphs)!=['baseline']or list(frames)!=['baseline']:
        raise RuntimeError('ROOT447 all finished variant graph frames released')
    with(Path(output)/f'rank{engine.rank}-bulk-prefill447-completed-numerical-records.json').open('x')as stream:
        json.dump(dict(rank=engine.rank,cases=cases,tested_graph_variants=names),stream,indent=2);stream.write('\\n')
''')
change("'decode_hidden_exact','decode_tokens_exact','decode_all128_query_KV_exact')",
    "'decode_hidden_exact','decode_tokens_exact','decode_all128_query_KV_exact',\n        'post_pair_prefill_hidden_stable','post_pair_prefill128_KV_exact')")
change('whole_target_event_ms=times,actual_owned_graph_frames=names,',
    'whole_target_event_ms=times,tested_graph_variants=names,maximum_bulk_diagnostic_graphs=2,memory_checkpoints=memory_records,')
change('        max_rank_costs=costs,physical_M=120,physical_S=24,',
    '        max_rank_costs=costs,maximum_bulk_diagnostic_graphs=2,physical_M=120,physical_S=24,')
change("    report=dict(schema='ROOT447.actual-initial-and-bulk-q5-prefill.v1'", """    paired_costs={}
    for index,case in enumerate(cases):
        name=case['name']
        baseline_samples=[max(v['cases'][index]['paired_baseline_event_ms'][j]for v in gathered)for j in range(9)]
        variant_samples=costs[name]['max_rank_samples_ms']
        paired_costs[name]=dict(baseline_samples_ms=baseline_samples,variant_samples_ms=variant_samples,
            baseline_p50_ms=statistics.median(baseline_samples),variant_p50_ms=statistics.median(variant_samples))
    report=dict(schema='ROOT447.actual-initial-and-bulk-q5-prefill.v1'""")
change('max_rank_costs=costs,maximum_bulk_diagnostic_graphs=2','max_rank_costs=costs,paired_max_rank_costs=paired_costs,maximum_bulk_diagnostic_graphs=2')
inverse=t
for old,new in reversed(edits):inverse=inverse.replace(new,old)
if inverse!=(OLD/'forward446.py').read_text().replace('446','447'):
    raise RuntimeError('ROOT447 complete446 numerical source inverse')
(P/'forward447.py').write_text(t)
print(P/'forward447.py')
