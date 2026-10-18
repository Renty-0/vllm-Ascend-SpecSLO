"""Actual device comparison of complete metadata pipelines, without neural work."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,copy,hashlib,json,statistics,sys,time
from pathlib import Path
from fixture259 import build,equal
P=Path(__file__).resolve().parent

def main():
    output=Path(sys.argv[1]);assert not output.exists()
    env=build();engine=env['engine'];host=env['host'];torch=env['torch'];pad=env['pad']
    from oracle255 import cases,prepare
    from branch_tail255 import State
    from projection255 import transform
    from deferred259 import host_factory,branch_prepare,projected
    frames=[]
    for name in ['N1-fork5-anchor0','N8-fork5-anchor0','N16-all4-anchor0','N24-all4-anchor0','N8-mixed-anchor0']:
        case=next(c for c in cases()if c['name']==name)
        plans,rows=prepare(case,[[103+i]*127 for i in range(24)]);n=case['N']
        pp,rr,dd,ss,real=pad(engine,plans,list(range(41,41+n)),rows,list(range(n)),[],[])
        physical=[p for plan in pp for p in plan.cache_positions.tolist()]
        owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
        engine._prepare_attention_metadata(owners,physical,True)
        frames.append((case,pp,rr,dd,ss))
    assert 'torch_npu'not in sys.modules
    predicate=copy.deepcopy(next(n for n in ast.parse(Path(host.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_uses_fused_attention'))
    ns=dict(vars(host));exec(compile(ast.fix_missing_locations(ast.Module(body=[predicate],type_ignores=[])),str(host.__file__)+'::original-device-predicate','exec'),ns)
    host._uses_fused_attention=ns[predicate.name]
    import torch_npu
    torch.npu.set_device(1);torch.npu.config.allow_internal_format=True
    engine.device=device=torch.device('npu:1')
    engine.model.embed_tokens.weight=engine.model.embed_tokens.weight.to(device)
    engine.cache_block_tables=engine.cache_block_tables.to(device)
    engine.model.layers[0].self_attn.block_table=engine.cache_block_tables
    state=engine._branch_tail_state;state.source=state.source.to(device);state.dest=state.dest.to(device)
    zero=engine.cache_allocation.block_tables[526][0]
    assert host._uses_fused_attention(device,engine.model.layers[0].self_attn)
    new_factory,proof=host_factory(host);old_factory=host.make_host_tree_attention_metadata
    new_prepare=branch_prepare(state);new_transform=projected()
    def pipeline(frame,new):
        case,pp,rr,dd,ss=frame
        raw=(new_factory if new else old_factory)(engine,pp,rr,dd,sequence_ids=ss,allow_causal_fast_path=False)
        intermediate=new_prepare(pp,ss,*raw)if new else State.prepare(state,pp,ss,*raw)
        return(new_transform if new else transform)(*intermediate,case['query_count'],zero,extent=2560,mask_capacity=11 if state.enabled else 10)
    rows=[]
    for frame in frames:
        equal(pipeline(frame,False),pipeline(frame,True))
        for _ in range(8):pipeline(frame,False);pipeline(frame,True)
        timings={'original':[],'deferred':[]};pairs=[]
        for pair in range(80):
            order=('original','deferred')if pair%2==0 else('deferred','original');measured={}
            for key in order:
                torch.npu.synchronize();begin=time.perf_counter();result=pipeline(frame,key=='deferred')
                torch.npu.synchronize();elapsed=(time.perf_counter()-begin)*1000
                measured[key]=elapsed;timings[key].append(elapsed)
            pairs.append(dict(order=order,milliseconds=measured))
        equal(pipeline(frame,False),pipeline(frame,True))
        rows.append(dict(case=frame[0]['name'],budgets=frame[0]['budgets'],paired_calls=80,
            final_device_values_dtypes_strides_aliases_and_mapping_equal=True,
            median_ms={key:statistics.median(values)for key,values in timings.items()},
            mean_ms={key:statistics.mean(values)for key,values in timings.items()},pairs=pairs))
    case,pp,rr,dd,ss=frames[-1];negative=[]
    for label,owners in [('duplicate',[0]*len(pp)),('negative',[-1]+ss[1:]),('unallocated',[300]+ss[1:])]:
        errors=[]
        for fn in(old_factory,new_factory):
            try:fn(engine,pp,rr,dd,sequence_ids=owners,allow_causal_fast_path=False)
            except Exception as error:errors.append((type(error).__name__,str(error)))
            else:raise AssertionError('invalid metadata accepted '+label)
        assert errors[0]==errors[1];negative.append(dict(case=label,error=errors[0]))
    payload=dict(schema='ROOT259.actual-NPU-whole-metadata-pipeline-paired-cost.v1',status='complete',
        physical_device=5,logical_device=1,rows=rows,negative_controls=negative,proof=proof,
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        candidate_source_sha256=hashlib.sha256((P/'deferred259.py').read_bytes()).hexdigest(),
        models_loaded=False,model_attention_or_KV_executed=False,Goodput=None,
        scope='HOST validation+factory+branch descriptors+canonical projection+device sync; no attention, neural or graphs',
        neural_qualified=False,production_installed=False,adopted=False)
    with output.open('x')as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='complete',rows=[dict(case=r['case'],median_ms=r['median_ms'])for r in rows])))
if __name__=='__main__':main()
