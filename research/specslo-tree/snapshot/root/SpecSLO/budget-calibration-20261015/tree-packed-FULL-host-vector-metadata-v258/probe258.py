"""Actual NPU metadata equality and paired construction cost; no neural calls."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,copy,dataclasses,hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
SOURCE=B/'tree-static-q5-M120-full463-own-AR-v256/check_full256_HOST.py'

def main():
    output=Path(sys.argv[1]);assert not output.exists()
    spec=importlib.util.spec_from_file_location('_packet258_native_CPU_fixture',SOURCE)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    node=copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
    cut=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='forward'for t in n.targets))
    node.name='_packet258_fixture';node.body=node.body[:cut]+[ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))]
    ns=dict(vars(fixture));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE)+'::original-fixture-prefix','exec'),ns)
    env=ns[node.name]();engine=env['engine'];host=env['host'];torch=env['torch'];pad=env['pad']
    assert 'torch_npu'not in sys.modules
    from oracle255 import cases,prepare
    frames=[]
    for name in ['N1-fork5-anchor0','N8-fork5-anchor0','N16-all4-anchor0','N24-all4-anchor0','N8-mixed-anchor0']:
        case=next(c for c in cases()if c['name']==name)
        plans,rows=prepare(case,[[103+i]*127 for i in range(24)])
        n=case['N'];roots=list(range(41,41+n))
        pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
        physical=[p for plan in pp for p in plan.cache_positions.tolist()]
        owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
        engine._prepare_attention_metadata(owners,physical,True)
        frames.append((case,(engine,pp,rr,dd),dict(sequence_ids=ss,allow_causal_fast_path=False)))
    # Restore the exact original device predicate; the CPU fixture stub is never
    # used to claim a fused device path. No arithmetic/model __call__ is invoked.
    predicate=copy.deepcopy(next(n for n in ast.parse(Path(host.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_uses_fused_attention'))
    namespace=dict(vars(host));exec(compile(ast.fix_missing_locations(ast.Module(body=[predicate],type_ignores=[])),str(host.__file__)+'::original-device-predicate','exec'),namespace)
    host._uses_fused_attention=namespace[predicate.name]
    import torch_npu
    torch.npu.set_device(1);torch.npu.config.allow_internal_format=True
    device=torch.device('npu:1');engine.device=device
    engine.model.embed_tokens.weight=engine.model.embed_tokens.weight.to(device)
    engine.cache_block_tables=engine.cache_block_tables.to(device)
    attention=engine.model.layers[0].self_attn
    attention.block_table=engine.cache_block_tables
    assert host._uses_fused_attention(device,attention)and not host._uses_fused_attention(torch.device('cpu'),attention)
    sys.path.insert(0,str(P));from packet258 import prepared
    new,proof=prepared(host);old=host.make_host_tree_attention_metadata
    def equal(a,b):
        for av,bv in zip(a[:2],b[:2],strict=True):
            assert av.device==bv.device==device and av.dtype==bv.dtype and av.stride()==bv.stride()and torch.equal(av.cpu(),bv.cpu())
        for field in dataclasses.fields(a[2]):
            av,bv=getattr(a[2],field.name),getattr(b[2],field.name)
            if isinstance(av,torch.Tensor):
                assert isinstance(bv,torch.Tensor)and av.device==bv.device and av.dtype==bv.dtype and av.stride()==bv.stride()and torch.equal(av.cpu(),bv.cpu()),field.name
            else:assert av==bv,field.name
        assert a[2].block_tables is a[2].request_block_tables and b[2].block_tables is b[2].request_block_tables
    rows=[]
    for case,args,kwargs in frames:
        equal(old(*args,**kwargs),new(*args,**kwargs))
        for _ in range(8):old(*args,**kwargs);new(*args,**kwargs)
        timings={'original':[],'packed':[]};pairs=[]
        for pair in range(80):
            order=('original','packed')if pair%2==0 else('packed','original');measured={}
            for key in order:
                fn=old if key=='original'else new
                torch.npu.synchronize();begin=time.perf_counter()
                result=fn(*args,**kwargs)
                torch.npu.synchronize();elapsed=(time.perf_counter()-begin)*1000
                measured[key]=elapsed;timings[key].append(elapsed)
            pairs.append(dict(order=order,milliseconds=measured))
        equal(old(*args,**kwargs),new(*args,**kwargs))
        rows.append(dict(case=case['name'],budgets=case['budgets'],paired_calls=80,
            all_actual_metadata_values_dtypes_strides_aliases_equal=True,
            median_ms={key:statistics.median(values)for key,values in timings.items()},
            mean_ms={key:statistics.mean(values)for key,values in timings.items()},pairs=pairs))
    negative=[]
    _,args,kwargs=frames[-1]
    for label,owners in [('duplicate',[0]*len(args[1])),('negative',[-1]+kwargs['sequence_ids'][1:]),('unallocated',[300]+kwargs['sequence_ids'][1:])]:
        errors=[]
        for fn in(old,new):
            try:fn(*args,**dict(kwargs,sequence_ids=owners))
            except Exception as error:errors.append((type(error).__name__,str(error)))
            else:raise AssertionError('invalid device metadata accepted '+label)
        assert errors[0]==errors[1];negative.append(dict(case=label,error=errors[0]))
    torch.npu.synchronize()
    payload=dict(schema='ROOT258.actual-NPU-original-HOST-metadata-paired-cost.v1',status='complete',
        physical_device=5,logical_device=1,torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        rows=rows,negative_controls=negative,original_device_predicate_restored=True,proof=proof,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        candidate_source_sha256=hashlib.sha256((P/'packet258.py').read_bytes()).hexdigest(),
        models_loaded=False,model_attention_or_KV_executed=False,
        scope='complete original checked HOST factory plus actual device synchronization; no projection, neural or graph execution',
        Goodput=None,neural_qualified=False,production_installed=False,adopted=False)
    with output.open('x')as stream:json.dump(payload,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='complete',rows=[dict(case=r['case'],median_ms=r['median_ms'])for r in rows])))

if __name__=='__main__':main()
