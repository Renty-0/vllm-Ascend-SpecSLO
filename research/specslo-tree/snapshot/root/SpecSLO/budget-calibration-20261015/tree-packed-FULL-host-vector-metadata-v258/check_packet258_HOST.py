"""Exact full metadata comparison on actual Native CPU protocol fixtures."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,copy,dataclasses,hashlib,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
SOURCE=B/'tree-static-q5-M120-full463-own-AR-v256/check_full256_HOST.py'
spec=importlib.util.spec_from_file_location('_packet258_original_CPU_fixture',SOURCE)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
node=copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
cut=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='forward'for t in n.targets))
node.name='_packet258_CPU_fixture';node.body=node.body[:cut]+[ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))]
ns=dict(vars(fixture));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE)+'::original-CPU-fixture-prefix','exec'),ns)
env=ns[node.name]();engine=env['engine'];host=env['host'];torch=env['torch'];pad=env['pad']
sys.path.insert(0,str(P))
from packet258 import prepared
new,proof=prepared(host);old=host.make_host_tree_attention_metadata
from oracle255 import cases,prepare
prefixes=[[11]*1505,[12]*2555]+[[13+i]*127 for i in range(22)]
def same(a,b):
    for av,bv in zip(a[:2],b[:2],strict=True):assert torch.equal(av,bv)and av.dtype==bv.dtype and av.stride()==bv.stride()
    for field in dataclasses.fields(a[2]):
        av=getattr(a[2],field.name);bv=getattr(b[2],field.name)
        if isinstance(av,torch.Tensor):assert isinstance(bv,torch.Tensor)and torch.equal(av,bv)and av.dtype==bv.dtype and av.stride()==bv.stride(),field.name
        else:assert av==bv,field.name
    assert (a[2].block_tables is a[2].request_block_tables)==(b[2].block_tables is b[2].request_block_tables)
count=0;last=None
for case in cases():
    plans,rows=prepare(case,prefixes);n=case['N'];roots=list(range(41,41+n))
    pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
    physical=[p for plan in pp for p in plan.cache_positions.tolist()];packed_ids=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
    engine._prepare_attention_metadata(packed_ids,physical,True)
    kwargs=dict(sequence_ids=ss,allow_causal_fast_path=False)
    a=old(engine,pp,rr,dd,**kwargs);b=new(engine,pp,rr,dd,**kwargs);same(a,b);count+=1;last=(pp,rr,dd,kwargs)
assert count==227
pp,rr,dd,kwargs=last;negative=[]
for label,changed in [('duplicate-owner',dict(kwargs,sequence_ids=[0]*len(pp))),('negative-owner',dict(kwargs,sequence_ids=[-1]+kwargs['sequence_ids'][1:])),('unallocated-owner',dict(kwargs,sequence_ids=[300]+kwargs['sequence_ids'][1:]))]:
    errors=[]
    for fn in(old,new):
        try:fn(engine,pp,rr,dd,**changed)
        except Exception as error:errors.append((type(error).__name__,str(error)))
        else:raise AssertionError('invalid metadata accepted '+label)
    assert errors[0]==errors[1];negative.append(dict(case=label,error=errors[0]))
assert 'torch_npu'not in sys.modules
value=dict(status='HOST_all227_original_FULL_metadata_values_dtypes_strides_and_aliases_exact',proof=proof,
    cases=count,independent_metadata_fields_checked=True,negative_controls=negative,
    original_H2D_paths_replaced_only_for_validated_FULL=True,last_layout=engine._packet258_last_layout,
    actual_NPU_used=False,model='original CPU protocol fixture with arithmetic double; no neural/attention/graph',
    original_fixture_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),performance_qualified=False)
with(P/'HOST_PACKET258_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
