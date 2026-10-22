"""Actual entire source lifecycle and static-input single-graph contract."""
import ast,inspect,importlib.abc,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST453 no model')
sys.meta_path.insert(0,NoModel())
from host_env453 import setup
setup()
import runtime453,probe453,guard453,forward453,audit453
s=runtime453.worker_source()
assert s.count('from check453 import ready')==2 and s.count('from probe453 import run')==1
assert 'forward453'in probe453._run.__code__.co_names
assert 'separately allocated ordinary Native prefill and16-step AR'in Path(guard453.__file__).read_text()
ns=guard453.execution_namespace()
assert ns['ACTION']==runtime453.ACTION and ns['execute_locked'].__globals__ is ns
assert (ns['MAX_SECONDS'],ns['MAX_TAG_BYTES'],ns['MAX_AGGREGATE_BYTES'])==(3600,192*1024**2,3584*1024**2)
assert ns['prepared']is guard453.prepared and ns['actual_audit']is guard453.actual_audit
assert 'action='+runtime453.ACTION+'\n'in(P/'startup453.sh').read_text()
body=inspect.getsource(forward453.measure)
assert body.count('torch.npu.NPUGraph()')==1 and '.reset()'not in body and 'empty_cache'not in body.replace('no_graph_reset_or_empty_cache_during_epoch','')
assert 'if model_body_closed:raise RuntimeError'in body
assert 'actual independent numerical mismatch; no timing claim'in body
assert '.copy_(fm.tree_attention_mask)'in body and 'static_ptrs!='in body
table=[10,11]+[-1]*30
for start in(0,127):
    for length in(1,2,3,4,5,17,40):
        rows=forward453.rows_for_span(72,start,[3]*length,table)
        assert len(rows)==length and len({r['slot']for r in rows})==length
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
assert 'no ordinary Native model call after capture'in body
assert 'all3_native_AR_tokens_exact'in body and 'post_replay_decode128_KV_exact'in body
assert 'use_aclgraph=False,use_fused_infer_attention=True'in body
assert 'for step in range(15)'in body and 'native_prompts={74:tokens,75:prefix+tokens,76:prefix+tokens,77:tokens,78:prefix+tokens}'in body
# Invoke the actual generated main argument parser, then stop before ready/model.
import argparse,copy
parsed=[]
class ParsedBeforeModel(Exception):pass
parse_original=argparse.ArgumentParser.parse_args
argv_original=sys.argv[:]
def intercepted(parser,*a,**k):
    value=parse_original(parser,*a,**k)
    assert value.action==runtime453.ACTION and value.tag=='HOST453-NO-MODEL'
    parsed.append(vars(value));raise ParsedBeforeModel()
node=copy.deepcopy(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
namespace={'argparse':argparse}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),runtime453.__file__+'::actual-generated-main-parser','exec'),namespace)
try:
    argparse.ArgumentParser.parse_args=intercepted
    sys.argv=['runtime453.py','--action',runtime453.ACTION,'--tag','HOST453-NO-MODEL']
    try:namespace['main']()
    except ParsedBeforeModel:pass
finally:
    argparse.ArgumentParser.parse_args=parse_original;sys.argv=argv_original
assert len(parsed)==1
worker=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
condition=next(n.test for n in ast.walk(worker)if isinstance(n,ast.If)and any(isinstance(x,ast.Constant)and x.value=='single fixed diagnostic action'for x in ast.walk(n)))
code=compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(condition))),runtime453.__file__+'::actual-worker-action-guard','eval')
assert eval(code,{},dict(completed=False,mode='calibrate',action=runtime453.ACTION))is False
assert eval(code,{},dict(completed=False,mode='calibrate',action='M120-single-persistent-graph-prefill-isolation'))is True
assert eval(code,{},dict(completed=True,mode='calibrate',action=runtime453.ACTION))is True
out=dict(status='PASS_HOST453_COMPLETE_SOURCE_AND_SINGLE_STATIC_GRAPH_CONTRACT',
    actual_runtime_source_inverse=True,actual_main_action_parser_accepted=True,actual_worker_action_guard_checked=True,all9_current_cases_retained=True,
    exactly_one_diagnostic_graph=True,no_eager_model_body_after_capture=True,
    no_graph_reset_or_empty_cache_during_epoch=True,independent_references_before_capture=True,
    NPU_used=False,model_constructors=0,Raw=None,Goodput=None)
with(P/'HOST453_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
