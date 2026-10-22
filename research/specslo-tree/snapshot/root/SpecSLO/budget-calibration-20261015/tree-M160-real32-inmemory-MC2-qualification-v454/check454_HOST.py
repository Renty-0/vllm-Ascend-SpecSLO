"""Actual generated parser/worker protocol and HOST-only collective routing."""
import argparse,ast,copy,hashlib,importlib.abc,inspect,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST454 no model imports')
sys.meta_path.insert(0,NoModel())
from host_env454 import setup
setup()
import runtime454,probe454,guard454,qualify454,audit454
source=runtime454.worker_source()
assert source.count('from check454 import ready')==2 and source.count('from probe454 import run')==1
assert 'qualify454'in probe454._run.__code__.co_names
ns=guard454.execution_namespace()
assert ns['ACTION']==runtime454.ACTION and ns['execute_locked'].__globals__ is ns
assert(ns['MAX_SECONDS'],ns['MAX_TAG_BYTES'],ns['MAX_AGGREGATE_BYTES'])==(3600,192*1024**2,3584*1024**2)
assert ns['prepared']is guard454.prepared and ns['actual_audit']is guard454.actual_audit
assert 'action='+runtime454.ACTION+'\n'in(P/'startup454.sh').read_text()
assert 'genuine32 separately prefilled owners'in Path(guard454.__file__).read_text()
# Exercise actual generated main parsing and stop before ready/model imports.
node=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
namespace={'argparse':argparse};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),runtime454.__file__+'::actual-main-parser','exec'),namespace)
class StopAfterParser(Exception):pass
old_parse=argparse.ArgumentParser.parse_args;old_argv=sys.argv[:];seen=[]
def parse_and_stop(parser,*a,**k):
 value=old_parse(parser,*a,**k);assert value.action==runtime454.ACTION and value.tag=='HOST454-NO-MODEL'
 seen.append(vars(value));raise StopAfterParser()
try:
 argparse.ArgumentParser.parse_args=parse_and_stop;sys.argv=['runtime454.py','--action',runtime454.ACTION,'--tag','HOST454-NO-MODEL']
 try:namespace['main']()
 except StopAfterParser:pass
finally:argparse.ArgumentParser.parse_args=old_parse;sys.argv=old_argv
assert len(seen)==1
worker=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
condition=next(n.test for n in ast.walk(worker)if isinstance(n,ast.If)and any(isinstance(x,ast.Constant)and x.value=='single fixed diagnostic action'for x in ast.walk(n)))
code=compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(condition))),runtime454.__file__+'::actual-worker-action','eval')
assert eval(code,{},dict(completed=False,mode='calibrate',action=runtime454.ACTION))is False
assert eval(code,{},dict(completed=False,mode='calibrate',action='M120-native-prefill-independent-AR-isolation'))is True
assert eval(code,{},dict(completed=True,mode='calibrate',action=runtime454.ACTION))is True
# Only check argument translation here; no fake model or numerical outputs.
group=object();calls=[]
class ArgumentRecorder:
 def get_global_rank(self,g,rank):assert g is group;return 1+rank
 def broadcast_object_list(self,obj,**kwargs):calls.append(('objects',kwargs));return obj
 def broadcast(self,obj,**kwargs):calls.append(('tensor',kwargs));return obj
proxy=qualify454.BorrowedTargetGroup(ArgumentRecorder(),group)
proxy.broadcast_object_list(['HOST'],src=0);proxy.broadcast('HOST',src=2)
assert calls==[('objects',dict(src=1,group=group)),('tensor',dict(src=3,group=group))]
try:proxy.broadcast_object_list([],group=object())
except RuntimeError:pass
else:raise AssertionError('non-target collective allowed')
loader=qualify454.LiveTensorLoader(object());loader.expected_path=Path('/HOST/rank-0.pt')
loader.payload={key:object()for key in('activation','weight','residual','gamma')}
actual=loader.load(loader.expected_path,map_location='cpu',weights_only=True)
assert all(actual[k]is loader.payload[k]for k in actual)and loader.calls==1
try:loader.load('/WRONG',map_location='cpu',weights_only=True)
except RuntimeError:pass
else:raise AssertionError('wrong in-memory input request accepted')
body=inspect.getsource(qualify454.measure)
assert 'recorder.capture(dict(N=32,budgets=[4]*32))'in body
assert 'changed_input_replays=1024'in body and 'samples=31'in body and 'minimum_speedup=1.02'in body
assert 'module.run_measurement_job(args,runtime)'in body and 'torch.save'not in body
for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
settings=json.loads((P/'HOST454_SETTINGS_RESULT.json').read_text());assert settings['NPU_used']is False
result=dict(status='PASS_HOST454_ACTUAL_PARSER_WORKER_GUARD_FRAME_AND_GROUP_ROUTING',
 actual_runtime_source_inverse=True,actual_main_parser_accepted=True,actual_worker_action_checked=True,
 target_only_broadcast_src_translation_checked=True,live_loader_identity_and_wrong_path_checked=True,
 input_source='genuine32 separately prefilled model owners required at NPU time',
 unchanged_original_qualifier_required=True,model_constructors=0,NPU_used=False,NN_qualification=False,Raw=None,Goodput=None)
with(P/'HOST454_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
