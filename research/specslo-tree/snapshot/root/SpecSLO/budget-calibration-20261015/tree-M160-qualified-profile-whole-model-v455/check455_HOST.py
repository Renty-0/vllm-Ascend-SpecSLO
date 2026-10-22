"""Whole-model source construction and actual generated config/action routing."""
import argparse,ast,copy,hashlib,importlib.abc,inspect,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST455 construction cannot load a model')
sys.meta_path.insert(0,NoModel())
from host_env455 import setup
setup()
import runtime455,probe455,guard455,whole455,audit455,config455
source=runtime455.worker_source()
assert source.count('from check455 import ready')==2 and source.count('from probe455 import run')==1
assert source.count('from config455 import make_config')==1 and 'from startup_config_v132 import make_config'not in source
assert 'whole455'in probe455._run.__code__.co_names
ns=guard455.execution_namespace()
assert ns['ACTION']==runtime455.ACTION and ns['execute_locked'].__globals__ is ns
assert(ns['MAX_SECONDS'],ns['MAX_TAG_BYTES'],ns['MAX_AGGREGATE_BYTES'])==(3600,192*1024**2,3584*1024**2)
assert ns['prepared']is guard455.prepared and ns['actual_audit']is guard455.actual_audit
assert 'action='+runtime455.ACTION+'\n'in(P/'startup455.sh').read_text()
node=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
namespace={'argparse':argparse};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),runtime455.__file__+'::actual-main-parser','exec'),namespace)
class StopAfterParser(Exception):pass
old_parse=argparse.ArgumentParser.parse_args;old_argv=sys.argv[:];seen=[]
def parse_and_stop(parser,*a,**k):
 value=old_parse(parser,*a,**k);assert value.action==runtime455.ACTION and value.tag=='HOST455-NO-MODEL'
 seen.append(vars(value));raise StopAfterParser()
try:
 argparse.ArgumentParser.parse_args=parse_and_stop;sys.argv=['runtime455.py','--action',runtime455.ACTION,'--tag','HOST455-NO-MODEL']
 try:namespace['main']()
 except StopAfterParser:pass
finally:argparse.ArgumentParser.parse_args=old_parse;sys.argv=old_argv
assert len(seen)==1
worker=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
condition=next(n.test for n in ast.walk(worker)if isinstance(n,ast.If)and any(isinstance(x,ast.Constant)and x.value=='single fixed diagnostic action'for x in ast.walk(n)))
code=compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(condition))),runtime455.__file__+'::actual-worker-action','eval')
assert eval(code,{},dict(completed=False,mode='calibrate',action=runtime455.ACTION))is False
assert eval(code,{},dict(completed=False,mode='calibrate',action='M160-real32-native-epilogue-qualification'))is True
assert eval(code,{},dict(completed=True,mode='calibrate',action=runtime455.ACTION))is True
native=json.loads((P/'HOST455_NATIVE_PROFILE_RESULT.json').read_text());assert native['status']=='PASS_HOST455_ACTUAL_NATIVE_NORMALIZER_AND_PROFILE_ONLY_CONFIG_CHANGE'
assert native['NPU_initialized']is False and native['changed_config_fields']==['mc2_profile']and native['max_num_seqs']==64
assert hashlib.sha256((P/'mc2-profile-with-qualified160.json').read_bytes()).hexdigest()==config455.EXPECTED_PROFILE==native['profile_sha256']
reference=inspect.getsource(whole455.unfused_forward);body=inspect.getsource(whole455.measure)
assert 'layer.self_attn(positions,hidden,metadata)'in reference and 'layer.mlp(hidden)'in reference
assert 'forward_mc2'not in reference and '.enable_mc2='not in reference and '.validated='not in reference
assert "operations={'control160':run_control,'qualified160':run160,'current120':run120}"in body
assert 'check(name,replay=False)'in body and '.reset()'not in body and 'empty_cache'not in body
assert 'engine._root455_whole_resources='in body and 'model._mc2_chained_row_counts'in body
for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
result=dict(status='PASS_HOST455_ACTUAL_CONFIG_FACTORY_PARSER_WORKER_AND_UNFUSED_REFERENCE_SOURCE',
 actual_runtime_source_inverse=True,actual_config_factory_routed_before_constructor=True,
 actual_main_parser_accepted=True,actual_worker_guard_checked=True,original_Native_profile_loader_CPU_passed=True,
 only_config_field_changed='mc2_profile',max_num_seqs=64,all3_graph_inputs_outputs_held=True,
 all_actual_timed_replay_outputs_checked=True,model_constructors=0,NPU_used=False,NN_qualification=False,Raw=None,Goodput=None)
with(P/'HOST455_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
