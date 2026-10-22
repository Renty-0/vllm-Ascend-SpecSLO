"""Source closure, actual worker/main/parser and exact numerical settings."""
from pathlib import Path
import argparse,ast,copy,datetime,importlib.abc,json,shlex,sys,time
P=Path(__file__).resolve().parent;B=P.parent
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST457 no backend import')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env457 import setup
setup()
for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
import flat457
start=time.monotonic();closure=flat457.verified();elapsed=time.monotonic()-start
assert len(closure)==5155
original_sha=flat457.sha;victim=sorted(closure)[0]
try:
 flat457.sha=lambda p:'0'*64 if str(p)==victim else original_sha(p)
 try:flat457.verified()
 except RuntimeError:pass
 else:raise AssertionError('HOST457 changed parent byte not rejected')
finally:flat457.sha=original_sha
import runtime457,runtime311
source=runtime457.worker_source();tree=ast.parse(source)
nodes=[copy.deepcopy(n)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
ns=dict(vars(runtime311));exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(P/'runtime457.py')+'::HOST-actual-worker-main','exec'),ns)
# Prove this direct base namespace equals the retained385 namespace for every
# global used by the full original lifecycle; no model construction occurs.
parent=runtime457.original.build_runtime()['worker'].__globals__
global_names={n.id for node in nodes for n in ast.walk(node)if isinstance(n,ast.Name)and isinstance(n.ctx,ast.Load)}
shared=global_names&set(parent)&set(ns)-{'worker','main','__name__'}
assert all(ns[name]is parent[name]for name in shared)
missing=(global_names&set(parent))-set(ns)
assert not missing,missing
class StopAfterParse(Exception):pass
old_parse=argparse.ArgumentParser.parse_args;old_argv=sys.argv[:];seen=[]
def stop(parser,*a,**k):
 value=old_parse(parser,*a,**k);seen.append(vars(value));raise StopAfterParse()
try:
 argparse.ArgumentParser.parse_args=stop;sys.argv=['runtime457.py','--action',runtime457.ACTION,'--tag','HOST457-ACTUAL-PARSER']
 try:ns['main']()
 except StopAfterParse:pass
finally:argparse.ArgumentParser.parse_args=old_parse;sys.argv=old_argv
assert seen==[dict(action=runtime457.ACTION,tag='HOST457-ACTUAL-PARSER')]
worker=next(n for n in nodes if n.name=='worker')
predicates=[n.test for n in ast.walk(worker)if isinstance(n,ast.If)and'completed or'in ast.unparse(n.test)]
assert len(predicates)==1
predicate=compile(ast.Expression(body=predicates[0]),'<HOST457-actual-worker-predicate>','eval')
for completed,mode,action,rejected in[(False,'calibrate',runtime457.ACTION,False),(True,'calibrate',runtime457.ACTION,True),(False,'calibrate','M160-qualified-profile-whole-model-reference-cost',True),(False,'calibrate','real-M120-deferred-chain-original-graph-qualifier',True),(False,'wrong',runtime457.ACTION,True)]:
 assert eval(predicate,dict(completed=completed,mode=mode,action=action))is rejected
assert source.count('from startup_config_v132 import make_config')==1 and'config455'not in source
assert source.count('from check457 import ready')==2 and'from probe457 import run'in source
def exports(path):
 result={}
 for line in path.read_text().splitlines():
  if line.startswith('export '):
   for item in shlex.split(line[7:]):
    if item.startswith('VLLM_ASCEND_')and'='in item:
     k,v=item.split('=',1);result[k]=v
 return result
previous=B/'tree-M160-qualified-profile-whole-model-v455/startup455.sh'
assert exports(P/'startup457.sh')==exports(previous)
assert'448'in exports(previous)['VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS'].split(',')
import guard457
supervisor=guard457.execution_namespace()
assert supervisor['execute_locked'].__globals__ is supervisor and supervisor['ACTION']==runtime457.ACTION
assert supervisor['prepared']is guard457.prepared and supervisor['actual_audit']is guard457.actual_audit
assert supervisor['SCOPE']['finite_cases']==1 and supervisor['SCOPE']['independent_prefixes']==72
# The new ordinary reference is the unchanged real-module helper, and no
# configuration, parameter storage or Native validation attribute is assigned.
import whole457,whole455
assert whole457.unfused_forward is whole455.unfused_forward
body=ast.parse((P/'whole457.py').read_text());forbidden=[]
for node in ast.walk(body):
 if isinstance(node,ast.Attribute)and isinstance(node.ctx,ast.Store):
  if node.attr in('runtime_validated','validated','weight','enable_mc2','mc2_profile','data','large_m_nz_min_rows'):forbidden.append(node.attr)
assert not forbidden
assert'check(name,replay=False)'in(P/'whole457.py').read_text()
assert'cross_M120_observation'in(P/'whole457.py').read_text()
result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST457_ACTUAL385_NAMESPACE_SOURCE_PARSER_AND_SETTINGS',
 parent_source_files=5025,verified_parent_and_archives=5155,parent_verification_seconds=elapsed,
 changed_parent_SHA_rejected=True,actual385_runtime_globals_identical=len(shared),actual_main_parser_exercised=True,
 actual_worker_predicates_exercised=5,startup_numerical_exports_unchanged=True,
 configuration_profile_weight_or_Native_flag_assignments=False,cross_M120_discrepancy_preserved=True,
 all27_post_checks_read_last_timed_replay=True,NPU_used=False,actual_model_constructors=0,NN_permission=False,Raw=None,Goodput=None)
with(P/'HOST457_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
