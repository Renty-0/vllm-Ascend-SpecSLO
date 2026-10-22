"""Bounded HOST arithmetic replay of authorized terminal205424 raw, not a CLI run."""
import ast,copy,hashlib,importlib.abc,json,runpy,sys,time,traceback
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
SCRIPT=D/'audit_v209_terminal_HOST.py';OLD=D/'audit_v205_terminal_HOST.py';start=time.perf_counter();result=dict(status='failed',future_terminal_CLI_executed=False,new209_actual_read=False,current_model_qualification=False,Torch_imported=False,NPU_used=False)
root=D/'v209-future-auditor-HOST-attempts';attempt=root/str(time.time_ns());attempt.mkdir(parents=True)
for q in(SCRIPT,Path(__file__)):(attempt/q.name).write_bytes(q.read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','triton','vllm','vllm_ascend','transformers','safetensors','acl'):raise ImportError('V209 future HOST forbids '+name)
sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P));sys.path.append(str(B/'tree-causal80-small-cohort-single-writer-full-v205'))
try:
 before={str(p):sha(p)for p in(SCRIPT,OLD,P/'CPU_RESULT209.json',P/'PROJECTIONS209.json')}
 ns=runpy.run_path(str(SCRIPT),run_name='_SOURCE_ONLY_NOT_TERMINAL_MAIN')
 fixtures=json.loads((P/'INPUTS209.json').read_text())['retained_actual205_HOST_fixture'];folder=Path(fixtures['directory']);directory=folder.parent
 cache={}
 def load(p):
  p=Path(p);key=str(p);assert key in fixtures['raw'],'only exact authorized terminal205424fixture'
  if key not in cache:
   assert sha(p)==fixtures['raw'][key];cache[key]=ns['read'](p)
  return cache[key]
 def evaluator(source,mutations=None):
  tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='independent_seven_usage');node=copy.deepcopy(node)
  count=0
  for n in ast.walk(node):
   if isinstance(n,ast.Constant)and isinstance(n.value,str)and n.value=='V209.seven-real-N-current-usage.v1':n.value='V205.seven-real-N-current-usage.v1';count+=1
  if 'mapper209'in ast.unparse(node):assert count==1
  env=dict(ns)
  def reader(p):
   key=str(Path(p));v=load(p)
   if mutations and key in mutations:v=copy.deepcopy(v);mutations[key](v)
   return v
  env['read']=reader
  exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual-independent-seven-usage-ONLY-schema-temporary','exec'),env)
  return env['independent_seven_usage']
 original=evaluator(OLD.read_text())(directory);current=evaluator(SCRIPT.read_text())(directory)
 assert original==current and len(cache)==424
 assert current['counts']['role_current_commits']==336 and current['counts']['new_zero_scalars']==1176 and current['counts']['new_integer_verifier_calls']==504 and current['counts']['new_owner_verdicts']==6048
 def cyc(rank):return str(folder/f'rank{rank}-pass2-N15-cycle3.json.gz')
 def nxt(rank):return str(folder/f'rank{rank}-pass2-N15-next.json.gz')
 cases=[
 ('late-rank0-own-draft-missing',cyc(0),lambda v:v.update(draft=None)),
 ('late-target-missing',cyc(3),lambda v:v.update(target=None)),
 ('late-target-reference-missing',cyc(3),lambda v:v.update(reference=None)),
 ('late-target-allphysical-head-missing',cyc(3),lambda v:v['crossM'].update(new_all80=None)),
 ('late-target-forged-draft',cyc(3),lambda v:v.update(draft={})),
 ('late-draft-variable-count-corrupt',cyc(0),lambda v:v['draft'].update(variable_num_draft_tokens=[4]*15)),
 ('late-draft-next-missing',nxt(0),lambda v:v.update(draft_proof=None)),
 ('late-target-next-fabricated',nxt(3),lambda v:v.update(draft_proof={})),
 ('original-integer-nonzero',cyc(3),lambda v:v['commit'].update(actual_structural_failed=1)),
 ('original-native-zero-nonzero',str(folder/'rank3-component.json.gz'),lambda v:v['traces'][-1]['Native']['delta'].update(aclgraph_captures=1)),
 ('original-target-math-marker-false',cyc(3),lambda v:v['reference'].update(all_physical_greedy_exact=False)),
 ('original-draft-math-marker-false',cyc(0),lambda v:v['draft'].update(exact_full8_encoded=False)),
 ]
 negatives=[]
 for label,path,mutate in cases:
  try:evaluator(SCRIPT.read_text(),{path:mutate})(directory)
  except(ValueError,TypeError,KeyError,RuntimeError)as e:negatives.append(dict(case=label,rejected=True,error_type=type(e).__name__,error=str(e)))
  else:raise AssertionError('must reject '+label)
 # Same source helpers/clock functions are reused, and the full-statistics delta is namespace-only.
 oldtree=ast.parse(OLD.read_text());newtree=ast.parse(SCRIPT.read_text())
 a=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name=='full_statistics');b=next(n for n in newtree.body if isinstance(n,ast.FunctionDef)and n.name=='full_statistics')
 assert ast.dump(a,include_attributes=False).replace('mapper205','mapper209')==ast.dump(b,include_attributes=False)
 assert ns['FINAL_FREEZE_SHA']is ns['FINAL_ARCHIVE_SHA']is ns['FINAL_ROOT_PREPARED_SHA']is None
 main=next(n for n in newtree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
 literal='Future209 terminal CLI remains disabled until exact ROOT final freeze/archive/prepared pins are bound in a reviewed revision'
 gate=next(i for i,n in enumerate(main.body)if literal in ast.unparse(n));receipt=next(i for i,n in enumerate(main.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='receipt_path'for t in n.targets));assert gate<receipt
 assert before=={p:sha(p)for p in before}
 assert not any(n.split('.')[0]in('torch','torch_npu','triton','vllm','vllm_ascend','acl')for n in sys.modules)
 result.update(status='passed',source_before=before,source_after={p:sha(p)for p in before},old205_independent_positive_equals209_role_sensitive_positive=True,
  raw_files424=len(cache),exact_raw_sha256=fixtures['raw'],temporary_adaptation='One schema string V209.seven-real-N-current-usage.v1 -> V205.seven-real-N-current-usage.v1 in AST copy only. No raw rewrite.',
  counts=current['counts'],crossM_behavior_equal=current['crossM_behavior_equal'],numerical_variant_only=current['numerical_variant_only'],negatives=negatives,
  complete_original_Native_integer_verifier_used=True,hidden_or_KV_tensors_recomputed=False,whole_AR=False,formal_Goodput=False,adopted=False,
  component_integer_helper_source_unchanged=True,full_statistics_AST_namespace_inverse=True,CLI_missing_final_ROOT_binding_hard_reject_before_receipt_read=True)
except BaseException:result['error']=traceback.format_exc()
result['elapsed_seconds']=time.perf_counter()-start;result['attempt']=str(attempt);result['source_snapshot']={str(q):sha(q)for q in attempt.glob('*.py')}
out=D/('V209_FUTURE_AUDITOR_ARITHMETIC_HOST_ATTEMPT_'+str(time.time_ns())+'.json');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');(attempt/'RESULT.json').write_bytes(out.read_bytes())
print(json.dumps(dict(path=str(out),sha256=sha(out),status=result['status'],elapsed_seconds=result['elapsed_seconds'],error=result.get('error'))))
raise SystemExit(result['status']!='passed')
