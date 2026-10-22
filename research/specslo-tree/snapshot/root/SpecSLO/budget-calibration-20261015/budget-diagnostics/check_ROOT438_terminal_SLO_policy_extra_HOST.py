"""Synthetic controls for the exact external terminal policy projection, no NN."""
import ast,collections,copy,hashlib,importlib.abc,json,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST policy auditor cannot import NN')
sys.meta_path.insert(0,NoModel())
for folder in ('tree-slo-aware-refill-complete-interface-full-v438','tree-nonempty-current-incumbent-full-proposal-v432','tree-nonempty-refill-accounting-witness-host-v429','tree-nonempty-wholeprompt-refill-resource-host-v426'):sys.path.insert(0,str(B/folder))
def need(v,m):
 if not v:raise AssertionError(m)
p=D/'audit_ROOT438_terminal_full_HOST.py';tree=ast.parse(p.read_text())
call=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='original.definitions')
source=(D/'audit_v197_terminal_HOST.py').read_text()
for a,b in ast.literal_eval(call.args[2]):source=source.replace(a,b)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='admission_and_full_statistics')
ns=dict(need=need,collections=collections,common_full_statistics=lambda a,b:{'synthetic':True})
exec(compile(ast.Module(body=[node],type_ignores=[]),str(p)+'::actual-policy-projection','exec'),ns)
config=dict(spec_rhythm_prefill_coalesce_min_requests=1,spec_rhythm_prefill_coalesce_max_wait_ms=0,spec_rhythm_verification_budget=96)
reports=[dict(configuration_before_constructor=config,configuration_after_constructor=config)for _ in range(4)]
def trace(old_count,new_count,ms):
 groups=[list(range(old_count)),list(range(old_count,old_count+new_count))];n=old_count+new_count;w=[]
 for k,g in enumerate(groups):
  old=[]if k==0 else groups[0]
  w.append(dict(incumbents=old,admitted=g,original_arrival_clock=1000.,prefill_ms=30.,total_reserved_pages=64+3*(len(old)+len(g)),incumbent_decode_before_ms=[50.]*len(old),incumbent_decode_after_ms=[80.]*len(old),original_incumbent_tokens_epochs_homes_and_ready_tickets_unchanged=True,new_root_tokens=[100+i for i in g]))
 return dict(sampling=[dict(slo_tpot_ms=ms)for _ in range(n)],scope_count=n,tokenized_inputs=[list(range(63))for _ in range(n)],absolute_arrivals=[999.]*n,standalone_prefill_calls=[dict(sequence_ids=g)for g in groups],stream_events=[dict(request_index=i,token_ids=[100+i],finished=False)for i in range(n)]+[dict(request_index=i,token_ids=[7],finished=True)for i in range(n)],ROOT432_refill_witnesses=w,sealed_before=True,sealed_after=True,cache_released=True,configured_capacity=64,admission_window160=64,active_peak=n,actual_capacity64_scope_completed=False,decode_counters=dict(spec_rhythm_slo_aware_admission_enabled=1,spec_rhythm_target_slo_first_enabled=1,spec_rhythm_prefill_coalesce_enabled=0,spec_rhythm_prefill_coalesce_deferred_polls=0))
rows=[]
for old,new,ms,allow in [(16,32,40.,True),(17,8,40.,False),(32,32,50.,True),(32,32,40.,False)]:
 t=trace(old,new,ms)
 try:r=ns['admission_and_full_statistics']([copy.deepcopy(t)for _ in range(4)],reports)
 except (RuntimeError,AssertionError)as e:
  assert not allow;rows.append(dict(old=old,new=new,slo_ms=ms,rejected=True,error=str(e)))
 else:
  assert allow and r['actual_admission_cohort_replay']['current_SLO_aware_refill_policy']is True;rows.append(dict(old=old,new=new,slo_ms=ms,accepted=True))
r=dict(status='PASS_HOST438_EXACT_TERMINAL_CURRENT_SLO_POLICY_CONTROLS',terminal_auditor_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),synthetic_controls=rows,NPU_used=False,NN_qualification=False,Raw=None,Goodput=None)
with(D/'ROOT438_TERMINAL_SLO_POLICY_EXTRA_HOST_RESULT.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
