"""One-off read-only HOST arithmetic; no production policy implementation.

The hypothetical dataclass projection below is evaluated only on an already
returned P176 legal E3 plan. It is not installed in any candidate/runtime.
"""
import ast,copy,hashlib,importlib.abc,json,sys,time
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-high-normal-cap3-active32-full-v176';BASE=B/'tree-high-normal-cap4-native-families-v148'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
RHYTHM=NATIVE.with_name('spec_rhythm.py')
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('HOST forbidden '+fullname)
sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P))
import host176,shape176,normal_admission_v140
from budget_policy import TreeBudgetPolicy
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
pins={str(p):sha(p) for p in (Path(__file__),P/'host176.py',P/'shape176.py',BASE/'budget_policy.py',BASE/'split_budget_v132.py',BASE/'normal_admission_v140.py',BASE/'tree-gain-model.json',BASE/'runtime_tree_decode.py',BASE/'ast_adapter_v123.py',BASE/'candidate_controls_v132.py',BASE/'fast_tree_draft_v5.py',RHYTHM,NATIVE)}
checks=host176.Checks();rhythm=checks.rhythm()
original_ast=ast.parse((BASE/'runtime_tree_decode.py').read_text())
rank_node=next(n for n in ast.walk(original_ast) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='eager' for t in n.targets) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='sorted')
rank_code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(rank_node)],type_ignores=[])),'actual-original-eager-sort','exec')
bucket_nodes=[copy.deepcopy(n) for n in ast.parse(NATIVE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_linear_draft_graph_buckets','_next_linear_draft_graph_bucket')]
assert len(bucket_nodes)==2
bucket_ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=bucket_nodes,type_ignores=[])),str(NATIVE)+'::original-buckets','exec'),bucket_ns)
bucket=lambda count:0 if not count else bucket_ns['_next_linear_draft_graph_bucket'](count,64)
def ranked(states,eager):
 ns=dict(states=states,eager=list(eager),projected_wait_ms=1.);exec(rank_code,ns);return ns['eager']
def legal_plan(N,E):
 states=checks.states(rhythm,N+E) if N+E else {}
 for i,state in states.items():
  # Deterministic heterogeneous HOST inputs only, not target outcomes.
  state.decode_elapsed_ms=150. if i%2 else 200.
  state.acceptance_ema=.9 if i%2 else .4;state.draft_confidence_ema=1.
  state.full_acceptance_ema=.95 if i%3 else .05
 eager=ranked(states,range(N,N+E))
 policy=TreeBudgetPolicy(read(BASE/'tree-gain-model.json'));normal_admission_v140.reset(policy,1)
 shaper=rhythm.SpecRhythmBudgetShaper(min_gamma=1,max_gamma=8,verification_budget=96)
 plan=shape176.shape(policy,shaper,plan_id=1,normal_request_indices=tuple(range(N)),eager_request_indices=tuple(eager),states=states,projected_wait_ms=1.,context_len=128,draft_token_budget=192,batch_size=N+E,eager_token_cap=3,eager_reserve_tokens=32,verification_roof=96)
 return states,plan,policy
def project_for_arithmetic(plan,K):
 # This is a diagnostic evaluation of dataclass conservation, not an adapter.
 chosen=tuple(plan.eager_budgets)[:K]
 eager={i:plan.eager_budgets[i] for i in chosen};reserved={i:v for i,v in plan.reserved_by_owner.items() if i in eager}
 out=replace(plan,eager_budgets=eager,eager_priorities={i:plan.eager_priorities[i] for i in eager},reserved_by_owner=reserved,reserve_used=sum(reserved.values()))
 assert dict(out.normal_budgets)==dict(plan.normal_budgets) and out.available_selected_candidates==plan.available_selected_candidates==192
 assert dict(out.raw_eager_budgets)==dict(plan.raw_eager_budgets) and dict(out.raw_reserved_by_owner)==dict(plan.raw_reserved_by_owner)
 assert plan.reserve_used-out.reserve_used==sum(v for i,v in plan.reserved_by_owner.items() if i not in eager)
 assert out.unused_draft_candidates-plan.unused_draft_candidates==3*(len(plan.eager_budgets)-len(eager))
 ids=[*out.normal_budgets,*out.eager_budgets];budgets=[*out.normal_budgets.values(),*out.eager_budgets.values()];roles=[False]*len(out.normal_budgets)+[True]*len(eager)
 out.require_work(ids,budgets,roles,[NS(candidate_budget=b) for b in budgets])
 return out
start=time.monotonic();summary=[];domains=0;bucket_counts={4:{},8:{}};active_e_cases=0
for N in range(33):
 for E in range(33-N):
  if not N+E:continue
  states,plan,policy=legal_plan(N,E);domains+=1
  if plan.eager_budgets:active_e_cases+=1
  for K in (4,8):
   after=project_for_arithmetic(plan,K)
   before_chain=bucket(len(plan.normal_budgets)+len(plan.eager_budgets));after_chain=bucket(len(after.normal_budgets)+len(after.eager_budgets))
   assert after_chain<=before_chain
   pair=str((before_chain,after_chain));bucket_counts[K][pair]=bucket_counts[K].get(pair,0)+1
   if (N,E) in ((0,32),(9,23),(12,20),(16,16),(20,12),(24,8),(25,7),(28,4),(8,24)):
    summary.append(dict(N=N,admitted_E=E,K=K,legal_E=len(plan.eager_budgets),kept_E=len(after.eager_budgets),normal_candidates=sum(after.normal_budgets.values()),before_E_candidates=sum(plan.eager_budgets.values()),after_E_candidates=sum(after.eager_budgets.values()),available=after.available_selected_candidates,before_unused=plan.unused_draft_candidates,after_unused=after.unused_draft_candidates,reserve_requested=after.reserve_requested,before_reserve_used=plan.reserve_used,after_reserve_used=after.reserve_used,raw_reserved_unchanged=dict(after.raw_reserved_by_owner)==dict(plan.raw_reserved_by_owner),before_chain_M=before_chain,after_chain_M=after_chain,before_frontier_F=bucket(len(plan.eager_budgets)),after_frontier_F=bucket(len(after.eager_budgets)),original_eager_order=list(plan.eager_budgets),selected_owners=list(after.eager_budgets)))
# Actual ticket API: only retained rows are ever materialized/published.
states,plan,policy=legal_plan(16,16);tickets=[];controller=rhythm.SpecRhythmPipelineController(states)
parents=[controller.new_ticket(i,gamma=3,eager=False)for i in range(16,32)];controller.publish(parents);parent_identity=dict(controller.ready)
filtered=project_for_arithmetic(plan,4)
for i,b in filtered.normal_budgets.items():tickets.append(controller.new_ticket(i,gamma=b,eager=False))
for i,b in filtered.eager_budgets.items():tickets.append(controller.new_ticket(i,gamma=b,eager=True))
filtered.require_published_tickets(tickets);controller.publish(tickets)
assert set(controller.staged_eager)==set(filtered.eager_budgets)
assert all(controller.ready[i] is parent_identity[i]for i in range(16,32))
# Negative conservation check: dropping reserved owners without fixing reserve
# is rejected by the actual dataclass __post_init__.
try:replace(plan,eager_budgets=dict(filtered.eager_budgets),eager_priorities=dict(filtered.eager_priorities))
except ValueError:bad_reserve_rejected=True
else:raise AssertionError('bad reserve was accepted')
# Ranking carries acceptance*confidence, but not full-parent survival EMA.
first_order=ranked(states,range(16,32))
for state in states.values():state.full_acceptance_ema=1.-state.full_acceptance_ema
assert ranked(states,range(16,32))==first_order
assert any(plan.eager_priorities[first_order[j]]<plan.eager_priorities[first_order[j+1]]for j in range(len(first_order)-1))
assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules)
assert all(sha(p)==digest for p,digest in pins.items())
result=dict(schema='read-only-final-legal-E3-topK-HOST.v1',status='passed',pins=pins,script_sha256=sha(__file__),source_changed=False,candidate_implemented=False,Torch_imported=False,NPU_used=False,model_constructed=False,device_latency_measured=False,Goodput_predicted=False,actual_original_shape_domains=domains,nonempty_legal_eager_domains=active_e_cases,actual_original_Native_bucket_transitions=bucket_counts,representative_arithmetic=summary,actual_original_work_and_ticket_conservation=True,actual_original_parent_tickets_unchanged=True,bad_reserve_projection_rejected=bad_reserve_rejected,raw_reserve_retained_separately=True,parent_survival_EMA_does_not_change_current_order=True,eager_priority_gap_is_not_runtime_sort_score=True,elapsed_seconds=time.monotonic()-start)
out=D/'EAGER_TOP4_TOP8_AFTER_LEGAL_E3_ACTUAL_HOST_ARITHMETIC.json'
with out.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),script_sha256=sha(__file__),domains=domains,elapsed=result['elapsed_seconds'])))
