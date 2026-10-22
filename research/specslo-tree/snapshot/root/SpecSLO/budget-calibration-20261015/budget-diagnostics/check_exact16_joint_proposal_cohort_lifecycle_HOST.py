"""Bounded original-controller HOST study; no candidate, model, or device.

Past verdicts and proposal payloads are explicit HOST carriers.  The original
control-plane methods and original candidate shape execute unchanged.  This
is a lifecycle proof check, not new model or production qualification.
"""
import ast,copy,hashlib,importlib.abc,json,sys,time,traceback,types
from dataclasses import fields,make_dataclass,replace
from pathlib import Path
from types import SimpleNamespace as NS

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
BASE=B/'tree-high-normal-cap4-native-families-v148'
P197=B/'tree-empty-active-cohort-causal80-active32-full-v197'
ACTIVE=B/'tree-active-window32-16-local-v160'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def require(value,message):
 if value is not True:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','transformers','triton'):
   raise ImportError('This original-controller HOST forbids backend '+name)

def original_rhythm():
 path=N/'spec_rhythm.py';tree=ast.parse(path.read_text());changed=0
 for index,node in enumerate(tree.body):
  if isinstance(node,ast.ImportFrom)and node.module=='vllm_ascend.spec_decode.pearl.roofline':
   require([n.name for n in node.names]==['ProfiledRoofline'],'one original type-only profile import')
   tree.body[index]=ast.parse('class ProfiledRoofline: pass').body[0];changed+=1
 require(changed==1,'only type-only roofline dependency is replaced by a HOST carrier')
 module=types.ModuleType('_exact16_cohort_actual_original_rhythm');sys.modules[module.__name__]=module
 exec(compile(ast.fix_missing_locations(tree),str(path)+'::no-model-type-carrier','exec'),vars(module))
 return module

def configuration():
 node=next(n for n in ast.walk(ast.parse((BASE/'startup_config_v75.py').read_text()))if isinstance(n,ast.Return))
 require(isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)
  and node.value.func.id=='PEARLConfig','actual config return constructor')
 values=eval(compile(ast.Expression(copy.deepcopy(node.value)),str(BASE/'startup_config_v75.py')+
  '::actual-config-return-expression','eval'),dict(PEARLConfig=lambda **values:values))
 config=next(n for n in ast.parse((N/'api.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='PEARLConfig')
 for name in ('spec_rhythm_draft_token_budget','spec_rhythm_eager_reserve_tokens'):
  default=next(n for n in config.body if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name)and n.target.id==name)
  require(name not in values,'constructor leaves actual original default '+name)
  values[name]=ast.literal_eval(default.value)
 cls=make_dataclass('Explicit_HOST_Config',[(k,object)for k in values]);original=cls(**values)
 unit=ast.parse((BASE/'candidate_controls_v132.py').read_text())
 select=next(n for n in ast.walk(unit)if isinstance(n,ast.FunctionDef)and n.name=='select')
 stmt=next(copy.deepcopy(n)for n in ast.walk(select)if isinstance(n,ast.Assign)
  and any(isinstance(t,ast.Attribute)and t.attr=='candidate_config'for t in n.targets))
 control=NS();env=dict(self=control,engine=NS(config=original),replace=replace)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[stmt],type_ignores=[])),
  str(BASE/'candidate_controls_v132.py')+'::actual-config-delta','exec'),env)
 candidate=control.candidate_config
 changed={f.name:(getattr(original,f.name),getattr(candidate,f.name))for f in fields(original)
  if getattr(original,f.name)!=getattr(candidate,f.name)}
 require(changed==dict(spec_rhythm_draft_token_budget=(None,192),spec_rhythm_max_eager_tokens=(5,3),
  spec_rhythm_eager_reserve_tokens=(0,32)),'exact real three config deltas')
 require(candidate.spec_rhythm_max_target_batch==32 and candidate.spec_rhythm_merge_ready_homes is True
  and candidate.spec_rhythm_verification_budget==96,'actual original planner32/merge/roof96')
 return candidate,changed

def observed_hook(rhythm,policy):
 install=next(n for n in ast.parse((BASE/'tree_entry.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='install')
 observed=next(copy.deepcopy(n)for n in ast.walk(install)if isinstance(n,ast.FunctionDef)and n.name=='observed')
 original=rhythm.SpecRhythmRuntimeState.record_verification
 ns=dict(record=original,policy=policy)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[observed],type_ignores=[])),
  str(BASE/'tree_entry.py')+'::actual-online-accounting-hook','exec'),ns)
 rhythm.SpecRhythmRuntimeState.record_verification=ns['observed'];return original

def trial(rhythm,cfg,ready_count,pattern,*,active_count=32,roof=96,reservation=None):
 from budget_policy import TreeBudgetPolicy
 from split_shape_v132 import shape,filter_eager_parents
 from normal_admission_v140 import reset
 policy=TreeBudgetPolicy(read(BASE/'tree-gain-model.json'));record=observed_hook(rhythm,policy)
 try:
  active=list(range(active_count));states={i:rhythm.SpecRhythmRuntimeState(i,0 if i<16 else 1,
   slo_tpot_ms=50.,slo_class='normal',max_gamma=5,output_token_budget=256,delivered_tokens=1,
   decode_elapsed_ms=200.,decode_start_elapsed_ms=0.)for i in active}
  ctl=rhythm.SpecRhythmPipelineController(states);payloads={};old=[]
  for i in range(16,16+ready_count):
   eager=pattern=='all-promoted-E3'or(pattern=='alternating-E3-B4'and i%2==0)
   budget=3 if eager else int(pattern[-1])if pattern.startswith('uniform-B')else 4
   ticket=ctl.new_ticket(i,gamma=3 if eager else budget,eager=False);ctl.publish([ticket])
   if eager:
    child=ctl.new_ticket(i,gamma=3,eager=True);ctl.publish([child])
    promoted=ctl.finish_verification(i,fully_accepted=True,proposed_tokens=3,accepted_tokens=3,
     delivered_tokens=4,draft_confidence=.7,ema_alpha=.4)
    require(promoted is child and ctl.validate_verification(i)is child,'real controller prior promotion/epoch')
    ticket=child
   old.append(ticket);payloads[ticket.proposal_id]=dict(ticket=ticket,plan=NS(candidate_budget=budget))
  require(not ctl.staged_eager,'complete previous cycle has no staged continuation')
  arguments=dict(active_indices_registered_unique=True,priority=True,projected_wait_ms=1.,priority_burst=2,
   target_slo_first=False,merge_ready_homes=cfg.spec_rhythm_merge_ready_homes,
   max_target_requests=cfg.spec_rhythm_max_target_batch,verification_budget=roof)
  if reservation is not None:
   arguments.update(reserved_target_request_indices=tuple(range(16,16+reservation)),reserved_target_home_batch_id=1)
  before_ready=tuple(i for i in active if i in ctl.ready)
  plan=ctl.build_plan(active,**arguments);normal=tuple(plan.normal_draft_request_indices)
  require(normal==tuple(range(16)),'actual original current plan drafts the opposite sixteen nonready owners')
  eligible=filter_eager_parents(plan.eager_candidate_indices,ctl,payloads,{})
  reset(policy,1)
  shaped=shape(policy,rhythm.SpecRhythmBudgetShaper(min_gamma=1,max_gamma=5,verification_budget=96),
   plan_id=plan.plan_id,normal_request_indices=normal,eager_request_indices=tuple(eligible),states=states,
   projected_wait_ms=1.,context_len=128,draft_token_budget=cfg.spec_rhythm_draft_token_budget,
   batch_size=active_count,eager_token_cap=cfg.spec_rhythm_max_eager_tokens,
   eager_reserve_tokens=cfg.spec_rhythm_eager_reserve_tokens,verification_roof=96)
  require(dict(shaped.normal_budgets)==dict.fromkeys(range(16),4)and not shaped.eager_budgets,
   'actual unchanged cap4/lowB5 allocator returns sixteen normal B4 and defers every new E as originally')
  group=[ctl.new_ticket(i,gamma=b,eager=False)for i,b in shaped.normal_budgets.items()]
  shaped.require_published_tickets(group)
  for ticket in group:payloads[ticket.proposal_id]=dict(ticket=ticket,plan=NS(candidate_budget=4))
  ctl.publish(group)
  # Original runtime publication precedes all current target finish calls.
  for ordinal,i in enumerate(plan.target_request_indices):
   ticket=ctl.validate_verification(i);b=ticket.gamma;accepted=min(b,4)if ordinal%2==0 else 0
   require(ctl.finish_verification(i,fully_accepted=accepted==b,proposed_tokens=b,
    accepted_tokens=accepted,delivered_tokens=accepted+1,draft_confidence=.5,ema_alpha=.4)is None,
    'no staged child exists when final E work is empty')
  ctl.finish_cycle(plan.target_home_batch_id)
  require(not ctl.staged_eager,'current complete target loop drains every staged row')
  before=[(i,id(t),t.gamma,t.required_prefix_epoch)for i,t in ctl.ready.items()]
  pruned=BOUND(NS(),ctl,payloads,{},active,roof)
  require(pruned==dict(pruned_proposals=0,pruned_candidates=0,invalidated_eager=0)
   and before==[(i,id(t),t.gamma,t.required_prefix_epoch)for i,t in ctl.ready.items()],
   'actual original ready-bound method changes no B4 payload under roof96 (or bounded test roof)')
  second=ctl.build_plan(active,active_indices_registered_unique=True,priority=True,projected_wait_ms=1.,
   merge_ready_homes=True,max_target_requests=32,verification_budget=roof)
  chosen=tuple(second.target_request_indices);same=chosen==tuple(range(16))
  if active_count==32 and roof==96 and reservation is None:
   require(tuple(plan.target_request_indices)==before_ready and not plan.deferred_target_request_indices,
    'all old ready fit actual merge32/C96, including prior promoted E')
   require(same and not second.deferred_target_request_indices and set(ctl.ready)==set(range(16)),
    'new group alone is ready; next actual planner selects identical sixteen')
   require(all(ctl.validate_verification(i)is t for i,t in zip(chosen,group,strict=True)),
    'the same proposal objects and prefix epochs survive to target')
  return dict(old_ready=ready_count,old_pattern=pattern,old_C=sum(t.gamma for t in old),
   active=active_count,roof=roof,reservation=reservation,current_target=list(plan.target_request_indices),
   old_deferred=list(plan.deferred_target_request_indices),new_normal=list(shaped.normal_budgets),
   final_E=dict(shaped.eager_budgets),next_target=list(chosen),next_C=sum(second.target_candidate_budgets.values()),
   exact_new_group=same,actual_old_E=sum(t.eager for t in old),staged_empty_after=True,
   group_proposal_ids=[t.proposal_id for t in group],HOST_past_verdicts_are_not_model_evidence=True)
 finally:rhythm.SpecRhythmRuntimeState.record_verification=record

def source_invariants():
 import live_decode_patch,empty197
 from window160 import adapt_source
 source=empty197.patch(adapt_source(live_decode_patch.build_live_runtime(),32));tree=ast.parse(source)
 calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='controller.build_plan']
 require(len(calls)==1,'one actual original planner call')
 keys={k.arg for k in calls[0].keywords}
 require(not keys&{'reserved_target_request_indices','reserved_target_home_batch_id','defer_loose',
  'remaining_service_first','remaining_service_target_first'},'actual caller has no hidden reservation or selective defer')
 fn=empty197.admission(tree)
 require(ast.unparse(fn.body[1])=='if active:\n    return closed','exact197 active guard after receive')
 fn=copy.deepcopy(fn);fn.returns=None
 for arg in (*fn.args.args,*fn.args.kwonlyargs):arg.annotation=None
 events=[];pending=[99];env=dict(active=list(range(16)),pending_admission=pending,
  receive_live_admissions=lambda **kw:events.append(kw)or False)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
  str(P197/'empty197.py')+'::actual-generated-admission-guard','exec'),env)
 require(env['admit_available']()is False and len(events)==1 and pending==[99]
  and env['active']==list(range(16)),'actual active guard preserves ready group and pending arrivals')
 normal=next(n for n in ast.walk(ast.parse((N/'spec_rhythm.py').read_text()))if isinstance(n,ast.Assign)
  and any(isinstance(t,ast.Name)and t.id=='default_normal'for t in n.targets))
 eager=next(n for n in ast.walk(ast.parse((N/'spec_rhythm.py').read_text()))if isinstance(n,ast.Assign)
  and any(isinstance(t,ast.Name)and t.id=='eager'for t in n.targets)
  and 'index for index in target if index not in self.staged_eager'in ast.unparse(n))
 require('index not in self.ready' in ast.unparse(normal)and'index not in self.staged_eager'in ast.unparse(normal),
  'original normal owners exclude ready and staged')
 require('for index in target'in ast.unparse(eager),'every new E source belongs to current selected target')
 require(source.index('controller.publish(published_tickets)')<source.index('controller.finish_cycle(plan.target_home_batch_id)'),
  'source publication before successful original finish-cycle')
 require("controller.finish_verification(index,"in source and 'active = [index for index in active if index not in set(finished)]'in source,
  'every actual selected owner finished and active order only filters completion')
 require('VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD=0'in(B/'budget-diagnostics/v197-root-execution/startup197.sh').read_text(),
  'original parent has no asynchronous prefetch reservation')
 return dict(actual_generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
  original_build_plan_keywords=sorted(keys),normal_and_eager_source_conditions=True,
  original_publication_finish_order=True,active_admission_guard=True,
  actual_generated_admission_guard_executed=True,HTTP_abort_excluded_from_finite_workload=True)

def bound_method():
 tree=ast.parse((N/'native_engine.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine')
 fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_bound_spec_rhythm_tree_ready'))
 unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
 ns={};exec(compile(ast.fix_missing_locations(unit),str(N/'native_engine.py')+'::actual-no-prune-control','exec'),ns)
 return ns[fn.name]

def main():
 sys.meta_path.insert(0,Firewall());sys.path[:0]=[str(P197),str(BASE),str(ACTIVE)]
 start=time.perf_counter();attempt=D/'exact16-cohort-lifecycle-HOST'/str(time.time_ns());attempt.mkdir(parents=True)
 paths=[Path(__file__),N/'spec_rhythm.py',N/'api.py',N/'native_engine.py',BASE/'split_shape_v132.py',BASE/'split_budget_v132.py',
  BASE/'budget_policy.py',BASE/'normal_admission_v140.py',BASE/'tree-gain-model.json',BASE/'tree_entry.py',
  BASE/'candidate_controls_v132.py',BASE/'startup_config_v75.py',BASE/'runtime_tree_decode.py',
  BASE/'live_decode_patch.py',BASE/'atomic_decode_patch.py',BASE/'ast_adapter_v123.py',BASE/'runtime_adapter_v132.py',
  P197/'empty197.py',P197/'source197.py',P197/'trace197.py',ACTIVE/'window160.py',
  B/'budget-diagnostics/v197-root-execution/startup197.sh']
 pins={str(p):sha(p)for p in paths};saved=attempt/'sources';saved.mkdir()
 for i,p in enumerate(paths):(saved/(str(i)+'-'+p.name)).write_bytes(p.read_bytes())
 result=dict(schema='exact16-joint-proposal-cohort-original-controller-HOST.v1',status='failed',
  source_only=True,source_before=pins,Torch_imported=False,NPU_used=False,model_used=False,
  candidate_created=False,old_suites_rerun=False,model_math_qualified=False)
 try:
  global BOUND
  BOUND=bound_method()
  rhythm=original_rhythm();cfg,delta=configuration();invariants=source_invariants();positive=[]
  for k in range(17):
   for pattern in ('uniform-B1','uniform-B2','uniform-B3','uniform-B4','uniform-B5',
    'all-promoted-E3','alternating-E3-B4'):
    positive.append(trial(rhythm,cfg,k,pattern))
  # These are real-controller counterexamples only after violating a proved
  # original197 premise.  They do not assert parent197 can reach that state.
  negative=[trial(rhythm,cfg,20,'uniform-B5',active_count=36),
   trial(rhythm,cfg,16,'uniform-B4',reservation=8)]
  require(all(not x['exact_new_group']for x in negative),'premise-changing cases genuinely merge other ready owners')
  require(not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend imported')
  result.update(status='passed',configuration_delta=delta,source_invariants=invariants,
   actual_original_policy_and_controller_cases=len(positive),all_cases_preserve_exact_group=True,
   cases=positive,premise_violation_counterexamples=negative,
   proof_scope='successful finite197 active<=32, C96/maxTarget32/mergeTrue, all oldB<=5, completed cycles staged-empty, finalE0, no HTTPabort/admission/reservation between publication and nextplan',
   generic_controller_only_is_not_sufficient=True,past_promotion_HOST_carriers_not_model_or_historical_frequency=True)
 except BaseException:result['error']=traceback.format_exc()
 result.update(source_after={str(p):sha(p)for p in paths},elapsed_seconds=time.perf_counter()-start)
 require(result['source_before']==result['source_after'],'source remains exact')
 path=attempt/'result.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=result['status'],elapsed_seconds=result['elapsed_seconds'])))
 if result['status']!='passed':print(result['error']);return 1
 return 0
if __name__=='__main__':raise SystemExit(main())
