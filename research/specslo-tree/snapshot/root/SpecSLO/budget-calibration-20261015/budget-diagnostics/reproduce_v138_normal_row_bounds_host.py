"""Actual immutable controller/allocator HOST counterexample; no device code."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import types

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-split192-config-snapshot-fix-v138'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')
ATTEMPTS=[]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
            ATTEMPTS.append(fullname);raise ImportError('No device/backend import in HOST reproduction')
        return None
sys.meta_path.insert(0,Guard())
source=N.read_text();tree=ast.parse(source)
removed=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='vllm_ascend.spec_decode.pearl.roofline']
assert len(removed)==1
tree.body=[n for n in tree.body if n not in removed]
module=types.ModuleType('_V138_immutable_actual_controller_HOST');sys.modules[module.__name__]=module
module.__dict__['ProfiledRoofline']=type('UnusedProfiledRoofline',(),{})
exec(compile(tree,str(N)+':HOST-only-no-roofline-import','exec'),module.__dict__)
def local(name):
    path=P/(name+'.py');spec=importlib.util.spec_from_file_location('_V138_HOST_'+name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[m.__name__]=m;spec.loader.exec_module(m);return m
budget=local('split_budget_v132');policy_module=local('budget_policy')
states={i:module.SpecRhythmRuntimeState(request_index=i,home_batch_id=i%2,slo_tpot_ms=40.,slo_class='tight',max_gamma=5,output_token_budget=256) for i in range(65)}
for state in states.values():state.decode_elapsed_ms=300.;state.delivered_tokens=1
active=tuple(i for i in range(65) if i!=1)  # Finish old odd owner1, then admit even owner64.
controller=module.SpecRhythmPipelineController(states,next_target_home_batch_id=1)
for i in active:
    if i%2:controller.publish([controller.new_ticket(i,gamma=3,eager=False)])
plan=controller.build_plan(active,active_indices_registered_unique=True,priority=True,projected_wait_ms=80.,
    merge_ready_homes=True,max_target_requests=32,verification_budget=96)
normal,eager=plan.normal_draft_request_indices,plan.eager_candidate_indices
assert len(active)==64 and len(normal)==33 and len(eager)==31 and not set(normal)&set(eager)
policy=policy_module.TreeBudgetPolicy(json.loads((P/'tree-gain-model.json').read_text()))
shaper=module.SpecRhythmBudgetShaper(min_gamma=1,max_gamma=8,verification_budget=96)
arguments=dict(plan_id=plan.plan_id,states=states,projected_wait_ms=80.,context_len=128,
    draft_token_budget=192,batch_size=64,eager_token_cap=3,eager_reserve_tokens=32,
    verification_roof=96,work_limit=budget.DraftWorkLimit(normal_per_request=3))
failure=None
try:budget.shape_draft_work(policy,shaper,normal_request_indices=normal,eager_request_indices=eager,admitted_eager_indices=eager,**arguments)
except ValueError as exc:failure=str(exc)
assert failure=='Draft row bounds exceeded before allocation'
# A fresh legal32+32 owner set entirely above64 passes the SAME real allocator.
high_normal=tuple(range(100,164,2));high_eager=tuple(range(101,164,2))
high_states={i:module.SpecRhythmRuntimeState(request_index=i,home_batch_id=i%2,slo_tpot_ms=40.,slo_class='tight',max_gamma=5,output_token_budget=256) for i in high_normal+high_eager}
for state in high_states.values():state.decode_elapsed_ms=300.;state.delivered_tokens=1
arguments['states']=high_states
control=budget.shape_draft_work(policy,shaper,normal_request_indices=high_normal,eager_request_indices=high_eager,admitted_eager_indices=high_eager,**arguments)
assert len(control.normal_budgets)==len(control.eager_budgets)==32 and control.allocated_selected_candidates==192
assert not ATTEMPTS and not any(name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for name in sys.modules)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
result=dict(schema='HOST.V138.actual-controller-row-bound-counterexample.v1',status='passed',
    actual_HOST_argv=sys.orig_argv,script_sha256=sha(__file__),
    source_sha256={str(path):sha(path) for path in (N,P/'split_budget_v132.py',P/'split_shape_v132.py',P/'budget_policy.py',P/'runtime_tree_decode.py')},
    actual_controller_AST_executed=True,actual_allocator_executed=True,
    removed_import=ast.unparse(removed[0]),roofline_class_stub='unused: fixed verification_budget96, controller/allocator/RuntimeState methods unchanged',
    legal_upstream_reproduction=dict(active=list(active),active_count=len(active),home_counts=[sum(states[i].home_batch_id==h for i in active) for h in (0,1)],
        normal=list(normal),eager=list(eager),normal_count=len(normal),eager_count=len(eager),combined_count=len(normal)+len(eager),
        target_count=len(plan.target_request_indices),target_candidates=sum(plan.target_candidate_budgets.values()),
        actual_error=failure,padding_added=False,work_plans_or_KV_executed=False),
    high_logical_owner_control=dict(normal=list(high_normal),eager=list(high_eager),normal_count=32,eager_count=32,combined_count=64,selected_candidates=control.allocated_selected_candidates,passed=True),
    evidence_boundary='These are HOST source counterexamples, not the unrecorded exact owner sets/counts at the actual failing full cycle. Actual runtime failure proves only the same exception; source constraints imply normal>32 when original target/active/disjointness gates hold.',
    fix_options=dict(preferred='Cap actual admitted normal rows32 in a new source epoch, explicitly defer whole excess requests without modifying active/state/home/lease; preserve combined actual work64 and target32/C96/Q128.',
      upstream='Truncate only an explicitly selected original normal order before derived immutable plan fields. Retain raw/admitted/deferred owners and counts. Current default order is active order, not a named normal_priority function; prefix slicing may repeatedly defer the tail.',
      allocator='Alternatively allow raw normal candidate set up to active64 but enforce positive normal-grant rows32 in charge/DTO; zero owners remain absent from runtime work/ticket/publish and explicit deferred list. This needs more source/data-contract changes than upstream admission.',
      prohibited='Do not merely enlarge normal_rows to64, change home IDs, mask physical overcapacity or silently drop owners. Do not claim next-round service/fairness without an explicit tested ordering definition.'),
    forbidden_import_attempts=ATTEMPTS,NPU_or_model_or_card_query_executed=False,frozen_sources_modified=False)
destination=B/'budget-diagnostics/V138_NORMAL_ROW_BOUND_HOST_COUNTEREXAMPLE.json'
with destination.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
print(json.dumps(dict(status='passed',output=str(destination),sha256=sha(destination),normal=33,eager=31,total=64,high_logical_owner_control='passed')))
