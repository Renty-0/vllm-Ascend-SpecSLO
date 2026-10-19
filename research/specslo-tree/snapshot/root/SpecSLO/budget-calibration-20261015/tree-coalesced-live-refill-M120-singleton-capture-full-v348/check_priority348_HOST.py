"""Execute the actual original controller and inspect the LIVE policy AST."""
import ast,hashlib,importlib.util,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path.append('/root/SpecSLO/budget-calibration-20261015/tree-static-q5-M120-SLO-first-full463-own-AR-v257')
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
    m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
load('vllm_ascend.spec_decode.pearl.roofline',N/'roofline.py')
module=load('vllm_ascend.spec_decode.pearl.spec_rhythm',N/'spec_rhythm.py')
def execute(enabled):
    states={}
    for i,(kind,slo,elapsed)in enumerate([('tight',40,10)]*2+[('normal',50,1000)]*2+[('loose',150,10000)]*2):
        states[i]=module.SpecRhythmRuntimeState(i,i%2,slo_tpot_ms=slo,slo_class=kind,output_token_budget=256,
            delivered_tokens=1,decode_elapsed_ms=elapsed)
    controller=module.SpecRhythmPipelineController(states)
    controller.publish([controller.new_ticket(i,gamma=4,eager=False)for i in states])
    before={i:(t.proposal_id,t.gamma,t.required_prefix_epoch)for i,t in controller.ready.items()}
    plan=controller.build_plan(list(states),active_indices_registered_unique=True,priority=True,
        target_slo_first=enabled,merge_ready_homes=True,max_target_requests=2,verification_budget=8)
    assert {i:(t.proposal_id,t.gamma,t.required_prefix_epoch)for i,t in controller.ready.items()}==before
    assert sum(plan.target_candidate_budgets.values())==8 and len(plan.target_request_indices)==2
    assert len(plan.deferred_target_request_indices)==4
    return list(plan.target_request_indices)
old=execute(False);new=execute(True)
assert set(old)=={4,5}and set(new)=={0,1}
sys.path.append(str(B/'tree-static-q5-M120-full463-own-AR-v256'))
from generation348 import patch,descriptor
from source348_HOST import source as actual_source
source=actual_source()
tree=ast.parse(patch(source));calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and ast.unparse(n.func)=='_root348_build_plan']
assert len(calls)==1 and ast.literal_eval(next(k.value for k in calls[0].keywords if k.arg=='target_slo_first'))is True
values=[n.values[i]for n in ast.walk(tree)if isinstance(n,ast.Dict)for i,k in enumerate(n.keys)if isinstance(k,ast.Constant)and k.value=='spec_rhythm_target_slo_first_enabled']
assert len(values)==1 and ast.literal_eval(values[0])==1
assert not any(n.split('.')[0]in('torch','torch_npu')for n in sys.modules)
value=dict(status='HOST_actual_controller_SLO_first_and_truthful_live_counter_passed',original_gap_first_selected=old,
    new_SLO_first_selected=new,whole_ready_tickets_preserved=True,verification_budget8_preserved=True,
    real_original_controller_executed=True,live_projection=descriptor(source),NPU_used=False,
    original_controller_source_sha256=hashlib.sha256((N/'spec_rhythm.py').read_bytes()).hexdigest())
with(P/'HOST_PRIORITY348_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
