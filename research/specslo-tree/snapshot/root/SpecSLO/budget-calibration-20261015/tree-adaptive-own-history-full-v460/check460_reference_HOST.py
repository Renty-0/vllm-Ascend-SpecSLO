"""HOST-only partition/causality controls for the independent AR loop."""
from pathlib import Path
from types import SimpleNamespace
import ast,datetime,importlib.abc,json,sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST460 reference controls no backend')
sys.meta_path.insert(0,NoModel())
from host_env460 import setup
setup()
import reference460
for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
reference460.constructed_run()
owners=[0,2,4,6,8,10]
prompts=[[11+i]*(17+3*i)for i in range(6)]
schedules=tuple(tuple([0]+[80 if(global_owner+position)%3==0 else 120 for position in range(1,256)])for global_owner in range(12))
initial=[101+i for i in range(6)];actual=[[token]for token in initial]
expected=[]
for i,start in enumerate(initial):
    values=[start]
    for position in range(1,256):
        m=schedules[owners[i]][position]
        values.append((values[-1]*73+(len(prompts[i])+position-1)*17+m+i)%151936)
    expected.append(values)
labels=[]
def lane(label,callback):
    labels.append(label);return callback()
def host_target_step(engine,data):
    rows=data['physical_M'];indices=data['real_owners'];heads=[0]*rows
    assert rows in(80,120)and len(data['tokens'])==len(data['ids'])==len(data['positions'])==rows
    result=[]
    for j,i in enumerate(indices):
        assert data['tokens'][5*j]==actual[i][-1]
        position=data['positions'][5*j]
        value=(data['tokens'][5*j]*73+position*17+rows+i)%151936
        heads[5*j]=value;result.append(value)
    return dict(heads80=heads,real_tokens=result,metadata=dict(physical_M=rows))
original=reference460.target_step
try:
    reference460.target_step=host_target_step
    steps,canaries=reference460.generate_steps(SimpleNamespace(is_draft=False),lane,prompts,actual,list(range(6)),owners,schedules)
finally:reference460.target_step=original
assert actual==expected and all(len(v)==256 for v in actual)
visits={(owner,position):0 for owner in range(6)for position in range(1,256)}
for record in steps:
    for owner in record['real_owners']:
        position=record['step']+1;visits[(owner,position)]+=1
        assert schedules[owners[owner]][position]==record['physical_M']
assert all(value==1 for value in visits.values())
assert len(canaries)==6 and all(row['equal']is True and row['zeros']==row['alternate']for row in canaries)
# Draft executes exactly the same stage labels, without a target call.
draft_labels=[]
def draft_lane(label,callback):
    draft_labels.append(label);return callback()
draft_steps,draft_canaries=reference460.generate_steps(SimpleNamespace(is_draft=True),draft_lane,prompts,[[v]for v in initial],list(range(6)),owners,schedules)
assert draft_labels==labels and draft_steps==draft_canaries==[]
source=ast.parse((P/'reference460.py').read_text())
for name in('generate_steps','target_step'):
    fn=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name==name)
    assert not any(isinstance(n,ast.Name)and n.id in('candidate','candidate_tokens','trace')for n in ast.walk(fn))
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST460_SYMBOLIC_INDEPENDENT_AR_PARTITION_AND_CAUSALITY',
    HOST_fixture_only=True,simulated_owners=6,independent_predictions=6*255,
    every_owner_position_predicted_once=True,all_reference_inputs_from_own_previous_outputs=True,
    exact_per_position_M_schedule_preserved=True,all_future_canary_controls_passed=True,
    draft_and_target_stage_labels_identical=True,original_full_AR_source_inverse_passed=True,
    actual_model_constructors=0,NPU_used=False,NN_permission=False,Raw=None,Goodput=None)
with(P/'HOST460_REFERENCE_LOOP_RESULT.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out))
