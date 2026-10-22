"""Explicit synthetic arithmetic controls; never actual NN/results/permission."""
import copy,hashlib,importlib.abc,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST final-goal control cannot import NN')
sys.meta_path.insert(0,NoModel())
import final_common_comparison as base
import final_common_comparison_slo_v2 as tested

def cells(counts,tree_raw=1000.):
 out=[]
 for block,order in enumerate(base.ORDERS,1):
  for pos,method in enumerate(order):
   n=counts[block-1]if method=='qualified_dynamic_tree'else 300
   gp=None if method=='qualified_dynamic_tree'else(base.FIXED_FLOOR if method=='fixed_gamma4'else base.OFFICIAL_FLOOR)
   raw=tree_raw if gp is None else gp*463/n
   if gp is None:gp=raw*n/463
   val=dict(source_and_environment_verified=True,physical_cards=[2,5,6,7],all_owned_workers_exited=True,all4_idle_after=True)
   if method=='qualified_dynamic_tree':val.update(all_original_and_new_gates_passed=True,whole_workload_AR_qualified=True,AR_equal_requests=463,AR_different_requests=0,candidate_freeze_sha256='synthetic-SAME-SOURCE')
   elif method=='official_MC2_ON':val.update(ordinary_MC2_actually_enabled=True,MC2_projection_modules_by_rank=[128]*4)
   metrics=dict(all=dict(requests=463,output_tokens=118528,Goodput=gp,Raw=raw,attained=n),by_class={k:dict(requests=n)for k,n in dict(tight=271,normal=96,loose=96).items()},comparison_method=method,request_identity=dict(sha256=base.EXPECTED_WORKLOAD),source_descriptors=[dict(path=f'EXPLICIT-HOST-SYNTHETIC-{block}-{pos}')])
   out.append(dict(block=block,position=pos,method=method,tag=f'HOST-SYNTHETIC-{block}-{pos}',metrics=metrics,validation=val))
 return out
controls=[]
for ns,want in [([400,400,400],True),([350,350,350],False),([400,370,400],False),([371,371,371],True),([370,370,370],False)]:
 c=cells(ns);old=base.aggregate(c);r=tested.aggregate(c)
 assert old['stated_performance_and_full_AR_goals_met']is True
 assert r['stated_all_Goodput_SLO_and_full_AR_goals_met']is want
 controls.append(dict(synthetic_attained=ns,original_throughput_only_passed=True,new_complete_goal_passed=want))
c=cells([400]*3,tree_raw=600.)
assert tested.aggregate(c)['stated_all_Goodput_SLO_and_full_AR_goals_met']is False
bad=cells([400]*3);bad[0]['metrics']['all']['Goodput']+=1.
try:tested.aggregate(bad)
except ValueError:pass
else:raise AssertionError('inconsistent Goodput fraction accepted')
value=dict(status='PASS_HOST_FINAL_STABLE80_SLO_AND_ORIGINAL_NINE_CELL_CONDITIONS',source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(tested.__file__),Path(base.__file__),Path(__file__)]},explicit_synthetic_inputs_only=True,synthetic_validation_fields_are_test_doubles_not_NN_permissions=True,controls=controls,low_Goodput_despite_high_SLO_rejected=True,inconsistent_Goodput_rejected=True,NPU_used=False,NN_qualification=False,Raw=None,Goodput=None)
with(D/'FINAL_STABLE80_SLO_V2_HOST_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in value.items()if k!='source_sha256'}))
