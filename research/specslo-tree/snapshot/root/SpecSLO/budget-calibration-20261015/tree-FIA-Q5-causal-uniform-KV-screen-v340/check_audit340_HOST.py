"""Schema regressions use fabricated HOST records, never device evidence."""
import copy,json,tempfile
from pathlib import Path
from probe340 import PREFIXES,HEADS,variants,REPEATS,ROUNDS
from audit340 import audit
P=Path(__file__).resolve().parent

def main():
 rows=[];timings=[]
 for h,k in HEADS:
  for p in PREFIXES:
   for name,e,i in variants(p):
    for phase in ('A1','B','A2'):
     rows.append(dict(heads=h,KVheads=k,prefix=p,name=name,extent=e,inner_precise=i,phase=phase,
      physical_M=120,groups=24,width=5,finite=True,independent_root_raw16_equal_by_depth=[True]*5,
      independent_root_max_abs_by_depth=[0.]*5,group_position_raw16_equal_by_depth=[True]*5,
      max_abs_uniform_FULL2560=0.,all_outputs_equal_uniform_FULL2560=True,
      complete_output_restored=True if phase=='A2' else None))
   names=['FULL2560']+([] if variants(p)[0][0]=='FULL2560' else [variants(p)[0][0]])+['causal-Native','causal-inner1']
   timings.append(dict(heads=h,KVheads=k,prefix=p,names=names,repeats=REPEATS,rotations=ROUNDS,
    all_graph_outputs_raw16_equal_eager=True,median_ms={n:1. for n in names},
    pairs=[dict(order=names[j%len(names):]+names[:j%len(names)],milliseconds_per_FIA={n:1. for n in names})for j in range(ROUNDS)]))
 fixture=dict(schema='ROOT340.synthetic-Q5-causal-uniform-KV.v1',seed=340,source_sha256='HOST-fixture',
  rows=rows,timings=timings,models_loaded=False,model_qualified=False,full_model_or_service_performance=False,Goodput=None)
 def reject(value):
  f.write_text(json.dumps(value))
  try:audit(f,expected_source='HOST-fixture')
  except RuntimeError:return
  raise AssertionError('corrupt HOST fixture accepted')
 with tempfile.TemporaryDirectory(dir=P)as t:
  f=Path(t)/'fixture.json';f.write_text(json.dumps(fixture));assert audit(f,expected_source='HOST-fixture')['rows']==294
  changes=[lambda v:v['rows'].pop(),lambda v:v['rows'][0].update(physical_M=121),
   lambda v:v['rows'][0].update(independent_root_raw16_equal_by_depth=[False]*5),
   lambda v:v['rows'][0].update(max_abs_uniform_FULL2560=1.),
   lambda v:v['rows'][2].update(complete_output_restored=False),
   lambda v:v['timings'][0]['pairs'][0]['milliseconds_per_FIA'].update(FULL2560=-1.),
   lambda v:v['timings'][0]['median_ms'].update(FULL2560=2.),
   lambda v:v.update(model_qualified=True)]
  for mutate in changes:
   v=copy.deepcopy(fixture);mutate(v);reject(v)
 result=dict(status='passed_HOST_auditor_only',fabricated_positive_schema_accepted=True,
  corrupt_fixtures_rejected=len(changes),no_actual_numeric_claim=True,NPU_used=False)
 (P/'HOST340_AUDITOR_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
