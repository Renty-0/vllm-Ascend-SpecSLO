"""HOST-only gain estimator checks using retained original-verdict witnesses."""
import json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
 sys.path.append(str(B/'tree-high-normal-cap4-native-families-v148'))
 from budget_policy import TreeBudgetPolicy
 from paired268 import install,observe,snapshot,begin_generation
 model=json.loads((B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
 witnesses=json.loads((B/'budget-diagnostics/ROOT_ORIGINAL_B5_B4_PAIRED_VERDICT_HOST_RESULT.json').read_text())
 assert witnesses['status']=='HOST_original_greedy_B5_vs_B4_counterfactual_proved'
 policy=TreeBudgetPolicy(model)
 # A cohort-confounded raw mean can make naive g5 lower than g4. For this
 # nested topology that difference cannot be a same-prefix causal effect.
 policy.observed['tight',4]=(100,400);policy.observed['tight',5]=(100,250)
 request=types.SimpleNamespace(slo_class='tight',output_token_budget=256,delivered_tokens=1)
 original=policy.gain;assert original(request,5)<original(request,4)
 before={n:original(request,n)for n in range(9)}
 state=install(policy);assert install(policy)is state
 assert policy.gain(request,5)>policy.gain(request,4)
 assert all(policy.gain(request,n)==before[n]for n in range(9)if n!=5)
 begin_generation(policy,1)
 samples=0
 for cls in ('tight','normal','loose'):
  request.slo_class=cls
  for witness in witnesses['examples'].values():
   parents=[-1,0,1,2,witness['parent']]
   path=witness['tree_path'];tokens=witness['tree_output']+[-1]*(5-len(witness['tree_output']))
   for _ in range(10):
    row=observe(policy,request,parents=parents,accepted=path,outputs=tokens,remaining_before=255)
    assert row['same_prefix_B4']==len(witness['chain_output'])
    assert row['B5']==len(witness['tree_output'])
    samples+=1
   observed=dict(policy.observed)
   for remaining in (1,2,3,4):
    assert observe(policy,request,parents=parents,accepted=path,outputs=tokens,remaining_before=remaining)is None
   assert policy.observed==observed
  for delivered in (252,253,254,255,256):
   request.delivered_tokens=delivered
   assert 0<=policy.gain(request,5)<=256-delivered
  request.delivered_tokens=1
  assert policy.gain(request,5)>=policy.gain(request,4)
  assert all(policy.gain(request,n)==original(request,n)for n in range(9)if n!=5)
 request.slo_class='tight';prior=snapshot(policy)
 bad=[dict(parents=[-1,0,1,2,3],accepted=[0,-1,-1,-1],outputs=[10,99,-1,-1,-1]),
      dict(parents=[-1,0,1,2,-1],accepted=[0,-1,1,-1],outputs=[10,20,99,-1,-1])]
 for case in bad:
  try:observe(policy,request,remaining_before=255,**case)
  except ValueError:pass
  else:raise AssertionError('malformed original verdict accepted')
 assert snapshot(policy)==prior and 'torch_npu'not in sys.modules
 assert sum(r['samples']for r in prior['by_class'].values())==samples
 assert policy.counts['ROOT268_actual_B5_paired_observations']==samples
 policy.observed.clear()  # The real original fresh_decode does this itself.
 begin_generation(policy,2)
 fresh=snapshot(policy)
 assert sum(r['samples']for r in fresh['by_class'].values())==0
 assert sum(r['samples']for r in fresh['by_generation']['1']['by_class'].values())==samples
 assert sum(r['samples']for r in fresh['by_generation']['2']['by_class'].values())==0
 assert begin_generation(policy,2)is state
 try:begin_generation(policy,1)
 except ValueError:pass
 else:raise AssertionError('stale generation accepted')
 result=dict(status='HOST_paired_B5_gain_estimator_and_original_verdict_witnesses_passed',
   examples_from_original_verdict=len(witnesses['examples']),HOST_fixture_observations=samples,
   original_other_budget_predictor_implementation_preserved=True,
   confounded_raw_B5_below_B4_does_not_create_false_negative_side_gain=True,
   terminal_horizon_observations_excluded=True,malformed_original_verdicts_rejected=True,
   original_fresh_gain_boundary_preserved=True,stale_generation_rejected=True,
   state=fresh,NPU_used=False,production_installed=False,neural_qualified=False,
   empirical_branch_probability_measured=False,Goodput=None)
 with(P/'HOST268_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items()if k!='state'}))
if __name__=='__main__':main()
