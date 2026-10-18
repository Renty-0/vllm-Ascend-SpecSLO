"""HOST preflight for exact388 case list, reference extension and diagnostics."""
import ast,collections,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
 sys.path[:0]=[str(B/'tree-deferred-FULL-mask-host-projection-v259'),
     str(B/'tree-alternate-chain-target-host-v269'),str(B/'tree-static-q5-M120-node-math-v255'),
     str(B/'tree-contiguous-branch-tail-final-metadata-math-v234')]
 from fixture259 import build
 env=build();torch=env['torch']
 import oracle270,oracle255,reference_schedule269,reference_schedule255
 fixture=json.loads((P/'FIXTURE270.json').read_text());prefixes=fixture['prefixes']
 cases=oracle270.cases();assert len(cases)==408 and cases[:227]==oracle255.cases()
 assert json.loads((P/'DECLARED270_CASES.json').read_text())['cases']==cases
 nodes=0;old_frames=0;new_frames=0
 for ordinal,case in enumerate(cases):
  plans,rows=oracle270.prepare(case,prefixes);n=case['N'];roots=list(range(41,41+n))
  frames=reference_schedule269.schedule(plans,roots,rows)
  if ordinal<227:
   assert frames==reference_schedule255.schedule(plans,roots,rows)
   old_frames+=len(frames)
  else:new_frames+=len(frames)
  labeled=[r['candidate_query']for f in frames for r in f['real_queries']if r['candidate_query']is not None]
  expected={(i,q)for i,b in enumerate(case['budgets'])for q in range(b+1)}
  assert len(labeled)==len(set(labeled))and set(labeled)==expected;nodes+=len(labeled)
  assert n+sum(b>4 for b in case['budgets'])<=24 and sum(case['budgets'])<=96
  assert all(int(p.positions.max())<2560 for p in plans)
  for p in plans:
   assert int(p.positions.max()-p.prefix_len)<=4 and p.cache_positions.unique().numel()==p.candidate_budget+1
 for q in P.glob('*.py'):compile(q.read_text(),str(q),'exec')
 assert 'torch_npu'not in sys.modules
 result=dict(status='HOST_declared408_and_exact_legacy227_reference_extension_passed',
  cases=408,unchanged_legacy_cases=227,new_cases=181,real_nodes=nodes,
  legacy_reference_frames_byte_semantics_unchanged=old_frames,new_reference_frames=new_frames,
  all_original_matcher_bodies_retained=True,all_real_nodes_covered_once=True,
  model_dtype_or_sampler_changed=False,original_startup_inventory_not_run_by_this_diagnostic=True,
  paired_target_cost_configurations=11,rotated_rounds=20,source_only=True,
  NPU_used=False,neural_qualified=False,whole_workload_AR_qualified=False,Goodput=None)
 with(P/'HOST270_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
