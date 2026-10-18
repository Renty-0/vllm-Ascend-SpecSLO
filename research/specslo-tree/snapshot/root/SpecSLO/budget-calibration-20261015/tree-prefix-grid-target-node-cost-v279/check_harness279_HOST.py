"""Regression: real historical cold dispatches and draft bank roots=None."""
import copy,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
 sys.path[:0]=[str(B/'tree-deferred-FULL-mask-host-projection-v259'),str(B/'tree-alternate-chain-target-host-v269')]
 from fixture259 import build
 env=build();engine=env['engine']
 from audit_static_calls279 import audit_calls
 old=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/static-M120-root-v266-restored-repair-byte-copy-full463-ownAR-v1'
 proof=json.loads((old/'canonical-independent-node-math/rank1-proof.json').read_text())
 cases=[row['case']for row in proof['math']['cases']]
 calls=proof['actual_static_graph_calls']
 result=audit_calls(cases,calls)
 assert result['positive_node_cases']==227 and result['actual_graph_dispatches']==231
 assert result['original_changed_input_restore_extra_dispatches']==4
 for bad in (calls[1:],calls[:1]+calls[2:],list(reversed(calls))):
  try:audit_calls(cases,bad)
  except RuntimeError:pass
  else:raise AssertionError('damaged historical cold-call sequence accepted')
 # Execute the complete draft role setup using the actual None bank field.
 # These retained counter values are only a CPU control fixture, not a run.
 import gzip
 t=json.loads(gzip.decompress((old/'windows/window48/full463/rank0-full463-actual-trace.json.gz').read_bytes()))
 engine.rank=0;engine.is_draft=True
 engine.graph_metrics=lambda:dict(t['native']['before'])
 from benchmark279 import run
 fixture=json.loads((P/'FIXTURE279.json').read_text())
 labels=[]
 def stage(label,fn):labels.append(label);return fn()
 value=run(engine,None,stage,dict(prefixes=fixture['prefixes'],roots=None))
 assert len(value['rows'])==11 and len(labels)==1155
 assert value['target_math_on_rank']is False
 assert all(v is None for row in value['rows']for pair in row['pairs']for v in pair['milliseconds'].values())
 assert all(row['original_greedy_output_stable']is None for row in value['rows'])
 assert 'torch_npu'not in sys.modules
 report=dict(status='HOST_real_cold_trace_and_None_draft_bank_regressions_passed',
   actual_historical_numeric_cases=227,actual_historical_dispatches=231,
   actual_original_changed_restored_extra_dispatches=4,damaged_cold_sequences_rejected=3,
   draft_None_bank_full_control_stages=len(labels),draft_no_neural_timing_claim=True,
   role_control_fixture_not_measured_NPU_data=True,NPU_used=False,Goodput=None)
 with(P/'HOST_HARNESS279_RESULT.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':main()
