"""Actual P161 natural parent/child/next closure, now in the full527 instance."""
from collections import Counter
from scope154 import require,case,contract,NATIVE_COUNTERS,FAILURES,digest
from base165 import V154,functions
import draft165 as draft
import source165 as epoch
import protocol165 as protocol
import target165 as target
functions(V154/'actor154.py',globals(),changes=((
 "gates=stage('discovery/raw-original-finish-gates',lambda:gate_rows(source,outputs,accepted));box['gates']=gates",
 "gates=stage('discovery/raw-original-finish-gates',lambda:gate_rows(source,outputs,accepted));box['gates']=gates\n  report['actual_discovery_gate_preview']=gates",1),
 ('import next154','import next165 as next154',1),
 ("staged_dependency=list(dependency),staged_frontier=eager['eager_frontier_token'],","staged_dependency=list(dependency),staged_frontier=eager['eager_frontier_token'],staged_parent_proposal_id=eager['eager_parent_proposal_id'],staged_dependency_length=length,",1),
 ("stage('release-original-cache',e._release_cache)","report['lease_error_status']=lease_error_status(e,stage)\n stage('release-original-cache',e._release_cache)",1)))

def lease_error_status(e,stage):
 """Expected HOST-only lease failure, actual common all4 Gloo raise.

 No cache mutation, model writes or proposal is performed by the negative. The
 original fatal handler's source is retained by the exact live AST inverse;
 this component catches its expected common status error before formal work.
 """
 from status159 import capture,common
 def local_negative():
  e._v159_draft_error=None
  if not e.is_draft:return dict(role='target',expected_local_error=False)
  from lease159 import guarded
  from repair_frontier_fusion import AllocationWitness
  before=epoch.ledger(e)
  try:
   with guarded(e,[0,1],AllocationWitness):e._release_cache_sequence(0)
  except RuntimeError as error:
   require('cache lifecycle mutation inside synchronous prelude' in str(error),'the real lease rejection was exercised')
   capture(e,error)
  else:raise RuntimeError('V165 expected actual lease rejection did not occur')
  require(epoch.ledger(e)==before,'negative made no ledger/device/allocation change')
  return dict(role='draft',expected_local_error=True,error=e._v159_draft_error)
 local=stage('lease-status/actual-protected-release-negative',local_negative)
 def shared():
  try:common(e)
  except RuntimeError as error:
   require('V159 common draft failure before proposal publication' in str(error),'actual common error marker')
   result=dict(all4_common_error_seen=True,error=str(error),new_collectives=1,
    model_or_KV_mutation=False,stale_success_or_proposal_returned=False)
  else:raise RuntimeError('V165 all4 status negative did not raise')
  finally:e._v159_draft_error=None
  return result
 return dict(local=local,common=stage('lease-status/common-all4-rejection-and-clear',shared),
  original_fatal_handler_rewritten=False,formal_generation_failed_on_purpose=False)
