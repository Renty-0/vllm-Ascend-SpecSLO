"""Two issued local scopes, observed-only protocol, then two complete zeros."""
from dataclasses import asdict
from scope151 import require,cases,source_contract,NATIVE_COUNTERS,FAILURES,digest
import draft151 as draft
import protocol151 as protocol

def native(e):
 m=e.graph_metrics();out={k:m[k] for k in NATIVE_COUNTERS}
 require(all(type(v)is int and v>=0 for v in out.values()) and not any(out[k] for k in FAILURES),'all21 original Native counters and zero failures')
 return out
def inventory(e):
 return {kind:[dict(key=repr(k),runtime_validated=bool(v.runtime_validated),validated_real_row_count=getattr(v,'validated_real_row_count',None))
  for k,v in getattr(e.graph_runner,name).items()] for kind,name in (('generic','entries'),('draft','draft_entries'),('target','target_entries'))}
def delta(before,after,*,zero):
 require(set(before)==set(after) and all(after[k]>=before[k] for k in before),'Native fields stable and monotone')
 result={k:after[k]-before[k] for k in before}
 require(not zero or not any(result.values()),'complete observed protocol trace zero discoveries/all21 counter delta0')
 return result

def run_actor(e,native_module,stage):
 import mc2_preflight151
 report=dict(source_contract=source_contract(),warmup=[],exploration=[],complete_zero=[[],[]],Native_flags_assigned=False,
  formal_Goodput=False,adopted=False,qualification=False)
 e._v151_partial=report
 from originals151 import original_function
 report['memory_before']=stage('original-memory-headroom-before',lambda:original_function('short_runtime_v126','memory')(e))
 report['MC2_static']=stage('original-all128-projection-all22-rows-MC2-static',lambda:mc2_preflight151.check(e) if not e.is_draft else None)
 for case in cases():
  report['warmup'].append(draft.warm(e,case,lambda n,f:stage(case.name+'/'+n,f)))
 def run(case,zero,label):
  before=stage(label+'/Native-before',lambda:native(e))
  trace=draft.prepare(e,case,lambda n,f:stage(label+'/'+n,f));trace['changed']=False
  result=protocol.run(e,native_module,case,trace,lambda n,f:stage(label+'/'+n,f))
  after=stage(label+'/Native-after',lambda:native(e))
  counters=stage(label+'/common-complete-trace-Native-delta',lambda:delta(before,after,zero=zero))
  return dict(case=asdict(case),draft_numerical=trace['numerical_proof'],generation=trace['generation_observation'],
   actual_source_lease=trace['actual_source_lease'],protocol=result,Native_before=before,Native_after=after,Native_delta=counters)
 for case in cases():report['exploration'].append(run(case,False,'explore/'+case.name))
 for repeat in range(2):
  for case in cases():report['complete_zero'][repeat].append(run(case,True,'zero'+str(repeat)+'/'+case.name))
 stage('original-unified-local-seal',e.seal_graph_cache)
 def final_gate():
  states=inventory(e);require(not states['generic'] and all(v['runtime_validated'] for k in ('draft','target') for v in states[k])
   and not e.graph_runner.disabled_entry_keys and bool(e.graph_runner.graph_cache_sealed),'all actual local keys naturally qualified and original seal')
  coverage=[row['protocol'] for row in report['exploration']]
  natural_positive=any(r['actual_promoted_child_consumed'] for r in coverage);natural_invalid=any(r['actual_natural_invalidation'] for r in coverage)
  per_scope=[dict(case=r['case']['repair'],natural_positive_and_consume=r['actual_promoted_child_consumed'],natural_invalidation=r['actual_natural_invalidation'],
   full_scope=r['actual_promoted_child_consumed'] and r['actual_natural_invalidation']) for r in coverage]
  full=all(v['full_scope'] for v in per_scope)
  return dict(status='passed_component' if full else 'passed_subset',natural_positive_and_real_child_consume=natural_positive,
   actual_natural_invalidation_and_nextdraft=natural_invalid,per_scope_natural_coverage=per_scope,full_local_scope_qualified=full,
   parent4_qualified=False,ordinary_deferE_production_gate_opened=False,adopted=False,formal_Goodput=False,
   final_native=native(e),actual_graph_states=states,source_epochs=getattr(e,'_v151_source_epoch',0))
 report['result']=stage('final-all4-local-coverage-and-inventory',final_gate)
 report['memory_after']=stage('original-memory-headroom-after',lambda:original_function('short_runtime_v126','memory')(e))
 stage('release-original-cache',e._release_cache)
 return report
