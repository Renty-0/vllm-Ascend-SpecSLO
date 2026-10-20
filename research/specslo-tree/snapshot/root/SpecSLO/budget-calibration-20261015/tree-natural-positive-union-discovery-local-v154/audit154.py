"""Thin actual report checks, not hidden/KV tensor recomputation."""
from scope154 import require,contract,NATIVE_COUNTERS
def audit(reports):
 require(len(reports)==4,'all4 actual components')
 common=None;statuses=[]
 for rank,report in enumerate(reports):
  require(report['rank']==rank and report['errors']==[] and report['cache_released'] is True
   and report['Native_flags_assigned'] is report['formal_Goodput'] is False,'actual worker completion')
  r=report['result'];require(r['contract']==contract() and r['full_qualification'] is r['two_complete_zero_traces'] is r['performance_measured'] is False,'explicit limited scope')
  d=r['discovery'];gates=d['gates'];promoted=d['promoted_owners'];invalid=d['invalidated_owners']
  require(len(gates)==32 and [v['owner'] for v in gates]==list(range(32,64))
   and sorted(promoted+invalid)==list(range(32,64)),'complete actual parent owner partition')
  for row in gates:
   require(row['accepted_count']==len(row['accepted_nodes']) and row['predicted_original_finish_eager_valid']==all(row['gates'].values()),'actual original guard conjunction')
  require([v['owner'] for v in gates if v['predicted_original_finish_eager_valid']]==promoted,'actual original promotion equals actual verdict predicate')
  value=dict(gates=gates,promoted=promoted,invalid=invalid)
  require(common is None or common==value,'four-rank exact raw verifier/ticket evidence');common=value
  require(set(r['Native_before'])==set(r['Native_after'])==set(r['Native_delta'])==set(NATIVE_COUNTERS)
   and all(r['Native_after'][k]-r['Native_before'][k]==r['Native_delta'][k]>=0 for k in NATIVE_COUNTERS),'actual Native counter subtraction')
  if promoted:
   u=r['union_sameM'];require(u is not None and u['sameM_pass'] is True and u['union_M']==36
    and u['independent_union_PA'] is u['all_physical_hidden'] is u['exact_full_physical_encoded'] is u['all56_written_query_KV'] is u['exact_ledger'] is True,'new actual union same-M own numerical receipts')
   c=r['continuation'];require(c['real_promoted_child_consumed'] is c['original_protocol_closed'] is True
    and c['consumed_owners']==promoted,'true consume and nextdraft actual numerical gates')
   require(c['nextdraft']['passed'] is True if rank==0 else c['nextdraft'] is None,'only actual draft role returns next numerical proof')
   require(len(r['phases'])==3 and r['result']['status']=='positive-consumed-and-next','actual full necessary continuation')
  else:
   require(r['union_sameM'] is r['continuation'] is None and r['result']['status']=='negative-only-subset'
    and r['early_stop']['normal_or_child_consume_claimed'] is False and len(r['phases'])==1,'negative discovery stops before whole oracle')
  statuses.append(r['result'])
 return dict(schema='V154.actual-discovery-audit.v1',status='passed-component-evidence' if common['promoted'] else 'passed-negative-subset',
  positive_count=len(common['promoted']),negative_count=len(common['invalid']),per_rank=statuses,
  hidden_KV_recomputed_by_HOST=False,two_complete_zeros_qualified=False,full_qualification=False,
  production_gates_opened=False,formal_Goodput=False,adopted=False)
