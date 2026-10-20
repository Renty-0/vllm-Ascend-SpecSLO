"""Small terminal structural audit; hidden/KV equality remains actual raw proof."""
from scope151 import require,cases,NATIVE_COUNTERS

def audit(reports):
 require(len(reports)==4 and sorted(r['rank'] for r in reports)==list(range(4)),'four actual original workers')
 statuses=[]
 for r in reports:
  require(r['action']=='normal4-parent3-child3-shared4-local' and not r['errors'] and r['cache_released'] is True
   and r['Native_flags_assigned'] is False and r['formal_Goodput'] is False,'actual component report terminal bounds')
  x=r['result'];require(len(x['exploration'])==2 and list(map(len,x['complete_zero']))==[2,2],'explore and two entire local traces')
  require(x['result']['adopted'] is x['result']['parent4_qualified'] is x['result']['ordinary_deferE_production_gate_opened'] is False,'scope remains local; no parent4 or production gate')
  for trace in [x['exploration'],*x['complete_zero']]:
   for row,c in zip(trace,cases(),strict=True):
    require(row['case']['repair']==c.repair and row['generation']['actual_shared_steps']==4
     and row['generation']['real_raw_nodes']==512 and row['generation']['selected_total']==192
     and row['draft_numerical']['all28_written_query_KV'] is True and row['draft_numerical']['exact_full_physical_encoded'] is True,'true mixed4 raw8/own complete draft reference')
    p=row['protocol'];require(p['synthetic_verdict'] is False and p['parent4_qualified'] is False and p['outcome']=='observed_target','only observed original verdict')
    require([v['label'] for v in p['phases']]==['parent3','normal-variable4']+(['naturally-promoted-child3'] if p['promoted_owners'] else []),'actual parent→normal→natural child')
    require(sorted(p['promoted_owners']+p['invalidated_owners'])==list(range(32,64)),'all natural parent outcomes retained')
    if r['role']=='target':
     for phase in p['phases']:
      q=phase['target'];require(q['shape']['N']<=32 and q['shape']['C']<=96 and q['shape']['real_Q']<=128
       and q['all128query_KV'] is q['allphysical_hidden'] is q['independent_complete_history'] is True
       and phase['committed_KV']['all128_committed_KV'] is True,'each actual native target exact legal partition/sameM/commit KV')
    else:require(p['next_draft']['passed'] is p['next_draft']['own_old_shared4_history_rebuilt'] is True,'draft true next after promotion or invalidation')
  for trace in x['complete_zero']:
   for row in trace:require(set(row['Native_delta'])==set(NATIVE_COUNTERS) and not any(row['Native_delta'].values()),'complete trace21 counters zero')
  events=r['boundary_events'];require(events and all(e['sequence']==i and all(v['error'] is None and v['sequence']==i and v['stage']==e['stage'] for v in e['ranks']) for i,e in enumerate(events)),'common actual stage ordering/no failed boundary')
  statuses.append(x['result']['status'])
 require(len(set(statuses))==1,'all4 same local coverage conclusion')
 return dict(status=statuses[0],structural_audit=True,hidden_KV_recomputed_by_HOST=False,qualification='actual local component only',
  production_adopted=False,parent4_qualified=False,formal_Goodput=False)
