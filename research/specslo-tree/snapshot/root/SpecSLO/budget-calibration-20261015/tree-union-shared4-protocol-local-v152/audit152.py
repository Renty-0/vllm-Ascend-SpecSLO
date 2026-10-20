"""Own structural audit: no HOST tensor qualification or synthetic outcomes."""
from scope152 import require,cases,NATIVE_COUNTERS

def audit(reports):
 require(len(reports)==4 and sorted(r['rank'] for r in reports)==list(range(4)),'four own actual workers')
 statuses=[]
 for r in reports:
  require(r['action']=='union-shared4-protocol-local' and not r['errors'] and r['cache_released'] is True
   and r['Native_flags_assigned'] is r['formal_Goodput'] is False,'actual terminal local component bounds')
  x=r['result'];require(len(x['exploration'])==4 and list(map(len,x['complete_zero']))==[4,4],'four repair scopes/two whole zeros')
  require(x['result']['adopted'] is x['result']['parent4_qualified'] is x['result']['ordinary_deferE_production_gate_opened'] is False,'no production gate or parent4')
  for trace in [x['exploration'],*x['complete_zero']]:
   for row,c in zip(trace,cases(),strict=True):
    require(row['case']['repair_count']==c.repair_count and row['generation']['repair_count']==c.repair_count
     and row['generation']['repair_owners']==list(c.repair_owners) and row['generation']['union_M']==c.union_M
     and row['generation']['actual_shared_steps']==4 and row['generation']['real_raw_nodes']==512
     and row['generation']['selected_total']==192,'actual partial repairs/true shared4/raw8')
    d=row['draft_numerical'];require(d['sameM_pass'] is d['independent_union_PA'] is d['all56_written_query_KV'] is d['exact_full_physical_encoded'] is True
     and d['union_M']==c.union_M and d['head_M']==32,'own same-U hidden/56KV/encoded canonical8')
    ttu=row['generation']['actual_TTU'];require(ttu['dynamic_TTU_proved'] is ttu['dynamic_prefix_cost_measured'] is False
     and all(type(v)is int and v>=0 for v in ttu['delta'].values()),'real numeric-window TTU counters only, no dynamic-cost inference')
    p=row['protocol'];require(p['synthetic_verdict'] is p['parent4_qualified'] is False and p['outcome']=='observed_target','only original actual target outcome')
    require([v['label'] for v in p['phases']]==['parent3','normal-variable4']+(['naturally-promoted-child3'] if p['promoted_owners'] else []),'real target phase order')
    require(sorted(p['promoted_owners']+p['invalidated_owners'])==list(range(32,64)),'all actual natural parent outcomes retained')
    if r['role']=='target':
     for phase in p['phases']:
      q=phase['target'];require(q['shape']['N']<=32 and q['shape']['C']<=96 and q['shape']['real_Q']<=128
       and q['all128query_KV'] is q['allphysical_hidden'] is q['independent_complete_history'] is True
       and phase['committed_KV']['all128_committed_KV'] is True,'own actual same-M target/commit proof per legal ticket partition')
    else:require(p['next_draft']['passed'] is p['next_draft']['own_old_shared4_history_rebuilt'] is True,'real next3 from own union old-source history')
  for trace in x['complete_zero']:
   for row in trace:
    a,b,delta=row['Native_before'],row['Native_after'],row['Native_delta']
    require(set(a)==set(b)==set(delta)==set(NATIVE_COUNTERS) and all(type(a[k])is type(b[k])is type(delta[k])is int and 0<=a[k]<=b[k] and b[k]-a[k]==delta[k]==0 for k in a),'all raw21 zero subtractions')
  events=r['boundary_events'];require(events and all(e['sequence']==i and all(v['error'] is None and v['sequence']==i and v['stage']==e['stage'] for v in e['ranks']) for i,e in enumerate(events)),'actual all4 common stage sequence')
  statuses.append(x['result']['status'])
 require(len(set(statuses))==1 and statuses[0] in ('passed_component','passed_subset'),'consistent actual coverage')
 return dict(status=statuses[0],structural_audit=True,hidden_KV_recomputed_by_HOST=False,qualification='actual sequential union component only',production_adopted=False,parent4_qualified=False,formal_Goodput=False,dynamic_prefix_cost_measured=False)
