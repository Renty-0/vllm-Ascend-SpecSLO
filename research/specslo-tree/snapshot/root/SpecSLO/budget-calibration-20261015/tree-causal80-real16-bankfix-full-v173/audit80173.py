"""Retained raw80 heads and original commit arithmetic; no HOST tensor claim."""
import gzip,json
from pathlib import Path
from base173 import require,sha
from mapper173 import GREEDY,HIDDEN
from scope_v146 import NATIVE_COUNTERS,FAILURES
from audit_v146 import expected_missing

def load(path):return json.loads(gzip.decompress(Path(path).read_bytes()))

def chain_verdict(rows,head):
 require(len(rows)==16 and len(head)==80 and all(len(r)==4 for r in rows),'actual sixteen linear4 query blocks')
 emitted=[];paths=[]
 for i,row in enumerate(rows):
  h=head[i*5:i*5+5];n=0
  while n<4 and row[n]==h[n]:n+=1
  emitted.append(row[:n]+[h[n]]);paths.append(list(range(n))+[-1]*(4-n))
 return emitted,paths

def repair(proof,previous):
 n=expected_missing(previous,[4]*16)
 require(proof['natural_missing_suffix_rows']==n and proof['natural_missing_suffix_observed']==(n>0),'natural repair count from actual previous accepted4+bonus')
 physical=16 if not n else next(v for v in (20,24,28,32) if v>=16+n)
 require(proof['first_PA_physical']==physical,'real N16 repair PA physical bucket')
 return n

def audit(directory):
 directory=Path(directory);reports=[load(directory/f'rank{r}-component.json.gz') for r in range(4)]
 all_history=[];labels=[];pins={};cross=[];missing=False;zeros=0
 for rank,r in enumerate(reports):
  require(r['schema']=='V173.causal80-own-component.v1' and r['rank']==rank and r['role']==('draft' if rank==0 else 'target')
   and r['completed'] is r['math_independent_qualified'] is r['cache_released'] is True and r['errors']==[]
   and r['adopted'] is r['formal_Goodput'] is r['cost_measured'] is r['Native_flags_assigned'] is r['component_sealed'] is False,
   'actual complete bounded same-M qualification only')
  cfg=r['configuration'];require(cfg['maxseq']==64 and cfg['maxqueued']==463 and cfg['storage']==527 and cfg['pages']==544 and cfg['block_size']==128,
   'full Native scope preserved')
  require(len(r['traces'])==3,'discovery plus two complete zero traces')
  history=[];rank_cross=[]
  for p,t in enumerate(r['traces']):
   require(t['pass_index']==p and t['zero_replay']==(p>0) and t['complete_protocol_cycles']==4,'complete four-cycle trace')
   n=t['Native'];require(set(n['before'])==set(n['after'])==set(n['delta'])==set(NATIVE_COUNTERS)
    and n['delta']=={k:n['after'][k]-n['before'][k] for k in NATIVE_COUNTERS}
    and all(v>=0 for v in n['delta'].values()) and not any(n['delta'][k] for k in FAILURES),'real21 Native subtraction')
   if p:require(not any(n['delta'].values()),'whole protocol zero no new capture/validation');zeros+=21
   lease=t['lease']['lease'];require(lease['pages']==108 and lease['storage_rows']==527
    and lease['candidate_owners']==list(range(16)) and lease['reference_owners']==list(range(16,32))
    and lease['dummy_owners']==list(range(463,507)) and lease['source_KV_copied'] is False,'disjoint actual108/544 pages')
   previous=None;trace_cross=[]
   for c in range(4):
    path=directory/f'rank{rank}-pass{p}-cycle{c}.json.gz';row=load(path);pins[str(path)]=sha(path)
    require(row['cycle']==c and row['selected']==[list(range(4))]*16 and len(row['rows'])==16
     and row['selected_normal_candidates']==64 and row['commit']['actual_structural_failed']==row['commit']['actual_numerical_failed']==0
     and row['commit']['original_finish_cycle_executed'] is True and row['independent_commit']['histories_equal'] is True
     and row['independent_commit']['candidate_KV_copied'] is False,'actual typed linear4 original verdict/commit and independent next history')
    if rank==0:
     proof=row['draft'];require(proof['raw_encoded_shape']==[16,4,2] and proof['raw_pool_nodes']==128
      and proof['all_physical_hidden_original_matcher'] and proof['all28_query_KV_original_matcher'] and proof['exact_full8_encoded']
      and proof['independent_owners']==list(range(16,32)) and proof['source_KV_copied'] is False,'all physical own16 draft hard gate')
     missing|=repair(proof,previous)>0
    else:
     a=row['target'];ref=row['reference'];old=row['old128_control'];cm=row['crossM']
     counts=row['original_autoqualification'];cold=p==0 and c==0
     expected_extra={'qualified_entries':1 if cold else 0,'untimed_extra_forwards':2 if cold else 0}
     require({k:counts['after'].get(k,0)-counts['before'].get(k,0) for k in expected_extra}==expected_extra,'actual original cold greedy autoqualifier extra-forward counters')
     require(a['all_physical_hidden'] and a['all64_query_KV'] and a['production_greedy_M']==a['separate_hidden_counterpart_M']==80
      and a['M1_head_recompute'] is False and ref['all_physical_hidden_original_matcher'] and ref['all64_query_KV_original_matcher']
      and ref['all_physical_greedy_exact'] and ref['candidate_KV_copied'] is False and ref['same_M']==80 and ref['same_S']==16,
      'new80 fullphysical hidden/all64queryKV/original80 head independent hard gates')
     for family,key in (('greedy',GREEDY),('hidden',HIDDEN)):
      phases=a[family];require([v['phase'] for v in phases]==['first','changed','restored'] and all(v['key']==repr(key) for v in phases),'actual new key F/T/F phases')
      require(all(v['runtime_validated_after'] is True for v in phases[1:]) and phases[0]['runtime_validated_before'] is (not(p==0 and c==0)),
       'cold exactly once and original natural changed gate')
      require(phases[0]['runtime_validated_after'] is (family=='greedy' or not(p==0 and c==0)),'cold greedy autoqualifier vs hidden original false first')
     g=a['greedy'];require(g[0]['full_physical_tokens']==g[2]['full_physical_tokens']==ref['real_query_tokens']==cm['new_real80']
      and len(cm['old_real80'])==len(cm['new_real80'])==80 and old['all128'][:80]==cm['old_real80']
      and old['sameM_greedy_and_KV_checked'] is True and old['new_hidden_key_created'] is False,'retained complete old/new80 heads and own old full128 check')
     positions=[i for i,(x,y) in enumerate(zip(cm['old_real80'],cm['new_real80'],strict=True)) if x!=y]
     require(cm['different_head_positions']==positions and cm['head_equal']==(not positions),'cross-M exact positions recomputed')
     newtokens,newpaths=chain_verdict(row['rows'],cm['new_real80']);oldtokens,oldpaths=chain_verdict(row['rows'],cm['old_real80'])
     require(newtokens==row['commit']['committed_tokens'] and newpaths==row['commit']['accepted_paths']==cm['new_accepted'],
      'raw full80 head recomputes actual original chain verdict and acceptance')
     require([[v for v in row if v>=0] for row in cm['new_verdict_tokens']]==newtokens
      and [[v for v in row if v>=0] for row in cm['old_verdict_tokens']]==oldtokens and cm['old_accepted']==oldpaths,
      'both actual Native host verifier results independently recomputed')
     equal=not positions and newtokens==oldtokens and newpaths==oldpaths
     require(cm['verdict_equal']==(cm['old_verdict_tokens']==cm['new_verdict_tokens']) and cm['accepted_equal']==(newpaths==oldpaths),'crossM verdict flags retain actual comparison')
     trace_cross.append(equal)
    previous=row['commit']
   path=directory/f'rank{rank}-pass{p}-next.json.gz';nxt=load(path);pins[str(path)]=sha(path)
   require({k:nxt[k] for k in ('source','selected','rows','confidences')}==t['next_draft'],'actual next draft retained')
   if rank==0:missing|=repair(nxt['draft'],previous)>0
   else:require(t['crossM_behavior_equal']==all(trace_cross),'crossM actual per-trace equality');rank_cross.extend(trace_cross)
   history.append(dict(commits=t['fingerprints'],next_draft=t['next_draft']))
  require(history[0]==history[1]==history[2],'fresh independent history exact two zero replays')
  all_history.append(history)
  if rank:require(r['old_padding_behavior_equal']==all(rank_cross),'separate compatibility classification');cross.append(rank_cross)
  events=r['boundary_events'];sequence=[(e['sequence'],e['label']) for e in events]
  require([e['sequence'] for e in events]==list(range(len(events))) and all(len(e['ranks'])==4
   and {v['rank'] for v in e['ranks']}==set(range(4)) and all(v['sequence']==e['sequence'] and v['label']==e['label'] and v['error'] is None for v in e['ranks']) for e in events),
   'actual same shared all4 stages and no role failure')
  labels.append(sequence)
  inv=r['inventory'];wanted={repr(GREEDY),repr(HIDDEN)} if rank else set()
  require(inv['inherited_entry_identity_preserved'] and inv['disabled_keys_unchanged'] and inv['Native_flags_assigned'] is False,'Native inherited identity/flags preserved')
  for f in ('entries','draft_entries','target_entries'):
   require(set(inv['before'][f])<=set(inv['after'][f]) and set(inv['after'][f])-set(inv['before'][f])==(wanted if f=='target_entries' else set()),'only two new80 Native keys')
  require(r['MC2_dispatch']['delta']['fallback']==r['MC2_dispatch']['delta']['exception']==0
   and (rank==0 or r['MC2_dispatch']['delta']['native_epilogue_success']>0),'actual original MC2 route')
  require(r['gain_reset']['after']['observed_after']==[] and r['gain_reset']['after']['calibrated_prior_unchanged'] is True
   and r['fresh_boundary']['known_KV_ledger_empty'] is True,'fixture gain/ledger cannot leak')
 require(all(h==all_history[0] for h in all_history) and all(v==labels[0] for v in labels),'all4 actual logical history/stage order')
 require(missing and zeros==4*2*21,'own natural missing suffix plus all168 full-trace21 zeros')
 require(all(v==cross[0] for v in cross),'TP actual crossM outcomes agree')
 compatible=all(cross[0])
 return dict(schema='V173.retained-causal80-audit.v1',status='passed',math_independent_qualified=True,
  old_padding_behavior_equal=compatible,numerical_variant_only=not compatible,adopted=False,formal_Goodput=False,
  own_natural_missing_suffix_observed=True,complete_protocol_commits=12,complete_protocol_zero_traces=2,
  actual21_zero_subtractions=zeros,crossM_all80_and_verdict_recomputed=True,
  crossM_comparisons=12,crossM_compatible_comparisons=sum(cross[0]),cause_established=False,
  HOST_recomputes_hidden_or_KV=False,actual_device_sameM_gates_required=True,
  raw_checkpoint_sha256=pins,raw_report_sha256={str(directory/f'rank{r}-component.json.gz'):sha(directory/f'rank{r}-component.json.gz') for r in range(4)})
