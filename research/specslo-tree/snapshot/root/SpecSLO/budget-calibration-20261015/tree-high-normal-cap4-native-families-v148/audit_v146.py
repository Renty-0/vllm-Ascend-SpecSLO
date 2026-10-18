"""Strict retained JSON audit; numerical tensor gates remain actual device checks."""
import gzip,json
from pathlib import Path
from scope_v146 import require,sha,cases,VECTORS,NATIVE_COUNTERS,FAILURES

def load(path):return json.loads(gzip.decompress(Path(path).read_bytes()))

def expected_missing(previous_commit,budgets):
 if previous_commit is None:return 0
 paths=previous_commit['accepted_paths'];tokens=previous_commit['committed_tokens']
 require(len(paths)==len(tokens)==len(budgets),'previous original commit owner count for natural repair')
 return sum(b==4 and len([j for j in p if j>=0])==4 and len(t)==5 for b,p,t in zip(budgets,paths,tokens,strict=True))

def validate_repair_evidence(proof,expected):
 require(type(proof['natural_missing_suffix_rows']) is int and proof['natural_missing_suffix_rows']==expected
  and type(proof['natural_missing_suffix_observed']) is bool and proof['natural_missing_suffix_observed']==(expected>0),
  'natural missing marker/count must derive from previous actual B4 all-accepted plus bonus commit')
 physical=32 if not expected else next(n for n in (36,40,44,48,64) if n>=32+expected)
 require(proof['first_PA_physical']==physical,'actual repair PA bucket must fit exactly the natural missing count')

def audit_reports(directory):
 directory=Path(directory);reports=[load(directory/f'rank{r}-component.json.gz') for r in range(4)]
 require([r['rank'] for r in reports]==list(range(4)) and [r['role'] for r in reports]==['draft','target','target','target'],'exact role/rank ownership')
 fingerprints=None;labels=None;qualified=None;checkpoint_pins={};observed_missing={0:False,1:False};final_next_missing={0:False,1:False}
 for report in reports:
  rank=report['rank'];target=rank!=0
  require(report['completed'] and report['passed_subset'] and report['sealed'] is False and report['cache_released'] and not report['errors']
   and report['cost_measured'] is report['formal_Goodput'] is report['adopted'] is report['full_workload_qualified'] is False,
   'bounded completed subset, no performance/adoption claim')
  require(report['construction']['maxseq']==64 and report['construction']['maxqueued']==463 and report['construction']['storage']==527
   and report['construction']['block_size']==128 and report['construction']['pages']==544,'new actual config domain')
  cfg=report['actual_config']
  require(cfg['max_num_seqs']==64 and cfg['max_num_queued_seqs']==463 and cfg['kvcache_block_size']==128
   and cfg['num_kvcache_blocks']==544,'actual safe Native config snapshot agrees with full-process contract')
  require(len(report['traces'])==6 and report['complete_protocol_zero_traces']==2,'exact six four-cycle traces/two fixed protocol zeros')
  current=[]
  for index,trace in enumerate(report['traces']):
   p,c=divmod(index,2);require(trace['pass_index']==p and trace['case']==c and trace['budgets']==list(VECTORS[c])
    and trace['complete_protocol_cycles']==4 and trace['zero_replay']==(p>0),'fixed two-vector/pass domain')
   raw=trace['Native'];require(set(raw['before'])==set(raw['after'])==set(raw['delta'])==set(NATIVE_COUNTERS)
    and raw['delta']=={k:raw['after'][k]-raw['before'][k] for k in NATIVE_COUNTERS}
    and all(v>=0 for v in raw['delta'].values()) and not any(raw['delta'][k] for k in FAILURES),'raw exact Native delta')
   if p:require(not any(raw['delta'].values()),'fixed complete protocol zero must discover/validate nothing')
   require(trace['lease']['lease']['pages']==140 and trace['lease']['lease']['storage_rows']==527,'actual independent bank/page lease')
   previous_commit=None;protocol_missing=False
   for cycle in range(4):
    path=directory/f'rank{rank}-pass{p}-case{c}-cycle{cycle}.json.gz';row=load(path);checkpoint_pins[str(path)]=sha(path)
    require(row['cycle']==cycle and row['commit']['actual_structural_failed']==row['commit']['actual_numerical_failed']==0
     and row['commit']['original_finish_cycle_executed'] and row['independent_commit']['histories_equal']
     and not row['independent_commit']['candidate_KV_copied'],'true original commit/independent history checkpoint')
    require([len(v) for v in row['rows']]==list(VECTORS[c]) and row['selected']==[list(range(b)) for b in VECTORS[c]],'retained actual variable-B selection')
    if target:
     require(row['target']['all_physical_hidden'] and row['target']['all64_query_KV'] and row['target']['full_physical_greedy_cloned_immediately']
      and row['reference']['all_physical_hidden_original_matcher'] and row['reference']['all64_query_KV_original_matcher']
      and row['reference']['all_physical_greedy_exact'] and row['reference']['candidate_KV_copied'] is False,'own same-M all physical target numerical gates')
     for arm in row['target']['greedy']:
      require(len(arm['full_physical_tokens'])==cases()[c]['M'],'all physical production returned tokens retained')
     require(row['target']['greedy'][0]['full_physical_tokens']==row['target']['greedy'][2]['full_physical_tokens'],'full physical F/T/F exact restore')
    else:
     proof=row['draft'];require(proof['all_physical_hidden_original_matcher'] and proof['all28_query_KV_original_matcher']
      and proof['exact_full8_encoded'] and proof['raw_encoded_shape']==[32,4,2] and proof['selected_C']==sum(VECTORS[c])
      and proof['variable_num_draft_tokens']==list(VECTORS[c]) and proof['source_KV_copied'] is False,'actual draft fullpool/exact ABI/independent reference')
     expected=expected_missing(previous_commit,VECTORS[c]);validate_repair_evidence(proof,expected)
     observed_missing[c]|=expected>0
     protocol_missing|=expected>0
    previous_commit=row['commit']
   path=directory/f'rank{rank}-pass{p}-case{c}-next-draft.json.gz';nextrow=load(path);checkpoint_pins[str(path)]=sha(path)
   require(nextrow['source']==trace['next_draft']['source'] and nextrow['rows']==trace['next_draft']['rows'],'true next-draft after final commit retained')
   if not target:
    expected=expected_missing(previous_commit,VECTORS[c]);validate_repair_evidence(nextrow['draft'],expected)
    final_next_missing[c]|=expected>0
    require(trace['natural_missing_suffix_protocol_observed']==protocol_missing
     and trace['final_next_draft_natural_missing_suffix_observed']==(expected>0)
     and trace['natural_missing_suffix_observed']==(protocol_missing or expected>0),'protocol repair coverage is separate from final next-draft-only repair')
   current.append(dict(commits=trace['fingerprints'],next_draft=trace['next_draft']))
  require(current[:2]==current[2:4]==current[4:6],'two fresh-history zero replays preserve exact logical trace')
  if fingerprints is None:fingerprints=current
  else:require(current==fingerprints,'all four original protocol logical histories independently agree')
  events=report['boundary_events'];seq=[(r['sequence'],r['label']) for r in events]
  require(not any('unified-seal' in r['label'] for r in events),'own component cannot execute the future production seal')
  retained=report['retained_boundary_prefix']
  require(retained==dict(stage_count=len(events),last_label=events[-1]['label'] if events else None,
   saving_stage_and_later_audit_not_retained=True),'retained report boundary is the exact pre-save prefix, not future events')
  require([r[0] for r in seq]==list(range(len(events))) and all(len(e['ranks'])==4 and sorted(r['rank'] for r in e['ranks'])==list(range(4))
   and all(r['sequence']==e['sequence'] and r['label']==e['label'] and r['error'] is None for r in e['ranks']) for e in events),'complete original common boundary order/errors')
  if labels is None:labels=seq
  else:require(labels==seq,'all four fixed shared stage sequences')
  if qualified is None:qualified=report['component_numerical_qualified']
  else:require(qualified==report['component_numerical_qualified'],'qualification coverage is a shared all4 decision')
  require(report['component_numerical_qualified']==(not report['missing_suffix_uncovered_vectors']),'missing natural coverage cannot become qualified')
  dispatch=report['MC2_dispatch']['delta']
  require(dispatch['fallback']==dispatch['exception']==0 and (not target or dispatch['native_epilogue_success']>0),'same full-process actual original MC2 dispatch')
  from full_process_v146 import validate_inventory_delta
  inventory=report['inventory_delta']
  validate_inventory_delta(inventory['before'],inventory['after'],is_draft=not target)
  require(inventory['inherited_entry_identity_preserved'] and inventory['Native_flags_assigned'] is False
   and inventory['disabled_keys_unchanged'],'full-process inherited Native entries retained')
  require(report['gain_reset']['before']['calibrated_prior_unchanged'] and report['gain_reset']['after']['calibrated_prior_unchanged']
   and report['gain_reset']['after']['observed_after']==[],'fixture observations cannot enter production gain state')
  fresh=report['production_fresh_boundary']
  require(fresh['original_cache_release_completed'] and fresh['known_KV_ledger_empty']
   and fresh['defer_flags_restored'] and fresh['runner_top2_restored']
   and fresh['Native_entries_flags_and_graph_owned_buffers_cleared'] is False,'fixture-only state must not enter fresh production allocation')
 missing=[c for c,v in observed_missing.items() if not v]
 for report in reports:
  votes=report['all4_required_coverage']
  require([v['rank'] for v in votes]==list(range(4)) and [v['role'] for v in votes]==['draft','target','target','target']
   and all(v['passed_subset'] and v['complete_protocol_zero_traces']==2 for v in votes),'retained all4 required-coverage votes')
  require(votes[0]['natural_missing_by_vector']=={str(c):v for c,v in observed_missing.items()}
   and report['missing_suffix_uncovered_vectors']==missing and report['component_numerical_qualified']==(not missing),
   'retained raw cycle/nextdraft natural coverage independently agrees with all4 votes and qualified flag')
  require(all(v['target_all_physical_sameM_checked'] for v in votes[1:]) and not votes[0]['target_all_physical_sameM_checked'],
   'role-correct target same-M required coverage')
 return dict(schema='V146.retained-component-audit.v1',status='passed',passed_subset=True,
  component_numerical_qualified=qualified,missing_suffix_uncovered_vectors=reports[0]['missing_suffix_uncovered_vectors'],
  independently_recomputed_protocol_missing_coverage={str(c):v for c,v in observed_missing.items()},final_next_draft_missing_coverage={str(c):v for c,v in final_next_missing.items()},
  qualification_exploration_commits=8,complete_fixed_protocol_replay_commits=16,formal_Goodput=False,adopted=False,
  future_full_two65zero_required=True,raw_checkpoint_sha256=checkpoint_pins,
  raw_report_sha256={str(directory/f'rank{r}-component.json.gz'):sha(directory/f'rank{r}-component.json.gz') for r in range(4)},
  HOST_recomputes_full_hidden_or_KV=False,actual_device_matcher_gates_retained=True)
