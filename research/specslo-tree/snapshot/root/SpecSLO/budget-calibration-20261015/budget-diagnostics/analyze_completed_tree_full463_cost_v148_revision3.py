"""Descriptive HOST costs only after a complete original independent audit.

Never imports a model, opens a tensor, launches a worker or queries hardware.
Native charged clocks and client stream clocks remain separate definitions.
"""
import argparse,ast,gzip,hashlib,json,math,os,re,statistics,sys
from collections import Counter
from pathlib import Path

AUDIT_SCRIPTS={
 'ROOT.V140.actual-normal-admission-audit.v1':'8505318726ab4665aa8137523e9b295120732b7d38cf5710fd23dc3778918944',
 'ROOT.V143.actual-standalone-prefill-control-audit.v1':'0c9fd194b03e0d01dcbbb2eae6c4bf42cf740694325f965820b21d9087857e35',
 'ROOT.V148.actual-full-cap4-audit.v1':'591ec9bd63c6c869056c596ae33216f1e8bb63a73492b4fb77f1fd888aa43838'}
ROLES=('warmup65','zero65-1','zero65-2','full463')
CLASSES=('tight','normal','loose')


def need(ok,message):
 if not ok:raise ValueError('completed tree cost: '+message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
 return h.hexdigest()
def desc(path):
 path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def read(path):
 def reject(value):raise ValueError('nonfinite JSON '+value)
 return json.loads(Path(path).read_bytes(),parse_constant=reject)
def artifact(info,directory):
 path=Path(info['path']).resolve();need(path.is_relative_to(directory) and desc(path)=={k:info[k] for k in ('path','bytes','sha256')},'actual artifact path/bytes/SHA')
 need(path.stat().st_size<=64*1024**2,'compressed bound')
 with gzip.open(path,'rb') as stream:raw=stream.read(256*1024**2+1)
 need(len(raw)==info['uncompressed_bytes'] and len(raw)<=256*1024**2,'expanded bound')
 return json.loads(raw)
def histogram(values):return dict(sorted(Counter(values).items()))
def ratio(a,b):return a/b if b else None
def available(mapping,key):return mapping[key] if key in mapping else None


def geometry(trace):
 calls=trace['normal_target_calls'];events=trace['logical_events']
 mixed=[e for e in events if e['kind']=='plan'];commits=Counter(e['cycle_id'] for e in events if e['kind']=='commit')
 need(all(commits[e['cycle_id']]==1 for e in mixed),'every mixed geometry actually committed')
 ordinary=Counter((c['physical_M'],c['physical_S']) for c in calls if c['family']=='full9')
 mixshapes=Counter((sum(e['query_partition']),len(e['query_partition'])) for e in mixed)
 prompt=Counter(e['end']-e['start'] for e in mixed)
 need(all(v>0 for v in prompt),'actual mixed prompt increments positive')
 normalM=sum(c['physical_M'] for c in calls);normalQ=sum(c['real_query_count'] for c in calls)
 mixedM=sum(sum(e['query_partition']) for e in mixed);mixedT=sum(len(e['target_ids'])+sum(e['budgets']) for e in mixed)
 prefill=sum(k*v for k,v in prompt.items());owners=sum(c['N'] for c in calls)+sum(len(e['target_ids']) for e in mixed)
 nodes=sum(sum(c['budgets']) for c in calls)+sum(sum(e['budgets']) for e in mixed)
 counters=trace['decode_counters'];rounds=counters['spec_rhythm_tree_rounds']
 need(owners==counters['spec_rhythm_tree_full_accepts']+counters['spec_rhythm_tree_rejections']
      and nodes==counters['spec_rhythm_verified_tokens'],'verified owner/node counters versus real call geometry')
 grouped=Counter()
 for c in calls:
  grouped[c['family'],c['N'],tuple(c['budgets']),c['physical_M'],c['physical_S'],c['real_query_count'],0]+=1
 for e in mixed:
  n=len(e['target_ids']);q=sum(e['budgets'])+n
  grouped['mixed-FULL',n,tuple(e['budgets']),sum(e['query_partition']),len(e['query_partition']),q,e['end']-e['start']]+=1
 geometries=[dict(family=f,N=n,B=list(b),M=m,S=s,Q=q,prompt_Q=p,dummy_Q=m-q-p,calls=v,wall_seconds=None,
                   wall_missing_reason='per-call/per-family wall was not retained')
             for (f,n,b,m,s,q,p),v in sorted(grouped.items())]
 standalone=trace['standalone_prefill_calls'];standaloneQ=sum(c['uncached_queries'] for c in standalone)
 return dict(target_rounds=rounds,ordinary_calls=len(calls),ordinary_causal=sum(c['family']=='causal' for c in calls),
  ordinary_FULL=sum(ordinary.values()),ordinary_FULL_shapes=[dict(M=m,S=s,calls=v) for (m,s),v in sorted(ordinary.items())],
  ordinary_N_hist=histogram(c['N'] for c in calls),ordinary_M_hist=histogram(c['physical_M'] for c in calls),
  ordinary_real_Q=normalQ,ordinary_physical_M_sum=normalM,ordinary_dummy_Q=normalM-normalQ,
  mixed_calls=len(mixed),mixed_shapes=[dict(M=m,S=s,calls=v) for (m,s),v in sorted(mixshapes.items())],
  mixed_N_hist=histogram(len(e['target_ids']) for e in mixed),mixed_prompt_Q_hist=dict(sorted(prompt.items())),
  mixed_prompt_positive_calls=sum(v for k,v in prompt.items() if k>0),mixed_prompt_zero_calls=sum(v for k,v in prompt.items() if k==0),
  mixed_prompt_Q=prefill,mixed_real_target_Q=mixedT,mixed_physical_M_sum=mixedM,mixed_dummy_Q=mixedM-mixedT-prefill,
  mixed_dummy_fraction=ratio(mixedM-mixedT-prefill,mixedM),mixed_fraction_missing_reason=None if mixedM else 'actual mixed call list is empty; ratio undefined',
  all_target_physical_Q_including_mixed_prompt_and_dummy=normalM+mixedM,FULL_physical_Q=sum(m*v for (m,s),v in ordinary.items())+mixedM,
  total_verified_owner_rounds=owners,total_verified_candidate_nodes=nodes,
  selected_target_B_hist=histogram([b for c in calls for b in c['budgets']]+[b for e in mixed for b in e['budgets']]),
  logical_event_counts=dict(Counter(e['kind'] for e in events)),standalone_calls=len(standalone),standalone_prompt_Q=standaloneQ,
  standalone_owner_rows=sum(len(c['sequence_ids']) for c in standalone),
  target_round_geometry_fully_covered=(rounds==len(calls)+len(mixed)),unrecorded_round_difference=rounds-len(calls)-len(mixed),
  real_geometry_table=geometries,per_family_wall_seconds=None,per_family_wall_missing_reason='No call clocks; volume/count averages cannot partition shared target wall')


def clocks(trace):
 rows=trace['generate_results'];need(len(rows)==463,'complete full463 output scope')
 first={};last={};tokens=[[] for _ in rows];finished=[False]*len(rows);previous=-1
 for event in trace['stream_events']:
  i=event['request_index'];need(type(i) is int and 0<=i<len(rows) and event['request_id']==rows[i]['request_id']
       and not finished[i] and event['elapsed_seconds']>=previous,'actual stream identity/order')
  previous=event['elapsed_seconds'];first.setdefault(i,previous);last[i]=previous;tokens[i].extend(event['token_ids']);finished[i]=event['finished']
 need(all(finished) and tokens==[r['completion_token_ids'] for r in rows] and all(len(t)==256 for t in tokens),'actual complete256 output/stream')
 wall=trace['host_generate_call_seconds'];need(wall==trace['elapsed_seconds'] and wall>0 and max(last.values())<=wall+1e-6,'original generate wall covers output')
 measures={name:[] for name in ('Native_paper','Native_observed','host_stream_paper','host_stream_observed')}
 for i,row in enumerate(rows):
  need(math.isclose(row['paper_tpot_ms'],row['observed_tpot_ms']*255/256,rel_tol=1e-10,abs_tol=1e-8),'reported Native denominator consistency')
  measures['Native_paper'].append(row['paper_tpot_ms']);measures['Native_observed'].append(row['observed_tpot_ms'])
  duration=(last[i]-first[i])*1000;measures['host_stream_paper'].append(duration/256);measures['host_stream_observed'].append(duration/255)
 def aggregate(values,indices,denominator,scope):
  selected=[values[i] for i in indices];ordered=sorted(selected);p=(len(ordered)-1)*.9;lo=math.floor(p);hi=math.ceil(p)
  attained=sum(values[i]<=rows[i]['slo_tpot_ms'] for i in indices)
  return dict(attained=attained,Goodput=256*attained/wall,mean_ms=statistics.mean(selected),median_ms=statistics.median(selected),
   P90_ms=ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo),TPOT_denominator=denominator,clock=scope)
 definitions=dict(Native_paper=(256,'original Native charged decode cycle time'),Native_observed=(255,'same Native charged decode cycle time'),
                  host_stream_paper=(256,'actual original event first-to-last wall'),host_stream_observed=(255,'same actual original event first-to-last wall'))
 result=dict(host_generate_wall_seconds=wall,Native_inner_seconds=trace['native_elapsed_seconds'],raw_tokens_per_second=len(rows)*256/wall)
 for name,values in measures.items():result[name]=aggregate(values,list(range(len(rows))),*definitions[name])
 result['by_class']={}
 for cls in CLASSES:
  indices=[i for i,row in enumerate(rows) if row['slo_class']==cls];need(indices,'actual class scope')
  result['by_class'][cls]=dict(requests=len(indices),SLO_ms=rows[indices[0]]['slo_tpot_ms'],
                             **{name:aggregate(values,indices,*definitions[name]) for name,values in measures.items()})
 result.update(P90_method='linear interpolation at zero-based (n-1)*0.9',
  Native_raw_finish_start_clocks_available=False,Native_formula_only_independent_clock_reconstruction=False,
  different_attainment_is_not_assigned_to_logging_error=True,clock_groups_are_not_added_or_substituted=True)
 return result


def future_conservation(traces,envelope,geom,combination):
 policy=envelope['actual_split_work_counts']['delta']['policy'];fast=envelope['actual_split_work_counts']['delta']['FastV5']
 c=traces[1]['decode_counters'];draft=traces[0]['decode_counters']
 needed=dict(generated=(policy,'actual_draft_eager_rows'),normal_rows=(policy,'actual_draft_normal_rows'),
  work_rows=(policy,'actual_draft_work_rows'),normal_nodes=(policy,'split_draft_final_normal_candidates'),
  future_nodes=(policy,'split_draft_final_eager_candidates'),promoted=(c,'spec_rhythm_tree_eager_promoted'),
  invalidated=(c,'spec_rhythm_tree_eager_invalidated'),aborted=(c,'spec_rhythm_aborted_requests'),
  roof_pruned=(c,'spec_rhythm_roof_pruned_candidates'),published_rows=(draft,'spec_rhythm_draft_publish_rows'),
  published_nodes=(draft,'spec_rhythm_tree_nodes'),frontier_rows=(fast,'eager_frontier_real_queries'))
 values={name:available(mapping,key) for name,(mapping,key) in needed.items()};missing=[name for name,value in values.items() if value is None]
 guards=dict(all463_complete256=True,fresh_allocation_all_ranks=all(t['fresh_allocation'] is True for t in traces),
  original_fresh_controller_source_bound=True,all_target_rounds_covered=geom['target_round_geometry_fully_covered'],
  no_aborted_requests=values['aborted']==0 if values['aborted'] is not None else None,
  no_roof_pruning=values['roof_pruned']==0 if values['roof_pruned'] is not None else None,
  exactB3_future_source_contract=(combination['eager_cap']==3 and combination['only_exactB3_eager_published'] is True
                                 and combination['eager_actual_shared_steps']==3 and combination['eager_deferred_when_normal_B_gt3'] is True))
 required_missing=[name for name in missing if name!='frontier_rows']
 if not required_missing:
  guards.update(all_model_work_published=values['published_rows']==values['work_rows'],
   normal_plus_future_work_rows=values['normal_rows']+values['generated']==values['work_rows'],
   all_selected_nodes_published=values['published_nodes']==values['normal_nodes']+values['future_nodes'],
   actual_selected_future_nodes_equal3perrow=values['future_nodes']==3*values['generated'])
 guards['one_actual_frontier_per_future_row']=(values['frontier_rows']==values['generated']) if values['frontier_rows'] is not None and values['generated'] is not None else None
 eligible=not required_missing and all(value is True for key,value in guards.items() if key!='one_actual_frontier_per_future_row')
 rowdiff=geom['total_verified_owner_rounds']-values['normal_rows'] if values['normal_rows'] is not None else None
 nodediff=geom['total_verified_candidate_nodes']-values['normal_nodes'] if values['normal_nodes'] is not None else None
 resolved=(values['generated']==values['promoted']+values['invalidated']) if not any(values[k] is None for k in ('generated','promoted','invalidated')) else None
 consume=(rowdiff==values['promoted'] and nodediff==3*rowdiff and rowdiff>=0) if rowdiff is not None and nodediff is not None and values['promoted'] is not None else None
 return dict(observed_raw_values=values,missing_fields=missing,required_missing_fields=required_missing,source_and_observation_preconditions=guards,
  conditional_aggregate_derivation_applicable=eligible,
  generated_future_partition=dict(equation='generated future rows = promoted + invalidated',numerically_equal=resolved),
  consumed_difference=dict(row_equation='verified owner rounds - normal published rows',candidate_equation='verified candidate nodes - normal selected/published nodes',
   raw_row_difference=rowdiff,raw_candidate_difference=nodediff,equals_promoted_and3nodes_per_row=consume,
   interpreted_consumed_rows=rowdiff if eligible and resolved and consume else None,
   interpreted_consumed_candidate_nodes=nodediff if eligible and resolved and consume else None),
  direct_consumed_counter_or_per_ticket_join_available=False,terminal_controller_ticket_maps_serialized=False,
  qualification='Conditional source-bound aggregate conservation; no per-child output/gain/latency or simultaneous live-owner alignment follows')


def draft_queries(envelope):
 fast=envelope['actual_split_work_counts']['delta']['FastV5'];rows=[]
 pairs=(('chain','chain_real_queries','chain_padded_queries'),('repair','repair_real_queries','repair_padded_queries'),
        ('frontier','eager_frontier_real_queries','eager_frontier_padded_queries'),('root_seed_PA','root_seed_PA_real_queries','root_seed_PA_padded_queries'))
 missing=[]
 for name,real,physical in pairs:
  absent=[key for key in (real,physical) if key not in fast];missing.extend(absent)
  rows.append(dict(component=name,real=available(fast,real),physical=available(fast,physical),missing_fields=absent))
 real=sum(row['real'] for row in rows) if not missing else None;physical=sum(row['physical'] for row in rows) if not missing else None
 return dict(real=real,physical=physical,padding=physical-real if not missing else None,components=rows,missing_fields=missing,
             generated_candidates=available(fast,'generated_candidates'),FastV5_counts=fast,
             generated_candidates_are_not_selected_ticket_gamma=True)


def source_contract(epoch,freeze):
 names=('runtime_tree_decode.py','atomic_decode_patch.py','original_hook_adapter.py','early_target.py','tree_entry.py',
        'production_probe_v101.py','poisson_coverage_probe.py','candidate_phases_v132.py','split_shape_v132.py','split_budget_v132.py')
 sources={}
 for name in names:
  path=epoch/name;need(freeze['files'][str(path)]==sha(path),'frozen source '+name);sources[name]=desc(path)
 runtime=(epoch/'runtime_tree_decode.py').read_text();tree=ast.parse(runtime)
 positions={}
 for name in ('draft_started','target_original_host_phase_ended','draft_ended'):
  nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
  need(len(nodes)==1,'unique original phase anchor '+name);positions[name]=nodes[0].lineno
 need(positions['draft_started']<positions['target_original_host_phase_ended']<positions['draft_ended'],'source nested target/draft intervals')
 need(runtime.count('controller = SpecRhythmPipelineController(states)')==1,'new empty controller on each original decode entry')
 atomic=(epoch/'atomic_decode_patch.py').read_text()
 need(atomic.count('if _v75_hooks.has_planned():\n    _v75_hooks.prepare_chunk_at_original_target_join(local_states)')==1,'actual conditional original late-join insertion')
 native=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')
 need(freeze['external_files'][str(native)]==sha(native),'frozen ticket lifecycle source');sources['Native.spec_rhythm.py']=desc(native)
 return dict(sources=sources,original_phase_source_lines=positions,rank0_target_interval_inside_rank0_draft_source=True,
  mixed_join_condition='has_planned only; standalone production has no mixed chunk work',
  intervals_are_HOST_worker_scopes_not_isolated_NPU_kernels=True,
  standalone_refill_scope='leader HOST interval around original _prefill_and_sample_target_batch; includes that helper, not pure attention kernel',
  per_call_or_family_wall_available=False)


def load_inputs(args):
 epoch=args.epoch_directory.resolve();rp=epoch/f'local-receipt-{args.tag}.json'
 need(desc(rp)['sha256']==args.expected_receipt_sha256,'ROOT terminal receipt exact SHA');receipt=read(rp)
 need(receipt['status']=='complete' and receipt['local_execution_complete'] is True and receipt['errors']==[]
      and receipt['child']['exit_code']==0,'complete original run only')
 need(desc(args.actual_audit)['sha256']==args.expected_audit_sha256,'ROOT independent full audit exact SHA')
 audit=read(args.actual_audit);need(audit['schema'] in AUDIT_SCRIPTS and audit['status']=='passed'
      and audit['actual_run_status']=='complete' and audit['receipt']==desc(rp),'independent audit passed for this actual receipt')
 need(audit['script']['sha256']==AUDIT_SCRIPTS[audit['schema']] and desc(audit['script']['path'])==audit['script'],'pinned independent auditor source')
 actual=audit['actual'];strict=actual['original_frozen_audit']
 need(actual['full463_completed'] is actual['original_frozen_worker_audit_reexecuted'] is strict['passed'] is True
      and strict['actual_S35_scope_count']==21 and strict['actual_S35_phase_count']==26
      and [x['role'] for x in strict['traces']]==list(ROLES),'original entire chain and four phases already passed')
 if audit['schema']=='ROOT.V148.actual-full-cap4-audit.v1':
  component=actual['own_full527_component']
  need(epoch==Path('/root/SpecSLO/budget-calibration-20261015/tree-high-normal-cap4-native-families-v148')
       and component['status']=='passed' and component['actual_full527_own_component_qualified'] is True
       and component['actual_local96_component_not_substituted'] is True
       and component['exact_actual_artifacts']==124,'actual own full527 variableB4 component already independently passed')
 if audit['schema'].startswith(('ROOT.V143.','ROOT.V148.')):
  mode=actual['production_standalone_prefill_control']
  need(mode['production_mixed_disabled'] is mode['same_M_mixed_21_scope_26_phase_still_required_and_not_relaxed'] is True
       and all(p['actual_mixed_calls']==p['actual_reservations']==p['actual_mixed_publications']==0 for p in mode['phases']), 'actual V143 standalone production gate')
 fp=epoch/'PREPARED_SOURCE_HASHES_V132.json';freeze=read(fp)
 need(audit['source_provenance']['passed'] is True and audit['source_provenance']['freeze']==desc(fp),'same audited actual freeze')
 directory=(epoch/'local-output'/args.tag).resolve();reports=[];traces=[];envelopes=[];artifacts=[]
 paths=[directory/f'rank{rank}-local-diagnostic.json' for rank in range(4)]
 need(receipt['report_paths']==list(map(str,paths)) and actual['final_worker_reports']==[desc(p) for p in paths],'same four fully audited reports')
 for rank,path in enumerate(paths):
  report=read(path);selected=[t for t in report['traces'] if t['role']=='full463'];need(len(selected)==1 and report['rank']==rank,'actual full worker identity')
  envelope=selected[0];trace=artifact(envelope['artifact'],directory)
  need(trace['rank']==rank and trace['role']=='full463' and trace['scope_count']==463
       and trace['source_generation']==envelope['source_generation']==actual['full463']['source_generation'],'full current generation')
  need(all(v==0 for v in trace['native']['delta'].values()) and len(trace['native']['delta'])==21,'full original zero21 Native counters')
  reports.append(report);envelopes.append(envelope);traces.append(trace);artifacts.append(envelope['artifact'])
 need(artifacts==actual['full463']['artifacts'],'same original independently audited full artifacts')
 for name in ('actual_gain_state','actual_normal_admission'):
  need(all(e[name]==envelopes[0][name] for e in envelopes),'actual four-rank '+name)
 need(all(e['actual_split_work_counts']['delta']['policy']==envelopes[0]['actual_split_work_counts']['delta']['policy'] for e in envelopes),
      'four-rank shared policy counts; FastV5 model counters remain role-local')
 return epoch,receipt,audit,freeze,reports,traces,envelopes,dict(receipt=desc(rp),actual_audit=desc(args.actual_audit),freeze=desc(fp),reports=[desc(p) for p in paths],artifacts=artifacts)


def compare_reference(result,path,wanted):
 need(desc(path)['sha256']==wanted,'reference summary SHA');old=read(path)
 fields=[('clocks','host_generate_wall_seconds'),('clocks','raw_tokens_per_second'),('geometry_full','target_rounds'),
         ('geometry_full','mixed_calls'),('geometry_full','standalone_calls'),('geometry_full','standalone_prompt_Q')]
 rows=[]
 for group,key in fields:
  a=old[group][key];b=result[group][key];rows.append(dict(field=group+'.'+key,reference=a,current=b,ratio=ratio(b,a),difference=b-a))
 return dict(reference=desc(path),descriptive_unpaired_only=True,formal_baseline_ratio_qualified=False,fields=rows)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch-directory',type=Path,required=True);p.add_argument('--tag',required=True)
 p.add_argument('--expected-receipt-sha256',required=True);p.add_argument('--actual-audit',type=Path,required=True);p.add_argument('--expected-audit-sha256',required=True)
 p.add_argument('--reference-summary',type=Path);p.add_argument('--expected-reference-sha256');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 need(sys.version_info[:2]==(3,11) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'retained3.11 and private tag')
 need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')),'HOST-only backend/thread caps')
 epoch,receipt,audit,freeze,reports,traces,env,inputs=load_inputs(args);geometry_full=geometry(traces[1]);clock=clocks(traces[1]);source=source_contract(epoch,freeze)
 audited=audit['actual']['full463']['output'];need(clock['Native_paper']['Goodput']==audited['native_paper']['goodput_tokens_per_second']
      and clock['host_stream_paper']['Goodput']==audited['event_wall_paper']['goodput_tokens_per_second'],'recomputed GP versus whole independent audit')
 combination=reports[1]['combination'];future=future_conservation(traces,env[0],geometry_full,combination)
 g=env[1]['actual_gain_state']['after'];need(g['actual_engine_generation']==traces[1]['source_generation'],'current actual gain generation')
 base={(x['slo_class'],r['B']):r['g'] for x in g['base'] for r in x['values']}
 gain=[dict(**row,g=row['delivered']/row['rounds'],calibrated_prior=base[row['slo_class'],row['B']],
       posterior_before_remaining_clamp=(12*base[row['slo_class'],row['B']]+row['delivered'])/(12+row['rounds'])) for row in g['observed']]
 draft=traces[0]['phase_seconds']['draft'];nested=traces[0]['phase_seconds']['target'];need(0<=nested<=draft,'actual nested rank0 HOST scope')
 c=traces[1]['decode_counters'];early={k:v/1e9 for k,v in c.items() if k.startswith('spec_rhythm_tree_early_target_') and k.endswith('_ns')}
 ad=env[1]['actual_normal_admission']['after'];refill=available(c,'spec_rhythm_online_refill_ms')
 result=dict(schema='HOST.completed-tree-full463-cost-summary.v1',status='complete-audited-descriptive-observations',actual_HOST_argv=sys.orig_argv,
  script=desc(__file__),inputs=inputs,source_contract=source,original_independent_full_audit_already_passed=True,original_auditor_or_parent_suites_rerun=False,
  clocks=clock,geometry_full=geometry_full,natural_eager=future,draft_queries=draft_queries(env[0]),
  observed_class_B_gain=gain,gain_current_generation=g['actual_engine_generation'],missing_actual_B_not_filled_with_zero=True,adaptive_gain_not_causal=True,
  normal_budget=dict(shape_calls=ad['call_count'],overflow_calls=ad['overflow_calls'],raw_N_hist=ad['raw_N_histogram'],admitted_N_hist=ad['admitted_N_histogram'],
   positive_grant_N_hist=ad['positive_grant_N_histogram'],normal_deferred_owner_records=len(ad['calls']),
   cap_deferred_owner_attempts=sum(len(e['cap_deferred_normal']) for e in ad['calls']),
   zero_grant_deferred_owner_attempts=sum(len(e['zero_grant_deferred_normal']) for e in ad['calls']),
   policy_counts=env[1]['actual_split_work_counts']['delta']['policy'],urgency_or_need_values_retained=False),
  phase_seconds_by_rank=[dict(rank=i,phase_seconds=t['phase_seconds']) for i,t in enumerate(traces)],
  nested_rank0_target=dict(seconds=nested,draft_seconds=draft,fraction_of_draft=ratio(nested,draft),draft_minus_nested_target_interval_seconds=draft-nested,
   difference_is_not_isolated_draft_device_time=True,rank0_target_time_is_inside_rank0_draft_interval=True),
  standalone_refill_seconds=refill/1000 if refill is not None else None,
  standalone_refill_missing_reason=None if refill is not None else 'raw leader HOST refill counter missing; no zero inferred',
  early_target_leader_seconds=early,early_target_missing_ns_fields=[k for k in ('submit_host_ns','CPU_scheduling_gap_ns','residual_wall_ns','total_wall_ns','worker_wall_prediction_ns')
    if 'spec_rhythm_tree_early_target_'+k not in c],
  task_update_counts_by_rank=[e['actual_task_update_counts'] for e in env],window_final_state_by_rank=[e['actual_window_cost_state']['after'] for e in env],
  per_family_or_percall_wall=None,limitations=['Worker phases overlap across roles and rank0 target is nested in draft; do not add them or call them pure NPU time.',
   'Geometry counts and Q volume do not identify causal/FULL/mixed wall; no average-based time split is made.',
   'Native charged TPOT and event stream wall TPOT remain independent source definitions; different attainment is not classified as logging error.',
   'The old raw Native finish/start clocks are absent; Native TPOT reports and the source formula do not reconstruct omitted clocks.',
   'Consumed future counts are conditional aggregate differences, without a per-ticket transcript or child-specific delivered-token g.',
   'Four ranks are one cooperating run. No formal paired-baseline ratio, AR claim or achieved objective follows.'],
  frozen_source_or_bridge_modified=False,Torch_imported=False,NPU_initialized=False,physical_cards_queried=False,model_or_query_executed=False)
 if args.reference_summary:
  need(args.expected_reference_sha256 is not None,'explicit reference identity');result['comparison_to_reference']=compare_reference(result,args.reference_summary,args.expected_reference_sha256)
 need(not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','numpy') for n in sys.modules),'stdlib HOST only')
 with args.output.open('x') as stream:stream.write(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(dict(output=desc(args.output),Native_paper_GP=clock['Native_paper']['Goodput'],stream_paper_GP=clock['host_stream_paper']['Goodput'],
       raw_tokens_per_second=clock['raw_tokens_per_second'],mixed_calls=geometry_full['mixed_calls'],future_aggregate_applicable=future['conditional_aggregate_derivation_applicable']),sort_keys=True))


if __name__=='__main__':main()
