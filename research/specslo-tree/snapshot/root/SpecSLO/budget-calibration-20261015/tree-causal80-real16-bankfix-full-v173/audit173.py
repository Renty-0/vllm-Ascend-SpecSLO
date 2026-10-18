"""Actual raw-only audit: unchanged original arithmetic plus window boundaries."""
import ast,copy,gzip,json,sys
from pathlib import Path
from base173 import P,BASE,ACTION,read,require,checked,functions,inherited_module,sha
from window160 import windows

def audit(directory,selected_windows):
 directory=Path(directory).resolve();require(directory.is_relative_to(P),'own actual directory')
 selected_windows=windows(selected_windows)
 original=inherited_module('audit_production_v101')
 reports=[read(directory/f'rank{rank}-local-diagnostic.json') for rank in range(4)]
 from audit_v146 import audit_reports as audit_component
 component=audit_component(directory/'variable-B4-own-full-process')
 require(component['component_numerical_qualified'] is True,'own retained original B4 component')
 from audit80173 import audit as audit80
 causal80=audit80(directory/'causal80-own-current-process')
 mode=inherited_module('no_mixed_prefill_v143');mode_ns=dict(vars(mode))
 functions(BASE/'no_mixed_prefill_v143.py',mode_ns,names=('audit_production_mode_trace',),changes=((
  'old_qualification_scopes_retained=True','old_qualification_scopes_retained=False,new_standalone_scope_only=True',1),))
 original_ns=dict(vars(original))
 functions(BASE/'audit_production_v101.py',original_ns,names=('output_arithmetic',),changes=((
  'no added timer/callback','unchanged elapsed fields; separate first-token wall samples retained',1),))
 functions(BASE/'audit_production_v101.py',original_ns,names=('audit_trace_set',))
 original_trace_audit=original_ns['audit_trace_set']
 roles=('warmup65','zero65-1','zero65-2','full463')
 schedule=[(w,'warmup65') for w in selected_windows]+[(w,r) for r in roles[1:3] for w in selected_windows]+[(w,'full463') for w in selected_windows]
 sequences=[];warm_schedules=[]
 for rank,r in enumerate(reports):
  require(r['schema']=='V173.active-window-full-worker.v1' and r['rank']==rank and r['action']==ACTION
   and r['errors']==[] and r['startup_completed'] is r['sealed'] is r['cache_released'] is r['full463_completed'] is True,
   'actual complete current worker')
  require(r['configuration_before_constructor']==r['configuration_after_constructor'],'unchanged original constructor config')
  require(r['private_IO']['output']==str(directory) and r['private_IO']['read_inputs']==str(BASE)
   and r['private_IO']['parent_directory_written'] is False,'private current runtime/progress output identity')
  require(r['old_21scope26phase_executed'] is False and r['standalone_scope']['N']==[]
   and r['standalone_scope']['mixed_reservation_allowed'] is r['standalone_scope']['original_S35_scope_claimed'] is False,'no borrowed mixed scope')
  startup=r['startup'];counts=dict(entries=0,draft_entries=60,target_entries=54) if rank==0 else dict(entries=0,draft_entries=0,target_entries=245)
  require(startup['actual_native_entry_counts']==counts and r['component']['audit']==component and r['component']['causal80']['audit']==causal80,'actual201/44/114 plus current B4 proof')
  require(r['component']['old_S19_S35_actor_executed'] is r['component']['old_21scope26phase_claimed'] is False,'unused actor not run')
  traces=r['windows']['traces'];require([(t['window'],t['role']) for t in traces]==schedule,'exact selected independent trace order')
  generations=[t['original_trace']['source_generation'] for t in traces]
  require(generations==list(range(generations[0],generations[0]+len(schedule))),'fresh generation per actual trace')
  sequences.append(generations)
  require(r['windows']['configuration']==dict(configured_capacity=64,queued=463,storage=527,scratch=64,
   target_max=32,candidate_roof=96,admission_windows=list(selected_windows)),'fixed Native capacity with selected admission windows')
  prewarm=r['prewarm'];require(prewarm['all_windows_converged'] is True and prewarm['Native_flags_assigned'] is False,'actual bounded prewarm')
  warm_order=[]
  for w in selected_windows:
   rows=[v for v in prewarm['traces'] if v['window']==w]
   require(1<=len(rows)<=8 and [v['attempt'] for v in rows]==list(range(len(rows)))
    and rows[-1]['agreement']['all21_all4_zero'] is True,'actual per-window convergence')
   for row in rows:
    raw=original.load_artifact(row['trace']['artifact'],directory);original.native_delta(raw,False)
    require(raw['scope_count']==65 and raw['admission_window160']==w,'own prewarm actual source')
    agreement=row['agreement']['actual_rank_records'];require(len(agreement)==4 and {v['rank'] for v in agreement}==set(range(4)),'prewarm all4 packets')
    require(row['agreement']['all21_all4_zero']==all(not any(v['native']['delta'].values()) for v in agreement),'actual warm convergence subtraction')
    warm_order.append((w,row['attempt'],row['trace']['source_generation']))
  warm_schedules.append(warm_order)
  require(r['final_controls']['mode']=='original' and r['final_controls']['CPU']['passed'] is True,'original CPU/controls restored')
 require(all(s==sequences[0] for s in sequences) and all(s==warm_schedules[0] for s in warm_schedules),'cross-rank generation/prewarm sequence')
 audited=[];native_zero_subtractions=0
 from mapper173 import expected_geometry
 from candidate_phases_v132 import split_delta,UPDATE_FIELDS
 for index,(cap,role) in enumerate(schedule):
  envelopes=[r['windows']['traces'][index] for r in reports]
  raw=[original.load_artifact(v['original_trace']['artifact'],directory) for v in envelopes]
  source_rows,prefix=original.manifest_rows('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl',463 if role=='full463' else 65)
  value=original_trace_audit(raw,source_rows,prefix,zero=role!='warmup65')
  standalone=[mode_ns['audit_production_mode_trace'](t,len(source_rows)) for t in raw]
  require(all(v==standalone[0] for v in standalone),'cross-rank standalone whole-prompt admission')
  for rank,(trace,envelope) in enumerate(zip(raw,envelopes,strict=True)):
   require(trace['admission_window160']==cap and trace['configured_capacity']==64 and 0<trace['active_peak']<=cap
    and trace['actual_window_covered160']==(trace['active_peak']==cap) and trace['actual_capacity64_scope_completed'] is False,
    'truthful actual window peak distinct from configured64')
   require(trace['sealed_before'] is (role=='full463') and trace['sealed_after'] is (role=='full463'),'one seal after every two zero traces')
   require('S35_structural_scope' not in trace and trace['standalone_scope173']==reports[rank]['standalone_scope'],'actual standalone scope reported')
   before=envelope['actual_split_work_counts']['before'];after=envelope['actual_split_work_counts']['after']
   require(envelope['actual_split_work_counts']['delta']==split_delta(before,after),'real per-trace split Counter arithmetic')
   require(envelope['actual_gain_state']['after']['actual_engine_generation']==trace['source_generation']
    and envelope['actual_gain_state']['before']['base']==envelope['actual_gain_state']['after']['base'],
    'own actual gain generation and unchanged calibrated prior')
   change=envelope['actual_task_update_counts'];require(all(set(change[s])==set(UPDATE_FIELDS) for s in ('before','after','delta')),'real TTU/skip field schema')
   require(all(change['after'][k]-change['before'][k]==change['delta'][k]>=0 for k in UPDATE_FIELDS),'real TTU/skip subtraction')
   if role!='warmup65':native_zero_subtractions+=len(trace['native']['delta'])
  for c in raw[1]['normal_target_calls']:
   n,q,f=c['N'],c['real_query_count'],c['family'];m,s,padding=expected_geometry(n,q,f)
   require((c['physical_M'],c['physical_S'],c['physical_query_partition'])==(m,s,[b+1 for b in c['budgets']]+list(padding)),'exact candidate mapper geometry')
  metrics=None
  if role=='full463':
   from trace173 import metrics as real_metrics
   metrics=real_metrics(envelopes[1]['original_trace']['artifact'])
   retained=reports[1]['windows']['per_rank_full_metrics'];require(next(v['output'] for v in retained if v['window']==cap)==metrics,'independently recomputed original clocks plus actual TTFT')
  audited.append(dict(window=cap,role=role,trace=value,standalone=standalone[0],metrics=metrics))
 require(native_zero_subtractions==4*len(selected_windows)*3*21,'all actual21 zero/full counter differences')
 return dict(schema='V173.actual-active-window-full-audit.v1',passed=True,windows=list(selected_windows),traces=audited,
  native_zero_subtractions=native_zero_subtractions,original_B4_component=component,causal80_component=causal80,math_independent_qualified=True,
  old_padding_behavior_equal=causal80['old_padding_behavior_equal'],numerical_variant_only=causal80['numerical_variant_only'],old_21scope26phase_executed=False,
  original_arrivals_and_outer_wall=True,hidden_KV_recomputed_by_HOST=False,whole_workload_AR_qualified=False,
  actual_full463_per_window=True,formal_Goodput=False,adopted=False)
