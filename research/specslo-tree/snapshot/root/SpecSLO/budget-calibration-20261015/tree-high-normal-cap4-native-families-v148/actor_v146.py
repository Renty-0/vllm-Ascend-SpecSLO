"""Bounded component qualification. No costs, client timers or Goodput."""
import gzip,json,os,traceback
from pathlib import Path
from scope_v146 import require,cases,validate_full_configuration,inventory_record,NATIVE_COUNTERS,FAILURES,MAX_CYCLES,digest

def save(directory,name,value):
 raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
 if name.endswith('.gz'):raw=gzip.compress(raw,mtime=0)
 with (Path(directory)/name).open('xb') as stream:stream.write(raw)

def native(engine):
 metrics=engine.graph_metrics();out={k:metrics[k] for k in NATIVE_COUNTERS}
 require(all(type(v) is int and v>=0 for v in out.values()),'all21 raw Native fields')
 return out

def delta(before,after,*,zero=False):
 values={k:after[k]-before[k] for k in NATIVE_COUNTERS}
 require(all(v>=0 for v in values.values()) and not any(values[k] for k in FAILURES),'no Native failure/fallback')
 if zero:require(not any(values.values()),'complete fixed four-cycle protocol zero discovered or revalidated a graph')
 return dict(before=before,after=after,delta=values)

def stage4(engine,events,label,callback,*,share=False):
 import torch,torch.distributed as dist
 result=None;error=None;seq=len(events)
 try:result=callback();torch.npu.synchronize()
 except BaseException:error=traceback.format_exc()
 rows=[None]*4
 dist.all_gather_object(rows,dict(rank=engine.rank,sequence=seq,label=label,error=error,evidence=result if share and error is None else None),group=engine.groups.verification_coordination_group)
 events.append(dict(sequence=seq,label=label,ranks=rows))
 require({r['rank'] for r in rows}==set(range(4)) and all(r['sequence']==seq and r['label']==label and r['error'] is None for r in rows),
  'actual all4 common stage '+repr(rows))
 engine._vote_spec_rhythm_prefill_finiteness()
 return rows if share else result

def fingerprint(record):
 """JSON-stable true history/selection evidence, excluding pointer/cold flags."""
 return dict(source=record['source'],selected=record['selected'],rows=record['rows'],confidences=record['confidences'],
  commit={k:record['commit'][k] for k in ('committed_tokens','accepted_paths','actual_prefix_epochs','actual_logical_tokens',
   'actual_structural_failed','actual_numerical_failed')})

def coverage_decision(records):
 require([v['rank'] for v in records]==list(range(4)),'all4 coverage ranks in original gather order')
 draft=[v for v in records if v['role']=='draft']
 require(len(draft)==1 and draft[0]['rank']==0 and [v['rank'] for v in records if v['role']=='target']==[1,2,3], 'actual draft0/targets123 coverage ownership')
 require(set(draft[0]['natural_missing_by_vector'])=={0,1},'both declared variable-B fixtures have truthful coverage states')
 missing=[i for i,observed in draft[0]['natural_missing_by_vector'].items() if not observed]
 qualified=not missing and all(v['passed_subset'] and v['complete_protocol_zero_traces']==2 for v in records)
 return missing,qualified

def repair_coverage(records,next_record):
 protocol=any(row['draft']['natural_missing_suffix_observed'] for row in records)
 final=next_record['draft']['natural_missing_suffix_observed']
 return dict(natural_missing_suffix_observed=protocol or final,natural_missing_suffix_protocol_observed=protocol,
  final_next_draft_natural_missing_suffix_observed=final)

def run_actor(engine,native_module,compiled,directory,events):
 import torch
 from dual_bank_v146 import build_bank
 from draft_v146 import generate
 from target_v146 import qualify,reference,commit_reference
 from original_tree_cycle_v75 import setup_normal_cycle,attach_normal_cycle_proposals,run_original_tree_cycle
 from config_snapshot_v138 import snapshot_config
 from import_origins_v132 import require_loaded_origins as assert_origins
 s=lambda label,callback:stage4(engine,events,label,callback)
 target=not engine.is_draft
 report=dict(schema='V146.variable-B4-component.v1',rank=engine.rank,role='draft' if engine.is_draft else 'target',
  errors=[],traces=[],completed=False,component_numerical_qualified=False,adopted=False,
  cost_measured=False,formal_Goodput=False,full_workload_qualified=False,future_full_requires_21scope26phase_and_two65zero=True,
  initial_source_root_teacher_forced_not_own_AR=True,subsequent_draft_verdict_commit_actual=True,
  actual_config=s('configuration',lambda:snapshot_config(engine.config)),
  actual_source_origins=s('source-origins',assert_origins),physical_cards=[2,5,6,7],physical_rank_card=[2,5,6,7][engine.rank])
 report['actual_environment']={k:v for k,v in sorted(os.environ.items()) if k.startswith(('VLLM_ASCEND_','SPECSLO_','HCCL_','ASCEND_RT_'))
  or k in ('TASK_QUEUE_ENABLE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTORCH_NPU_ALLOC_CONF','RANK','WORLD_SIZE','LOCAL_RANK')}
 report['actual_logical_device']=str(engine.device)
 report['actual_target_leader_rank']=int(engine.topology.target_leader_rank)
 engine._v146_partial=report
 def construction():
  config=validate_full_configuration(engine)
  require(len(engine.model.layers)==(28 if engine.is_draft else 64),'actual model layer count')
  require(os.environ.get('SPECSLO_VISIBLE_DEVICES')=='2,5,6,7' and os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='0,1,2,3'
   and os.environ.get('LOCAL_RANK')==str(engine.rank) and engine.local_rank==engine.rank
   and engine.topology.target_leader_rank==1,'actual physical visibility/local-rank/target-leader mapping')
  return dict(config,layers=len(engine.model.layers),initial_native=native(engine),
   existing_inventory=inventory_record(engine),same_full_process=True)
 report['construction']=s('full-process-construction',construction)
 from vllm_ascend.spec_decode.pearl.mc2 import snapshot_mc2_dispatch_counters
 dispatch_start=s('MC2-before',snapshot_mc2_dispatch_counters)
 expected={}
 # Shared fixed loops: no branch/label depends on a role-private return value.
 for pass_index in range(3):
  zero=pass_index>0
  for case in cases():
   prefix=f'pass{pass_index}/case{case["index"]}'
   bank=build_bank(engine,case,stage=lambda name,fn:s(prefix+'/fresh-dual-bank/'+name,fn))
   before=s(prefix+'/Native-before',lambda:native(engine));state=None;records=[]
   for cycle in range(MAX_CYCLES):
    label=f'{prefix}/cycle{cycle}'
    stage=lambda name,fn: s(label+'/'+name,fn)
    work=generate(engine,native_module,bank,case,(pass_index*2+case['index'])*5+cycle,stage)
    def proposals():
     nonlocal state
     if state is None:state=setup_normal_cycle(native_module,work['source'],work['plans'],work['rows'],work['confidences'],compiled_protocol=compiled)
     attach_normal_cycle_proposals(state,work['source'],work['plans'],work['rows'],work['confidences'],cycle_id=cycle)
     work['work_plan'].require_published_tickets(state['tickets'])
    stage('original-proposals-and-typed-publication',proposals)
    candidate=qualify(engine,case,bank,work,stage=stage)
    oracle=reference(engine,case,bank,work,candidate,stage=stage)
    commit=run_original_tree_cycle(engine,native_module,work['source'],work['plans'],work['rows'],
     candidate['output'] if target else None,state,cycle_id=cycle,target_home=cycle%2,stage_all4=stage)
    own=stage('independent-own-KV-compaction-and-history',lambda:commit_reference(engine,bank,work,oracle,commit))
    row=dict(cycle=cycle,source=work['source'],selected=work['selected'],rows=work['rows'],confidences=work['confidences'],
     work_plan_id=work['work_plan'].plan_id,selected_normal_candidates=work['work_plan'].allocated_selected_candidates,
     draft=work['draft_proof'],target=None if not target else candidate['report'],reference=None if not target else oracle['report'],
     commit=commit,independent_commit=own,controlled_typed_fixture_grants_not_policy_allocation=True)
    records.append(row)
    # Checkpoint precedes any post-trace auditor. A later failure cannot erase it.
    stage('retain-completed-cycle',lambda:save(directory,f'rank{engine.rank}-pass{pass_index}-case{case["index"]}-cycle{cycle}.json.gz',row))
    # Release device-only snapshots before the next cycle, preserving actual banks.
    del candidate,oracle,work
   next_stage=lambda name,fn:s(prefix+'/next-draft/'+name,fn)
   next_work=generate(engine,native_module,bank,case,(pass_index*2+case['index'])*5+4,next_stage)
   next_record=dict(source=next_work['source'],selected=next_work['selected'],rows=next_work['rows'],confidences=next_work['confidences'],draft=next_work['draft_proof'],work_plan_id=next_work['work_plan'].plan_id)
   s(prefix+'/retain-next-draft',lambda:save(directory,f'rank{engine.rank}-pass{pass_index}-case{case["index"]}-next-draft.json.gz',next_record))
   del next_work
   def trace_finish():
    coverage=repair_coverage(records,next_record) if engine.is_draft else dict.fromkeys(('natural_missing_suffix_observed',
     'natural_missing_suffix_protocol_observed','final_next_draft_natural_missing_suffix_observed'))
    value=dict(pass_index=pass_index,case=case['index'],name=case['name'],budgets=case['budgets'],M=case['M'],S=case['S'],
     complete_protocol_cycles=4,zero_replay=zero,original_verdict_commit_next_draft=True,lease=bank.evidence(),
     Native=delta(before,native(engine),zero=zero),fingerprints=[fingerprint(row) for row in records],
     next_draft={k:next_record[k] for k in ('source','selected','rows','confidences')},
     **coverage,
     source_root_is_fixture=True)
    actual=dict(commits=value['fingerprints'],next_draft=value['next_draft'])
    if zero:require(actual==expected[case['index']],'fresh-history fixed-trace protocol zero must reproduce every emitted token/path/grant/next-draft')
    else:expected[case['index']]=actual
    return value
   report['traces'].append(s(prefix+'/complete-trace',trace_finish))
   s(prefix+'/release-dual-bank',engine._release_cache)
 report['inventory_after_local_protocol']=s('observed-full-process-inventory',lambda:inventory_record(engine))
 report['MC2_dispatch']=s('MC2-after',snapshot_mc2_dispatch_counters)
 def mc2_gate():
  d={k:report['MC2_dispatch'][k]-dispatch_start[k] for k in dispatch_start}
  require(all(v>=0 for v in d.values()) and d['fallback']==d['exception']==0 and (engine.is_draft or d['native_epilogue_success']>0),'original MC2 own-process actual dispatch')
  return dict(before=dispatch_start,after=report['MC2_dispatch'],delta=d)
 report['MC2_dispatch']=s('MC2-positive-original-route',mc2_gate)
 report['sealed']=bool(engine.graph_runner.graph_cache_sealed)
 require(not report['sealed'],'component must not seal the full production graph cache')
 report['final_Native']=s('final-Native',lambda:native(engine))
 report['complete_protocol_zero_traces']=2
 report['exploration_cycles_per_vector']=4;report['fixed_replay_cycles_per_vector']=8
 report['natural_missing_suffix_observed']=any(t['natural_missing_suffix_observed'] for t in report['traces']) if engine.is_draft else None
 report['missing_suffix_coverage_missing']=None # Shared mandatory protocol coverage is set by the original all4 join below.
 def coverage():
  return dict(rank=engine.rank,role='draft' if engine.is_draft else 'target',passed_subset=True,
   complete_protocol_zero_traces=2,natural_missing_by_vector={c['index']:any(t['case']==c['index'] and t['natural_missing_suffix_protocol_observed'] for t in report['traces']) for c in cases()} if engine.is_draft else None,
   target_all_physical_sameM_checked=target)
 votes=stage4(engine,events,'all4-component-required-coverage',coverage,share=True)
 records=[v['evidence'] for v in votes];missing,qualified=coverage_decision(records)
 report['all4_required_coverage']=records;report['missing_suffix_uncovered_vectors']=missing
 report['missing_suffix_coverage_missing']=bool(missing)
 report['passed_subset']=True
 report['component_numerical_qualified']=qualified
 report['component_scope_only']=True;report['adopted']=False;report['completed']=True
 report['boundary_events']=events;report['cache_released']=engine.cache_allocation is None
 return report
