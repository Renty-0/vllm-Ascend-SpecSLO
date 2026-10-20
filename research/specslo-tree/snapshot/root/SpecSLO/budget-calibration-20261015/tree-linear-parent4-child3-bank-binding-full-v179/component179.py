"""Current P3 representatives plus true P4 numeric/protocol proof before full32."""
import json,gzip
from pathlib import Path
from base179 import require,B,functions

def audit_saved(directory):
 from audit179 import audit_reports,numeric_raw
 from raw179 import representative_raw
 reports=[json.loads(gzip.decompress((Path(directory)/f'rank{i}-component.json.gz').read_bytes())) for i in range(4)]
 require([r['rank'] for r in reports]==list(range(4)),'actual four component reports')
 p3=[representative_raw(v) for v in reports[0]['parent3_representatives']]
 require([v['name'] for v in p3]==['N1-E1-R1','N1-E3-R1-interior-padding','N32-E32-R32'],'actual original P3 current full527 three representatives')
 natural=audit_reports([r['natural'] for r in reports])
 bounds=[numeric_raw(v['numerical_proof']) for v in reports[0]['parent4_math_boundaries']]
 require([v['R'] for v in bounds]==[0,16],'new actual P4 R0 and R16 allphysical mathematical boundaries')
 for report in reports:
  require(report['Native_flags_assigned'] is False and report['full527_configuration']['storage']==527,'current full527 instance without flags assigned')
  require(report['natural']['result']['lease_error_status']['common']['all4_common_error_seen'] is True,'actual current lease failure reaches common all4 status')
 qualified=natural['positive_count']>0 and natural['own_parent4_sameM_qualified'] is True
 return dict(schema='V179.current-full527-P3-P4-component.v1',component_numerical_qualified=qualified,
  parent3_representatives=p3,parent4_natural=natural,parent4_math_boundaries=bounds,
  parent4_positive_required=True,actual_lease_common_error=True,old_21scope26phase_executed=False,
  finite_representative_evidence=True,per_call_structural_domain_checked=True,all_token_or_tuple_AR_qualified=False,
  hidden_KV_recomputed_by_HOST=False,full463_qualified=False,formal_Goodput=False)

def run(e,native,stage,policy,output_directory):
 from full_process_v146 import capture_fixture_state,finish_fixture_state,reset_fixture_gain
 import actor179,draft179,pure179
 from scope179 import Case
 require(e.config.max_num_seqs==64 and e.config.max_num_queued_seqs==463 and e._cache_storage_sequence_capacity==527
  and e.prefix_cache.num_blocks==544 and not e.graph_runner.graph_cache_sealed,'same original full527 current constructor')
 folder=Path(output_directory)/'new-shared4-parent4-own-full527'
 stage('P4/fresh-folder',lambda:folder.mkdir(parents=True,exist_ok=False) if e.rank==e.topology.draft_leader_rank else None)
 state=stage('P4/original-fixture-state',lambda:capture_fixture_state(e));stage('P4/reset-fixture-gain',lambda:reset_fixture_gain(policy))
 p3=pure179.run_cases(e,native,lambda n,f:stage('unchanged-P3-current-representatives/'+n,f))
 boundaries=[]
 for R in (0,16):
  value=draft179.prepare(e,Case(repair_count=R),lambda n,f:stage('P4-math-R'+str(R)+'/'+n,f))
  boundaries.append(dict(case=Case(repair_count=R).__dict__,numerical_proof=value['numerical_proof'],generation_observation=value['generation_observation']))
  stage('P4-math-R'+str(R)+'/release',e._release_cache)
 result=actor179.run_actor(e,native,lambda n,f:stage('P4-natural/'+n,f))
 natural=dict(rank=e.rank,result=result,cache_released=e.cache_allocation is None and e.cache_block_tables is None,Native_flags_assigned=False,formal_Goodput=False)
 report=dict(rank=e.rank,parent3_representatives=p3,parent4_math_boundaries=boundaries,natural=natural,
  full527_configuration=dict(storage=527,queued=463,maxseq=64,pages=544),Native_flags_assigned=False,formal_Goodput=False)
 def save():
  raw=json.dumps(report,allow_nan=False,separators=(',',':')).encode()
  with (folder/f'rank{e.rank}-component.json.gz').open('xb') as f:f.write(gzip.compress(raw,mtime=0))
 stage('P4/retain-actual-component-raw',save)
 audit=stage('P4/reopen-actual-current-producer-and-reexecute-verifier',lambda:audit_saved(folder))
 stage('P4/restore-original-fixture-state',lambda:finish_fixture_state(e,state));stage('P4/reset-production-gain',lambda:reset_fixture_gain(policy))
 domain=stage('P4/current-original-Native-domain',lambda:native_domain(e))
 proof=dict(current_Native_domain=domain,directory=str(folder),audit=audit,current_engine_id=id(e),actual_full527_new_usage=True,no_old_26scope_repeated=True,
  full463_qualified=False,formal_Goodput=False)
 e._v179_actual_component=proof
 def verify(actual,value):
  return actual is e and value is proof and getattr(e,'_v179_actual_component',None) is proof and proof['audit']['component_numerical_qualified'] is True
 if audit['component_numerical_qualified']:
  from live179 import activate
  proof['activation']=stage('P4/current-own-proof-opens-bounded-production',lambda:activate(e,proof,verify))
  e._v179_qualified_component=proof
 return proof

functions(B/'tree-live-shared4-union-full-v165/component165.py',globals(),names=('native_domain',))
