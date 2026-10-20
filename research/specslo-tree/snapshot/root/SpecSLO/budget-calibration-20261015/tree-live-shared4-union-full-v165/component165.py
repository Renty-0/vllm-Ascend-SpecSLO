"""Concrete own-process producer and activation callback for new live usage."""
import gzip,json
from pathlib import Path
from base165 import require

REPRESENTATIVE_SHARED=frozenset({(1,1),(1,3),(32,32)})
REPRESENTATIVE_UNION=frozenset({(1,1),(3,1),(32,1),(32,32)})

def qualifying_capability():
 from scope159 import Capability
 return Capability(REPRESENTATIVE_SHARED,REPRESENTATIVE_UNION,mode='qualifying')

def production_capability():
 from scope159 import Capability
 # Finite representative math + every original Native family gate, not a
 # device enumeration over every Btuple/token. Per-call structural scope and
 # unchanged original typed admission remain mandatory in the installed code.
 return Capability(frozenset((n,e) for n in range(1,33) for e in range(1,33)),
  frozenset((e,r) for e in range(1,33) for r in range(1,33)),mode='production')

def save(path,value):
 raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
 with Path(path).open('xb') as f:f.write(gzip.compress(raw,mtime=0))

def configuration(e):
 require(e.config.max_num_seqs==64 and e.config.max_num_queued_seqs==463
  and e._cache_storage_sequence_capacity==527 and e.prefix_cache.num_blocks==544
  and tuple(e._tree_graph_scratch_sequence_ids)==tuple(range(463,527))
  and tuple(e._mixed_target_scratch_sequence_ids)==() and not e.graph_runner.graph_cache_sealed,
  'actual full64/463/527/544 constructor, no mixed graph and before seal')
 require(e.config.spec_rhythm_draft_token_budget==192 and e.config.spec_rhythm_max_eager_tokens==3
  and e.config.spec_rhythm_eager_reserve_tokens==32,'actual selected candidate controls')
 return dict(maxseq=64,queued=463,storage=527,pages=544,target_C=96,target_Q=128,target_N=32,
  normal_C=96,eager_C=96,draft_C=192,old_mixed_21scope26phase=False,Native_flags_assigned=False)

def native_domain(e):
 """Current authoritative Native entries, not retained receipts or counts."""
 from finite_inventory_plan import draft_keys
 if not e.is_draft:return dict(role='target',draft_family_gate_not_applicable=True)
 runner=e.graph_runner;families=draft_keys();rows=[]
 for key in families['chain']:
  entry=runner.draft_entries.get(key)
  require(entry is not None and entry.runtime_validated is True and entry.validated_real_row_count>=key[1]
   and key not in runner.disabled_entry_keys,'actual current real-row-complete original draft family')
  rows.append(dict(family='draft_entries',key=repr(key),runtime_validated=True,validated_real_row_count=entry.validated_real_row_count))
 for key in (*families['hidden_PA'],*families['frontier_PA']):
  entry=runner.target_entries.get(key)
  require(entry is not None and entry.runtime_validated is True and key not in runner.disabled_entry_keys,
   'actual current original PA family naturally closed')
  rows.append(dict(family='target_entries',key=repr(key),runtime_validated=True))
 return dict(role='draft',entries=rows,Native_flags_assigned=False)

def audit_saved(directory):
 """Reopen actual per-rank producer bytes before activation in this process."""
 from audit154 import audit
 reports=[json.loads(gzip.decompress((Path(directory)/f'rank{i}-component.json.gz').read_bytes())) for i in range(4)]
 natural=audit([r['natural'] for r in reports])
 from audit_component165 import natural_raw,representative_raw
 raw_natural=natural_raw([r['natural'] for r in reports])
 raw_representatives=[representative_raw(item) for item in reports[0]['representatives']]
 require(natural['status']=='passed-component-evidence' and natural['positive_count']>0 and natural['negative_count']>0,
  'current actual genuine positive consume+negative invalidation+next, no inherited certificate')
 expected=['N1-E1-R1','N1-E3-R1-interior-padding','N32-E32-R32']
 for rank,r in enumerate(reports):
  require(r['rank']==rank and r['full527_configuration']['storage']==527 and r['Native_flags_assigned'] is False
   and [v['case']['name'] for v in r['representatives']]==expected,'complete actual full527 new usage report')
  probe=r['natural']['result']['lease_error_status']
  require(probe['common']['all4_common_error_seen'] is True,'actual all4 status rejection path')
  if rank==0:
   for item in r['representatives']:
    n=item['numerical'];m=item['metadata']
    require(all(n[k] is True for k in ('sameM_pass','allphysical_union_hidden','allphysical_fourstep_hidden','full_encoded','all56_written_KV','exact_canonical8_scores_and_ledger'))
     and all(m['field_equal'].values()) and all(m['own_lease'].values()),'actual complete physical same-M and own metadata gates')
   middle=r['representatives'][1]['numerical']
   require((middle['real_E'],middle['F'],middle['real_R'],middle['U'])==(3,4,1,8),'actual interior dummy representative')
 return dict(schema='V165.own-current-full527-new-usage.v1',component_numerical_qualified=True,
  natural=natural,raw_natural=raw_natural,raw_representatives=raw_representatives,finite_new_usage_representatives=expected,actual_all4_status_rejection=True,
  fullphysical_hidden_encoded_KV_source_gates=True,Native_current_family_gate=True,
  full527_current_instance=True,per_call_structural_domain_checked=True,
  numerical_token_tuple_enumeration=False,arbitrary_AR_equivalence=False,full463_qualified=False,
  Native_flags_assigned=False,formal_Goodput=False)

def run(e,native,stage,policy,output_directory):
 from pure165 import run_cases
 import natural165
 from full_process_v146 import capture_fixture_state,finish_fixture_state,reset_fixture_gain
 folder=Path(output_directory)/'new-shared4-union-own-full527'
 stage('new-component/fresh-folder',lambda:folder.mkdir(parents=True,exist_ok=False) if e.rank==e.topology.draft_leader_rank else None)
 config=stage('new-component/full-configuration',lambda:configuration(e))
 state=stage('new-component/fixture-state',lambda:capture_fixture_state(e))
 stage('new-component/fresh-fixture-gain',lambda:reset_fixture_gain(policy))
 e._v165_partial={}
 representatives=run_cases(e,native,lambda n,f:stage('representatives/'+n,f))
 natural=natural165.run_actor(e,native,lambda n,f:stage('natural-full527/'+n,f))
 # The old audit accepts the identical actual parent/child/next schema and
 # never asserts local128 storage. Current full527 config is separately exact.
 natural_report=dict(rank=e.rank,errors=[],result=natural,cache_released=e.cache_allocation is None and e.cache_block_tables is None,
  Native_flags_assigned=False,formal_Goodput=False)
 families=stage('new-component/current-Native-family-domain',lambda:native_domain(e))
 report=dict(rank=e.rank,full527_configuration=config,representatives=representatives,natural=natural_report,
  current_Native_domain=families,Native_flags_assigned=False,formal_Goodput=False)
 stage('new-component/retain-own-producer',lambda:save(folder/f'rank{e.rank}-component.json.gz',report))
 audit=stage('new-component/audit-own-retained-four-rank-producer',lambda:audit_saved(folder))
 stage('new-component/clear-fixture-state',lambda:finish_fixture_state(e,state))
 stage('new-component/fresh-production-gain',lambda:reset_fixture_gain(policy))
 result=dict(directory=str(folder),audit=audit,current_engine_id=id(e),
  actual_full527_new_usage=True,no_old_26scope_repeated=True,full463_qualified=False,formal_Goodput=False)
 e._v165_actual_component=result
 def verify(actual,capability):
  require(actual is e and getattr(actual,'_v165_actual_component',None) is result
   and result['audit']['component_numerical_qualified'] is True,'same current engine actual issuer identity')
  require(capability==production_capability(),'exact bounded production domain derived after representative gates')
  native_domain(actual)
  return True
 from live159 import activate_after_component
 stage('new-component/activate-actual-closed-live-domain',lambda:activate_after_component(production_capability(),engine=e,verify_actual_component=verify))
 # P166's shared standalone issuer additionally checks this current object
 # identity. It never returns the old mixed S35 authorization set.
 e._v165_qualified_component=result
 return result
