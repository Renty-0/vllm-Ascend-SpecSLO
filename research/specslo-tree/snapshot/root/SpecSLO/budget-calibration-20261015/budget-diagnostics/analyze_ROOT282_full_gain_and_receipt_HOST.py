"""Independent arithmetic on completed raw outputs; never invent unsampled g."""
import argparse,gzip,hashlib,json,statistics,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';P=B/'tree-dynamic-root-alternate-paired-prefix-grid-full-v282'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(tag):
 receipt=read(O/f'receipt-{tag}.json');a=O/'actual'/tag
 assert receipt['status']=='complete'and receipt['errors']==[]and receipt['owned_remaining']==[]and receipt['sources_unchanged']is True and receipt['devices_after']['all_idle']is True
 for k in('pre_verifier','post_verifier'):
  v=receipt[k];assert v['exit_code']==0 and sha(v['path'])==v['sha256'];v=v['actual_result'];assert v['status']=='pass'and v['errors']==[]and(v['checked_frozen_files'],v['checked_external_files'],v['checked_packages'])==(10496,15,212)
 pins=read(P/'SOURCE282_RECEIPT.json');assert sha(P/'FREEZE282.json')==pins['freeze_sha256']and sha(P/'SOURCE282.tar.gz')==pins['archive_sha256']
 files=read(P/'FREEZE282.json')['files']
 for f,h in files.items():assert sha(f)==h,f
 inventory={**files,str(P/'FREEZE282.json'):pins['freeze_sha256']}
 with tarfile.open(P/'SOURCE282.tar.gz','r:gz')as ar:
  assert len(ar.getmembers())==len(inventory)==pins['members']
  for m in ar:assert m.isfile()and hashlib.sha256(ar.extractfile(m).read()).hexdigest()==inventory['/'+m.name]
 reports=[read(a/f'rank{rank}-local-diagnostic.json')for rank in range(4)]
 # Reopen raw per-case data independently of the guard's summary.
 for rank,report in enumerate(reports):
  proof=report['ROOT282_current_math'];assert proof['cases']==408 and len(proof['math']['cases'])==408
  if rank:
   rawfiles=list((a/'canonical-independent-node-math').glob(f'rank{rank}-canonical-case[0-9]*.json'));assert len(rawfiles)==408
   observations={read(f)['case']['name']:read(f)['observation']for f in rawfiles}
   for row in proof['math']['cases']:
    assert observations[row['case']['name']]==row
    assert all(row[k]is True for k in('exact_greedy','hidden_bitwise_equal','KV_bitwise_equal','all128_KV_original_matcher_passed','graph_eager_KV_bitwise_equal'))
    assert row['candidate_tokens']==row['reference_tokens']and row['candidate_KV_copied']is False
  draft=report['ROOT282_actual_geometry']['draft_numeric_proof']
  assert len(draft['primary_alternate45']['cases'])==45 and len(draft['postbranch20']['cases'])==20
  if rank==0:
   assert len(list((a/'independent-genuine-draft-math').glob('draft-case-*.json')))==45
   for i,row in enumerate(draft['primary_alternate45']['cases']):
    assert read(a/'independent-genuine-draft-math'/f'draft-case-{i:03d}.json')==row
    assert all(row[k]is True for k in('exact_primary_top2','primary_allhidden_and56KV','alternate_allhidden_bitwise','alternate_all56KV_bitwise','exact_alternate_top2','changed_restored_exact','canonical_mainKV_ledger_and_tables_unchanged','original_Native_runtime_validated','all_prefixes_checked_by_caller'))
   raw=draft['primary_alternate45']['raw_copy'];assert raw['all65536_payloads']is True and raw['actual_graph_replays']==6
   assert len(list((a/'independent-postbranch-draft-math').glob('rank0-branch-followup-*.json')))==20
   for i,row in enumerate(draft['postbranch20']['cases']):
    assert read(a/'independent-postbranch-draft-math'/f'rank0-branch-followup-{i:03d}.json')==row
    assert all(row[k]is True for k in('original_packed_root_hidden_checked','original_next_primary_exacttop2_and_all56KV','next_genuine_alternate_bitwise_hidden_and56KV','canonical_prefix_KV_ledger_tables_protected','neural_model_executed'))
 full=[]
 for env in reports[0]['windows']['traces']:
  r=json.loads(gzip.decompress(Path(env['original_trace']['artifact']['path']).read_bytes()))
  if r['scope_count']==463:full.append(r)
 assert len(full)==1;trace=full[0];generation=str(trace['source_generation'])
 geometries=[r['ROOT282_actual_geometry']for r in reports]
 paired=geometries[0]['actual_paired_gain'];assert paired['generation']==int(generation)
 hist=geometries[0]['actual_request_gain_observations'][generation]
 for other in geometries[1:]:
  assert other['actual_paired_gain']==paired and other['actual_request_gain_observations'][generation]==hist
 aggregate={c:{}for c in('tight','normal','loose')};per_request=[]
 for owner,request in hist.items():
  cls=request['slo_class']
  for b,row in request['by_budget'].items():
   n=int(b);samples=row['samples'];actual=row['delivered_sum'];side=row['accepted_side_nodes'];shadow=row['shadow_B4_sum']
   assert samples>0 and row['shadow_B4_available']is(n>=4)
   assert sum(row['delivered_histogram'])==sum(row['side_histogram'])==samples
   assert sum(i*v for i,v in enumerate(row['delivered_histogram']))==actual
   assert sum(i*v for i,v in enumerate(row['side_histogram']))==side
   assert 0<=side<=samples*max(0,n-4)
   assert actual-shadow==side if n>=4 else shadow==side==0
   dest=aggregate[cls].setdefault(b,dict(samples=0,delivered=0,side=0,shadow=0,requests=0))
   for key,value in [('samples',samples),('delivered',actual),('side',side),('shadow',shadow),('requests',1)]:dest[key]+=value
   per_request.append(dict(owner=int(owner),slo_class=cls,B=n,samples=samples,mean_delivered=actual/samples,mean_side=side/samples))
 for cls,bybudget in aggregate.items():
  paired_class=paired['by_generation'][generation]['by_class'][cls]
  for b,r in bybudget.items():
   n=int(b);r['mean_delivered']=r['delivered']/r['samples'];r['mean_side']=r['side']/r['samples'];r['same_prefix_shadow_B4']=(r['shadow']/r['samples']if n>=4 else None)
   if n>=5:
    observed=paired_class['by_budget'][b]
    assert (observed['samples'],observed['sum_actual'],observed['sum_shadow4'],observed['side_nodes'])==(r['samples'],r['delivered'],r['shadow'],r['side'])
  assert paired_class['calls']==sum(v['samples']for k,v in bybudget.items()if int(k)>=5)
  assert paired_class['primary_shadow_calls']==bybudget.get('4',{}).get('samples',0)
 actual=receipt['actual_full_audit'];assert actual['passed']is True and actual['whole_workload_AR_qualified']is True
 result=dict(status='PASS_independent_ROOT282_full_gain_histogram_and_source_audit',tag=tag,source_freeze_sha256=pins['freeze_sha256'],receipt_sha256=sha(O/f'receipt-{tag}.json'),actual_full_audit=actual,generation=int(generation),aggregate_actual_g_by_class_and_observed_B=aggregate,per_request_means=per_request,all4_gain_observations_identical=True,same_prefix_identities_and_histograms_exact=True,unsampled_B_values_not_estimated=True,warning='Online selected-budget samples are adaptive and request-dependent; not a randomized causal comparison of arbitrary B across requests',goal_met_candidate_only=False,final_three_blocks_still_required=True)
 out=D/f'ROOT282_INDEPENDENT_FULL_GAIN_ANALYSIS-{tag}.json'
 with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],path=str(out),aggregate=aggregate,actual_full_audit=actual)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);analyze(p.parse_args().tag)
