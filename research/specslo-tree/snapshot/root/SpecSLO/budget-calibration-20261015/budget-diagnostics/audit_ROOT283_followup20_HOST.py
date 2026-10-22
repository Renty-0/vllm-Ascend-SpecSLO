"""Reopen actual retained followup controls; no performance claim."""
import hashlib,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-genuine-branch-followup-component-v283';T='genuine-branch-followup-root-v283-reject0-accept1to4-20-ownprefix-v1'
O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';A=O/'actual'/T
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 r=read(O/f'receipt-{T}.json');pins=read(P/'SOURCE283_RECEIPT.json')
 assert r['status']=='complete'and r['errors']==[]and r['owned_remaining']==[]and r['sources_unchanged']is True and r['devices_after']['all_idle']is True
 for k in('pre_verifier','post_verifier'):
  v=r[k];assert v['exit_code']==0 and sha(v['path'])==v['sha256'];v=v['actual_result']
  assert v['status']=='pass'and v['errors']==[]and(v['checked_frozen_files'],v['checked_external_files'],v['checked_packages'])==(10496,15,212)
 assert sha(P/'FREEZE283.json')==pins['freeze_sha256']and sha(P/'SOURCE283.tar.gz')==pins['archive_sha256']
 files=read(P/'FREEZE283.json')['files']
 for p,h in files.items():assert sha(p)==h,p
 inventory={**files,str(P/'FREEZE283.json'):pins['freeze_sha256']}
 with tarfile.open(P/'SOURCE283.tar.gz','r:gz')as ar:
  assert len(ar.getmembers())==len(inventory)==pins['members']
  for m in ar:assert m.isfile()and hashlib.sha256(ar.extractfile(m).read()).hexdigest()==inventory['/'+m.name]
 report=read(A/'rank0-numeric.json');cases=report['cases']
 assert len(cases)==report['case_count']==20 and [(c['N'],c['accepted_alternate_tokens'])for c in cases]==[(n,k)for n in(1,4,8,12)for k in range(5)]
 assert len(list(A.glob('rank0-branch-followup-*.json')))==20
 for index,c in enumerate(cases):
  assert read(A/f'rank0-branch-followup-{index:03}.json')==c
  assert all(c[k]is True for k in('teacher_forced_commit_control','original_packed_root_hidden_checked','original_next_primary_exacttop2_and_all56KV','next_genuine_alternate_bitwise_hidden_and56KV','canonical_prefix_KV_ledger_tables_protected','neural_model_executed'))
  assert c['actual_target_verdict_sample']is False and c['empirical_g_observation']is False and c['Goodput']is None
  assert c['actual_root_seed_used']is(c['accepted_alternate_tokens']>0)
  assert c['new_prefixes']==[p+c['accepted_alternate_tokens']+1 for p in c['old_prefixes']]
  assert c['repair_start_positions']==[p+1 for p in c['old_prefixes']]
 raw=report['raw_copy'];assert raw['width']==1024 and raw['all65536_payloads']is True and raw['actual_graph_replays']==len(raw['rows'])==6
 assert all(x['whole_KV_raw16_equal']is True for x in raw['rows'])
 for rank in range(4):assert read(A/f'rank{rank}-numeric.json')['model_dtype']=='torch.bfloat16'
 assert read(A/'worker-exit.json')==dict(closed=True,all_exited=True)
 result=dict(status='PASS_independent_genuine_branch_followup20_and_raw_copy_audit',tag=T,receipt=str(O/f'receipt-{T}.json'),receipt_sha256=sha(O/f'receipt-{T}.json'),frozen_sources_and_archive_verified=True,original_environment_pre_post_verified=True,owned_cleanup_complete=True,all4_idle_at_terminal=True,actual_neural_controls=20,all56KV_hidden_exact_top2_and_prefix_protection=True,raw65536payloads6replays=True,teacher_forced_controls_only=True,actual_target_g_measured=False,whole463_AR=False,Goodput=None,adopted=False)
 out=D/'ROOT283_INDEPENDENT_FOLLOWUP20_AND_RAW_COPY_AUDIT.json'
 with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
