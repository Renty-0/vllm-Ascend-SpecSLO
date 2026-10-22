"""Audit retained numerical evidence from the interrupted full284; no GP."""
import hashlib,json,gzip,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-dynamic-root-alternate-paired-prefix-grid-full-v284';O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';T='dynamic-root-alternate-root-v284-inprocess-stack-full463-reproducer-v1';A=O/'actual'/T

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=read(O/f'receipt-{T}.json');assert r['status']=='failed'and r['owned_remaining']==[]and r['sources_unchanged']is True and r['devices_after']['all_idle']is True
 assert len(r['errors'])==1 and 'ROOT284 supervisor received signal 15'in r['errors'][0]
 for k in('pre_verifier','post_verifier'):
  v=r[k];assert v['exit_code']==0 and sha(v['path'])==v['sha256'];v=v['actual_result'];assert v['status']=='pass'and v['errors']==[]and(v['checked_frozen_files'],v['checked_external_files'],v['checked_packages'])==(10496,15,212)
 pins=read(P/'SOURCE284_RECEIPT.json');assert sha(P/'FREEZE284.json')==pins['freeze_sha256']and sha(P/'SOURCE284.tar.gz')==pins['archive_sha256']
 files=read(P/'FREEZE284.json')['files']
 for p,h in files.items():assert sha(p)==h,p
 inv={**files,str(P/'FREEZE284.json'):pins['freeze_sha256']}
 with tarfile.open(P/'SOURCE284.tar.gz','r:gz')as ar:
  assert len(ar.getmembers())==len(inv)==pins['members']
  for m in ar:assert m.isfile()and hashlib.sha256(ar.extractfile(m).read()).hexdigest()==inv['/'+m.name]
 for rank in(1,2,3):
  proof=read(A/'canonical-independent-node-math'/f'rank{rank}-proof.json');rows=proof['math']['cases'];assert proof['cases']==len(rows)==408
  raw=list((A/'canonical-independent-node-math').glob(f'rank{rank}-canonical-case[0-9]*.json'));assert len(raw)==408
  actual={read(f)['case']['name']:read(f)['observation']for f in raw}
  for row in rows:
   assert actual[row['case']['name']]==row
   assert all(row[k]is True for k in('exact_greedy','hidden_bitwise_equal','KV_bitwise_equal','all128_KV_original_matcher_passed','graph_eager_KV_bitwise_equal'))
  assert proof['math']['all_independent_prefixes_unchanged']is True
 draft=read(A/'rank0-genuine-draft-proof.json')
 for key,folder,pattern in [('primary_alternate45','independent-genuine-draft-math','draft-case-*.json'),('postbranch20','independent-postbranch-draft-math','rank0-branch-followup-*.json')]:
  rows=draft[key]['cases'];paths=sorted((A/folder).glob(pattern));assert len(rows)==len(paths)==(45 if key=='primary_alternate45'else 20)
  assert [read(f)for f in paths]==rows
  fields=('alternate_allhidden_bitwise','alternate_all56KV_bitwise','exact_alternate_top2')if key=='primary_alternate45'else('original_next_primary_exacttop2_and_all56KV','next_genuine_alternate_bitwise_hidden_and56KV','canonical_prefix_KV_ledger_tables_protected')
  assert all(all(row[k]is True for k in fields)for row in rows)
 for rank in range(4):
  p=next((A/'generation-progress').glob(f'rank{rank}-*.json'));v=read(p);assert v['generation']==5 and v['status']=='running'and v['sealed']is True
  p=A/'causal80-seven-real-N-current-process'/f'rank{rank}-component.json.gz';v=json.loads(gzip.decompress(p.read_bytes()));assert v['completed']is True and v['errors']==[]
 assert not list((A/'windows/window48/full463').iterdir())and not(A/'completed.json').exists()
 result=dict(status='PASS_independent_partial284_numerical_evidence_and_owned_cleanup_audit',full_run_status='stalled generation5; guard SIGTERM cleanup completed',receipt_sha256=sha(O/f'receipt-{T}.json'),fresh_target_cases_per_rank=408,target_ranks=[1,2,3],fresh_genuine_draft_cases=45,fresh_postbranch_controls=20,all_retained_cases_reopened=True,source_and_archive_verified=True,original_environment_prepost_verified=True,owned_cleanup_complete=True,all4_idle_at_terminal=True,full463_completed=False,whole463_AR=False,empirical_full_g=False,Goodput=None,adopted=False,goal_achieved=False)
 out=D/'ROOT284_INDEPENDENT_PARTIAL_NUMERIC_AND_STALL_AUDIT.json'
 with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
