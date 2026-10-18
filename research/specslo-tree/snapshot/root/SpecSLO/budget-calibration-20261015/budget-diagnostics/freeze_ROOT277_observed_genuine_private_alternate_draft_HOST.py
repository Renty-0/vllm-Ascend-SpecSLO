"""Freeze408-case alternate-chain numeric diagnostic, never serving scope."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-genuine-alternate-draft-observed-component-v277';H=B/'tree-private-alternate-draft-tail-host-v271'
OLD=B/'tree-genuine-alternate-draft-component-v273/FREEZE273.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not(P/'FREEZE277.json').exists()and not(P/'SOURCE277.tar.gz').exists()
 assert sha(OLD)=='8db5b8bb105087971681d1cbd0cb21925545af0765279bdb97a85b6229c1708b'
 host=json.loads((P/'HOST277_RESULT.json').read_text())
 assert host['symbolic_cases']==680 and host['original_PA_metadata_cases']==75
 assert host['actual_NN_cases_required']==45
 observer=json.loads((P/'HOST_OBSERVER277_RESULT.json').read_text());assert observer['case_count']==12 and observer['observer_installed_before_any_real_constructor']is True
 review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   schema='ROOT277.genuine-private-alternate-draft-review.v1',
   mechanism='Unchanged actual primary-four generation; take its real rootsecond token and evaluate1..3 genuine additional PA steps on private copied partial-tail tables',
   model='Original BF16 Qwen3-0.6B TP1/native3.11/CANN9.0; original primary top2 and original greedy/top2, unchanged32B TP3 idle in synchronized role protocol',
   genuine_cases='N1/2/4/8/12 times1/2/3 alternate steps times original/changed/restored roots;45 cases',
   independent_reference='Fresh24 independently prefilled owners,12 candidate and12 reference; no candidate prefix/hidden/KV initializes the reference',
   gates='Exact top2 encoding, original matcher and bitwise alternate allphysical hidden/all56 KV, independently evaluated primary allhidden/56KV, all24 prefixes protected after everycase',
   private_copy='Retained263 exact rawbyte kernel specialized width1024; separate65536payload/6graphreplay qualification before using persistent28layer copy graph',
   controls='Original changed/restored input gates with original entry flags; private page leases/root validation; canonical primary KV/tables/ledger protected',
   HOST_cases680=680,HOST_metadata_cases75=75,HOST_observer_regression_cases12=12,
   original_observer_installed_before_constructor=True,rawbit_checkpoint_retained_before_neural=True,
   original_startup_inventory_not_run=True,original_baseline_pre_post10496_15_212_verifiers_required=True,
   fresh_genuine_draft_qualification_required=True,empirical_g_not_measured=True,
   no_serving_generation=True,neural_qualified=False,whole_workload_AR_qualified=False,Goodput=None,adopted=False)
 (P/'REVIEW277.json').write_text(json.dumps(review,indent=2)+'\n')
 files=dict(json.loads(OLD.read_text())['files']);files[str(OLD)]=sha(OLD)
 extra=[*P.glob('*'),*H.glob('*'),Path(__file__),
        D/'ROOT266_COMPLETE_RESULT_FOR_NEXT_OPTIMIZATION.json',D/'ROOT266_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        D/'ROOT267_INDEPENDENT_SYNTHETIC_LIFETIME_KV_AUDIT.json',
        D/'ROOT270_PARTIAL408_NUMERIC_PASS_AND_COST_HARNESS_FAILURE.json',D/'ROOT273_MISSING_ORIGINAL_OBSERVER_HARNESS_FAILURE.json',D/'ROOT_20261018_1719_ACTIVE272_RETAIN270_FAILURE_HOST271_HANDOFF.json']
 for p in extra:
  if p.is_file():
   if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
   files[str(p)]=sha(p)
 for path,wanted in files.items():assert sha(path)==wanted,path
 freeze=P/'FREEZE277.json'
 with freeze.open('x')as f:json.dump(dict(schema='ROOT277.frozen-bounded-target-node-diagnostic.v1',UTC=review['UTC'],files=files,NPU_used=False,production_scope=False,whole_workload_AR_qualified=False),f,indent=2);f.write('\n')
 members={**files,str(freeze):sha(freeze)}
 with tarfile.open(P/'SOURCE277.tar.gz','x:gz')as archive:
  for path,wanted in sorted(members.items()):
   data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
   info=tarfile.TarInfo(path.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
   archive.addfile(info,io.BytesIO(data))
 with tarfile.open(P/'SOURCE277.tar.gz','r:gz')as archive:
  assert len(archive.getmembers())==len(members)
  for member in archive:assert member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name]
 result=dict(status='HOST_private_genuine_draft_sources_frozen',freeze_sha256=sha(freeze),archive_sha256=sha(P/'SOURCE277.tar.gz'),members=len(members),NPU_used=False)
 with(P/'SOURCE277_RECEIPT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
