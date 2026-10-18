"""Freeze408-case alternate-chain numeric diagnostic, never serving scope."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-genuine-alternate-draft-component-v273';H=B/'tree-private-alternate-draft-tail-host-v271'
OLD=B/'tree-alternate-chain-target-node-cost-v272/FREEZE272.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not(P/'FREEZE273.json').exists()and not(P/'SOURCE273.tar.gz').exists()
 assert sha(OLD)=='ce992331e3bbae25dcc9bd3df3e385836b411a941af425a678bf1efc2aa929e3'
 host=json.loads((P/'HOST273_RESULT.json').read_text())
 assert host['symbolic_cases']==680 and host['original_PA_metadata_cases']==75
 assert host['actual_NN_cases_required']==45
 review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   schema='ROOT273.genuine-private-alternate-draft-review.v1',
   mechanism='Unchanged actual primary-four generation; take its real rootsecond token and evaluate1..3 genuine additional PA steps on private copied partial-tail tables',
   model='Original BF16 Qwen3-0.6B TP1/native3.11/CANN9.0; original primary top2 and original greedy/top2, unchanged32B TP3 idle in synchronized role protocol',
   genuine_cases='N1/2/4/8/12 times1/2/3 alternate steps times original/changed/restored roots;45 cases',
   independent_reference='Fresh24 independently prefilled owners,12 candidate and12 reference; no candidate prefix/hidden/KV initializes the reference',
   gates='Exact top2 encoding, original matcher and bitwise alternate allphysical hidden/all56 KV, independently evaluated primary allhidden/56KV, all24 prefixes protected after everycase',
   private_copy='Retained263 exact rawbyte kernel specialized width1024; separate65536payload/6graphreplay qualification before using persistent28layer copy graph',
   controls='Original changed/restored input gates with original entry flags; private page leases/root validation; canonical primary KV/tables/ledger protected',
   HOST_cases680=680,HOST_metadata_cases75=75,
   original_startup_inventory_not_run=True,original_baseline_pre_post10496_15_212_verifiers_required=True,
   fresh_genuine_draft_qualification_required=True,empirical_g_not_measured=True,
   no_serving_generation=True,neural_qualified=False,whole_workload_AR_qualified=False,Goodput=None,adopted=False)
 (P/'REVIEW273.json').write_text(json.dumps(review,indent=2)+'\n')
 files=dict(json.loads(OLD.read_text())['files']);files[str(OLD)]=sha(OLD)
 extra=[*P.glob('*'),*H.glob('*'),Path(__file__),
        D/'ROOT266_COMPLETE_RESULT_FOR_NEXT_OPTIMIZATION.json',D/'ROOT266_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        D/'ROOT267_INDEPENDENT_SYNTHETIC_LIFETIME_KV_AUDIT.json',
        D/'ROOT270_PARTIAL408_NUMERIC_PASS_AND_COST_HARNESS_FAILURE.json',D/'ROOT_20261018_1719_ACTIVE272_RETAIN270_FAILURE_HOST271_HANDOFF.json']
 for p in extra:
  if p.is_file():
   if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
   files[str(p)]=sha(p)
 for path,wanted in files.items():assert sha(path)==wanted,path
 freeze=P/'FREEZE273.json'
 with freeze.open('x')as f:json.dump(dict(schema='ROOT273.frozen-bounded-target-node-diagnostic.v1',UTC=review['UTC'],files=files,NPU_used=False,production_scope=False,whole_workload_AR_qualified=False),f,indent=2);f.write('\n')
 members={**files,str(freeze):sha(freeze)}
 with tarfile.open(P/'SOURCE273.tar.gz','x:gz')as archive:
  for path,wanted in sorted(members.items()):
   data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
   info=tarfile.TarInfo(path.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
   archive.addfile(info,io.BytesIO(data))
 with tarfile.open(P/'SOURCE273.tar.gz','r:gz')as archive:
  assert len(archive.getmembers())==len(members)
  for member in archive:assert member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name]
 result=dict(status='HOST_private_genuine_draft_sources_frozen',freeze_sha256=sha(freeze),archive_sha256=sha(P/'SOURCE273.tar.gz'),members=len(members),NPU_used=False)
 with(P/'SOURCE273_RECEIPT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
