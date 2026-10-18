"""Freeze408-case alternate-chain numeric diagnostic, never serving scope."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-alternate-chain-target-node-cost-v272';H=B/'tree-alternate-chain-target-host-v269'
OLD=B/'tree-alternate-chain-target-node-math-v270/FREEZE270.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not(P/'FREEZE272.json').exists()and not(P/'SOURCE272.tar.gz').exists()
 assert sha(OLD)=='63765ec014f721d509cf02ed5f3fba83f8ce0505af4d54379d4cecc4514af51d'
 host=json.loads((P/'HOST272_RESULT.json').read_text());assert host['cases']==408 and host['unchanged_legacy_cases']==227 and host['new_cases']==181
 assert json.loads((H/'HOST269_RESULT.json').read_text())['cases']==1252
 assert json.loads((H/'HOST_REFERENCE269_RESULT.json').read_text())['labeled_nodes']==83632
 assert json.loads((P/'DECLARED272_CASES.json').read_text())['count']==408
 regression=json.loads((P/'HOST_HARNESS272_RESULT.json').read_text())
 assert regression['actual_historical_dispatches']==231 and regression['draft_None_bank_full_control_stages']==1155
 review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   schema='ROOT272.alternate-chain-B1to8-bounded-target-numeric-review.v1',
   domain='Primary4 plus one alternatechain1..4, totalpath depth<=4, B1..8,N24,C96,primary+sidegroups<=24',
   model='Unchanged BF16 Qwen3-32B TP3 and0.6B TP1/native3.11/CANN9.0, original greedy, no quantization',
   backend='M120/S24/uniformq5/KV2560, original projection and checked263 raw byte copy1792',
   original227_case_set_retained=True,new_cases=181,actual_total_cases=408,
   independent_reference='Fresh same-instance72 own prefixes; independently teacher-forced primary and alternate paths; no candidateKV/hidden copied into references',
   target_gates='Original numerical matcher plus bitwise hidden/all128KV/greedy, graph/eager comparison, all committed prefixes protected',
   paired_cost='11 configurations, N4/8/12/16/20/23, sideowners1/max, B4..8 rotated20rounds; synchronized actual whole-target calls; stable outputs; all timings after math',
   benchmark_variants_included_in_independent_math=True,
   original_cold_dispatch_reconstruction_retained=True,draft_None_target_roots_preserved=True,
   completed_math_checkpoint_before_costs=True,
   original_startup_inventory_not_run=True,original_baseline_pre_post10496_15_212_verifiers_required=True,
   genuine_draft_alternate_generation_not_implemented=True,empirical_g_not_measured=True,
   no_serving_generation=True,neural_qualified=False,whole_workload_AR_qualified=False,Goodput=None,adopted=False)
 (P/'REVIEW272.json').write_text(json.dumps(review,indent=2)+'\n')
 files=dict(json.loads(OLD.read_text())['files']);files[str(OLD)]=sha(OLD)
 extra=[*P.glob('*'),*H.glob('*'),Path(__file__),
        D/'ROOT266_COMPLETE_RESULT_FOR_NEXT_OPTIMIZATION.json',D/'ROOT266_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        D/'ROOT267_INDEPENDENT_SYNTHETIC_LIFETIME_KV_AUDIT.json',
        D/'ROOT270_PARTIAL408_NUMERIC_PASS_AND_COST_HARNESS_FAILURE.json']
 for p in extra:
  if p.is_file():
   if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
   files[str(p)]=sha(p)
 for path,wanted in files.items():assert sha(path)==wanted,path
 freeze=P/'FREEZE272.json'
 with freeze.open('x')as f:json.dump(dict(schema='ROOT272.frozen-bounded-target-node-diagnostic.v1',UTC=review['UTC'],files=files,NPU_used=False,production_scope=False,whole_workload_AR_qualified=False),f,indent=2);f.write('\n')
 members={**files,str(freeze):sha(freeze)}
 with tarfile.open(P/'SOURCE272.tar.gz','x:gz')as archive:
  for path,wanted in sorted(members.items()):
   data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
   info=tarfile.TarInfo(path.lstrip('/'));info.size=len(data);info.mode=0o600;info.mtime=0
   archive.addfile(info,io.BytesIO(data))
 with tarfile.open(P/'SOURCE272.tar.gz','r:gz')as archive:
  assert len(archive.getmembers())==len(members)
  for member in archive:assert member.isfile()and hashlib.sha256(archive.extractfile(member).read()).hexdigest()==members['/'+member.name]
 result=dict(status='HOST408_bounded_target_math_and_cost_sources_frozen',freeze_sha256=sha(freeze),archive_sha256=sha(P/'SOURCE272.tar.gz'),members=len(members),NPU_used=False)
 with(P/'SOURCE272_RECEIPT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
