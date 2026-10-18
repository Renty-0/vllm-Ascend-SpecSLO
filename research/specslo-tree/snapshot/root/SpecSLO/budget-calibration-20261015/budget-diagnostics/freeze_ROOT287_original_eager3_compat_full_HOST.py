"""Freeze explicit full candidate; prior device probes are inputs, not scope."""
import datetime,hashlib,io,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-dynamic-root-alternate-eager-compat-full-v287'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert not(P/'FREEZE287.json').exists()and not(P/'SOURCE287.tar.gz').exists()
 for f in ('HOST_FULL287_PROTOCOL_RESULT.json','HOST_COST287_RESULT.json','HOST_BUDGET287_RESULT.json','HOST_PLANS287_RESULT.json','HOST_PRIORITY287_RESULT.json','HOST_ADMISSION287_RESULT.json','HOST_LIVE287_POST_SNAPSHOT_RESULT.json','HOST_FAULT287_RESULT.json','HOST_SNAPSHOT287_RESULT.json','HOST_ALGORITHM287_EQUIVALENCE.json','HOST_EAGER287_RESULT.json'):
  assert read(P/f)['status'].startswith('HOST_'),f
 assert read(P/'HOST_FULL287_PROTOCOL_RESULT.json')['actual_observer_cases']==408
 assert read(P/'HOST_LIVE287_POST_SNAPSHOT_RESULT.json')['actual_policy_cases']==132
 dependencies=[('tree-dynamic-root-alternate-paired-prefix-grid-full-v284',284),('tree-dynamic-root-alternate-paired-prefix-grid-full-v282',282),('tree-restored-cost-repair-byte-KV-full-v266',266),('tree-prefix-grid-target-replay-node-cost-v280',280),('tree-genuine-branch-followup-component-v283',283),('tree-paired-alternate-gain-host-v274',274),('tree-root-alternate-live-policy-host-v276',276)]
 files={}
 for name,epoch in dependencies:
  freeze=B/name/f'FREEZE{epoch}.json'
  for path,wanted in read(freeze)['files'].items():
   if path in files:assert files[path]==wanted,path
   files[path]=wanted
  files[str(freeze)]=sha(freeze)
 for f in ('ROOT280_INDEPENDENT_TARGET408_AND_PAIRED_COST_AUDIT.json','ROOT283_INDEPENDENT_FOLLOWUP20_AND_RAW_COPY_AUDIT.json','ROOT277_INDEPENDENT_GENUINE_DRAFT45_AND_RAW_COPY_AUDIT.json'):
  assert read(D/f)['status'].startswith('PASS_'),f
  files[str(D/f)]=sha(D/f)
 review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema='ROOT287.genuine-root-alternate-paired-dynamic-B-full.v1',
  actual_LIVE_B_domain=[1,8],normal_C=96,future_C=96,global_C=192,target_primary_plus_side_groups=24,physical_M=120,physical_S=24,KV_extent=2560,active_admission_cap=48,
  branch_pool='primary4 plus genuine root alternate4; retained prefix B1..8; B6..8 add three actual draft NN steps',
  draft_work_accounting='selected candidates and raw NN queries are distinct; extra padded queries, calls and unselected raw decisions retained',
  gain='original primary gain and actual same-prefix counterfactual shadow plus censored alternate survival; reset each original generation',
  target_copy='exact original263 raw-byte kernel with validated prefix grid and grid in actual280 replay-first capture key',
  single_duplicate_HOST_contract_check_removed=True,one_complete_anti_alias_bounds_contract_still_before_index_H2D=True,
  fresh_current_target408_and72_own_prefixes_required=True,fresh_current_draft45_and_postbranch20_required=True,
  original_warm_twozero_seal_full463_and_independent_ownAR_required=True,original463_arrivals_SLO_BF16_sampler_unchanged=True,
  prior_numeric_scope_inherited=False,full_Goodput_measured=False,adopted=False,diagnostic_only=False,algorithm_parent284_unchanged_except_exact_original_eager3_compatibility=True,inprocess_stack_signal_registered=True,bounded_online_state_snapshots=True,performance_invalidated_if_signal_requested=True)
 (P/'INTEGRATION_REVIEW287.json').write_text(json.dumps(review,indent=2)+'\n')
 for p in P.rglob('*'):
  if p.is_file()and '__pycache__'not in p.parts:
   if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
   files[str(p)]=sha(p)
 files[str(Path(__file__))]=sha(__file__)
 files[str(D/'ROOT284_FULL_FAILURE_AND_STACK_DIAGNOSIS.json')]=sha(D/'ROOT284_FULL_FAILURE_AND_STACK_DIAGNOSIS.json')
 for path,h in files.items():assert sha(path)==h,path
 f=P/'FREEZE287.json'
 with f.open('x')as stream:json.dump(dict(schema='ROOT287.frozen-source-full-ownAR.v1',UTC=review['UTC'],files=files,prior_sources_edited=False,NPU_qualification_inherited=False,adopted=False),stream,indent=2);stream.write('\n')
 inventory={**files,str(f):sha(f)}
 with tarfile.open(P/'SOURCE287.tar.gz','x:gz')as ar:
  for path,h in sorted(inventory.items()):
   data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==h
   info=tarfile.TarInfo(path.lstrip('/'));info.mode=0o600;info.size=len(data);info.mtime=0;ar.addfile(info,io.BytesIO(data))
 with tarfile.open(P/'SOURCE287.tar.gz','r:gz')as ar:
  assert len(ar.getmembers())==len(inventory)
  for m in ar:assert m.isfile()and hashlib.sha256(ar.extractfile(m).read()).hexdigest()==inventory['/'+m.name]
 pins=dict(freeze_sha256=sha(f),archive_sha256=sha(P/'SOURCE287.tar.gz'),members=len(inventory))
 (P/'SOURCE287_RECEIPT.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps(pins))
if __name__=='__main__':main()
