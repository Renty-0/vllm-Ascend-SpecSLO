"""Read-only terminal181 failure/base audit. No device/backend/model imports."""
import ast
import datetime
import gzip
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tarfile

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-parent3-repair-frontier-union3-active32-full-v181'
BASE=B/'tree-high-normal-cap4-native-families-v148'
TAG='parent3-union3-active32-v181-full463-v1'
A=P/'actual'/TAG
OUT=D/'V181_INDEPENDENT_TERMINAL_FAILURE_AND_B4_BASE_AUDIT.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def gz(path):return json.loads(gzip.decompress(Path(path).read_bytes()))

class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
   raise RuntimeError('Terminal HOST audit cannot load '+fullname)

def main():
 sys.meta_path.insert(0,Firewall())
 receipt=P/('receipt-'+TAG+'.json');r=read(receipt)
 assert sha(receipt)=='90671acf1a98672a351a802d467deff2cfd237aa734555d8a9414b4c72a316f2'
 assert r['status']=='failed' and r['child']['exit_code']==1 and r['owned_remaining']==[]
 assert r['sources_unchanged']is True and r['devices_after']['all_idle']is True
 assert r['elapsed_seconds']<r['limits']['child_execution_seconds']==7200
 assert len(r['errors'])==1 and 'actual child completed within7200seconds'in r['errors'][0]
 assert not any('bounded7200second actual child'in x for x in r['errors'])
 for name in ('pre_verifier','post_verifier'):
  v=r[name];actual=read(v['path']);assert sha(v['path'])==v['sha256'] and actual==v['actual_result']
  assert v['exit_code']==0 and actual['status']=='pass' and actual['errors']==[]
  assert (actual['checked_frozen_files'],actual['checked_external_files'],actual['checked_packages'])==(10496,15,212)
 exited=read(A/'worker-exit.json');assert exited==dict(closed=True,all_exited=True)
 freeze=P/'FROZEN_SOURCE181.json';f=read(freeze)
 assert sha(freeze)==r['readiness']['freeze_sha256']
 assert sha(P/'CPU_RESULT181.json')==r['readiness']['CPU_sha256']==f['CPU_sha256']
 frozen={}
 for name in ('files','external_files','execution_guard_inputs','retained_HOST_evidence','peer_reviews'):
  for path,want in f[name].items():assert sha(path)==want;frozen[path]=want
 frozen[str(P/'CPU_RESULT181.json')]=f['CPU_sha256'];frozen[str(freeze)]=sha(freeze)
 frozen[f['root_review']['path']]=f['root_review']['sha256']
 archive=r['readiness']['archive'];assert sha(archive['path'])==archive['sha256']
 with tarfile.open(archive['path'],'r:gz')as saved:
  members=saved.getmembers();assert len(members)==len(frozen)==archive['members']==649
  assert len({m.name for m in members})==len(members)
  for m in members:
   assert m.isfile() and '/'+m.name in frozen
   assert hashlib.sha256(saved.extractfile(m).read()).hexdigest()==frozen['/'+m.name]
 for name in ('audit_v146.py','scope_v146.py','full_process_v146.py'):
  assert str(BASE/name)in f['external_files']
 sys.path.insert(0,str(BASE))
 from audit_v146 import audit_reports
 base=audit_reports(A/'variable-B4-own-full-process')
 assert base['status']=='passed' and base['component_numerical_qualified']is True
 assert base['missing_suffix_uncovered_vectors']==[]
 assert base['HOST_recomputes_full_hidden_or_KV']is False
 raw={**base['raw_checkpoint_sha256'],**base['raw_report_sha256']}
 assert len(raw)==124 and len(base['raw_checkpoint_sha256'])==120
 zeros=cycles=nexts=0;inventory=[]
 for rank in range(4):
  report=gz(A/'variable-B4-own-full-process'/f'rank{rank}-component.json.gz')
  assert len(report['traces'])==6 and report['component_numerical_qualified']is True
  assert report['cache_released']is True and report['sealed']is False
  for trace in report['traces']:
   assert trace['complete_protocol_cycles']==4;cycles+=4;nexts+=1
   if trace['zero_replay']:
    assert len(trace['Native']['delta'])==21 and not any(trace['Native']['delta'].values());zeros+=21
  before=report['inventory_delta']['before'];after=report['inventory_delta']['after']
  counts=lambda value:{k:len(v)for k,v in value.items()}
  assert counts(before)==dict(entries=0,draft_entries=60,target_entries=54)if rank==0 else counts(before)==dict(entries=0,draft_entries=0,target_entries=245)
  assert counts(after)==dict(entries=0,draft_entries=60,target_entries=54)if rank==0 else counts(after)==dict(entries=0,draft_entries=0,target_entries=247)
  inventory.append(dict(rank=rank,before=counts(before),after=counts(after)))
 assert (cycles,nexts,zeros)==(96,24,336)
 label='V181/union3/representatives/N1-E1-R1/actual-allphysical-FTF-own-metadata-and-prefix'
 boundary=None;error_pins={}
 for rank in range(4):
  path=A/f'rank{rank}-last-error.json';x=read(path);error_pins[str(path)]=sha(path)
  assert x['rank']==rank and x['formal_Goodput']is False
  marker='failed '+label+': ';assert marker in x['error']
  rows=ast.literal_eval(x['error'].split(marker,1)[1].strip())
  assert [v['rank']for v in rows]==list(range(4)) and all(v['label']==label for v in rows)
  assert rows[0]['error'].startswith('RuntimeError: Expected all tensors to be on the same device')
  assert 'index is on npu:0' in rows[0]['error'] and 'other tensors on cpu'in rows[0]['error']
  assert all(v['error']is None for v in rows[1:])
  assert boundary is None or boundary==rows;boundary=rows
  assert x['partial_union']=={} and x['partial_natural']=={}
 assert not any((A/name).exists()for name in ('completed.json','prewarm','windows'))
 assert not list((A/'new-parent3-union3-own-full527').iterdir())
 startup={}
 for rank in range(4):
  root=A/'startup-progress';mc=read(root/f'MC2-preflight-rank{rank}.json')
  assert mc['status']=='passed' and type(mc['failed_checks'])is list and mc['failed_checks']==[] and mc['all4rank_contract_passed']is True
  if rank==0:
   draft=read(root/'finite-draft-rank0.json')
   assert draft['status']=='complete' and draft['explicit_keys']==86 and draft['existing_native_generic_keys_retained']==28
   startup[str(rank)]=dict(explicit_draft_keys=86,existing_native_keys=28,total_logical_original_families=114)
  else:
   target=read(root/f'finite-target-rank{rank}.json');prefill=read(root/f'prefill-rank{rank}.json')
   assert target['status']=='complete' and type(target['native_keys'])is list
   assert len(target['native_keys'])==len(set(target['native_keys']))==target['fixtures']==44
   assert prefill['status']=='complete' and prefill['completed']==prefill['total']==201
   startup[str(rank)]=dict(prefill201_complete=True,target44_complete=True)
 source_pins={str(p):sha(p)for p in (BASE/'audit_v146.py',BASE/'scope_v146.py',BASE/'full_process_v146.py',BASE/'startup_receipts_v75.py',P/'pure181.py',P/'runtime181.py',D/'v181-root-execution/guard181.py',D/'v154-root-execution/supervise_v154.py',Path(__file__),D/'ROOT_V181_ACTUAL_ORACLE_MIXED_DEVICE_INDEX_FAILURE.json')}
 original=(D/'v154-root-execution/supervise_v154.py').read_text()
 assert "row['elapsed_seconds']<MAX_SECONDS and child.returncode==0"in original
 evidence={str(p):sha(p)for p in A.rglob('*')if p.is_file()}
 record=dict(schema='V181.independent-terminal-failure-and-original-base-HOST.v1',status='passed_failure_classification_and_base_component_audit',candidate=str(P),tag=TAG,
  created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=dict(path=str(receipt),sha256=sha(receipt)),
  original_audit146=base,original_B4_current_component_reaudited=True,base_raw_files=124,base_protocol_cycles_all_ranks=96,base_natural_next_all_ranks=24,base_Native_zero_subtractions=zeros,base_native_dictionary_continuity=inventory,
  constructor_startup_retained=startup,original_source_inputs=source_pins,frozen_source_archive=dict(source_files=len(frozen),archive_sha256=archive['sha256'],members=649,all_current_and_archived_bytes_match=True),
  failure=dict(boundary=label,all4_exact_error_packets=boundary,error_files=error_pins,source_line='pure181.py:31',cause='Independent PA reference creates one NPU index and applies it to CPU context_lens; actual error and source agree. Production union numerical behavior is not disproved by this oracle failure.',new_component_completed=False,natural_positive_qualified=False,prewarm_executed=False,full463_executed=False,graph_seal_executed=False,performance_result=None),
  outer=dict(child_exit_code=1,elapsed_seconds=r['elapsed_seconds'],limit_seconds=7200,timeout_triggered=False,generic_success_assertion='elapsed<MAX_SECONDS AND child.returncode==0',workers=exited,owned_remaining=[],source_unchanged=True,original_prepost_10496_15_212_passed=True,saved_final_all4_idle=True,device_state_requeried=False),
  actual_terminal_manifest=evidence,failed_HOST_auditor_schema_attempts={str(p):sha(p)for p in(D/'v181-terminal-audit-attempts').iterdir()if p.is_file()},limitations=['HOST rechecks retained metadata/counters/history and original audit, not full hidden/KV tensors.','The original B4 component is qualified only in this terminated instance; it does not authorize a new process.','No new union or full463 result exists. Frozen181 and inheriting182 remain unusable until a separate corrected epoch is qualified.'],
  Torch_imported='torch'in sys.modules,NPU_used=False,model_constructed=False,parent_suites_rerun=False,adopted=False,goal_achieved=False)
 assert record['Torch_imported']is False
 assert not OUT.exists();OUT.write_text(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),bytes=OUT.stat().st_size,status=record['status'],original_base_qualified=True,timeout=False)))

if __name__=='__main__':main()
