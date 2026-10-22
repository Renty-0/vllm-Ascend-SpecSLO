"""Verified private archival of terminal425; current435 and all frozen sources untouched."""
import gzip,hashlib,json,shutil,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='canonical-tail-binding-parent4-E4-global64-root-v425-current75-full463-ownAR-v1'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def main():
 pinned=set()
 def collect(value):
  if isinstance(value,dict):
   for k,v in value.items():
    if isinstance(k,str)and k.startswith('/')and Path(k).is_file():pinned.add(k)
    collect(v)
  elif isinstance(value,list):
   for v in value:collect(v)
  elif isinstance(value,str)and value.startswith('/')and Path(value).is_file():pinned.add(value)
 for freeze in B.glob('*/FREEZE*.json'):collect(json.loads(freeze.read_text()))
 source=DONOR/'actual'/TAG
 assert source.is_dir()and not(source/'ARCHIVED.json').exists()
 receipt=DONOR/f'receipt-{TAG}.json';row=json.loads(receipt.read_text())
 assert row['status']=='failed'and row.get('finished_UTC')and row.get('owned_remaining')==[]
 assert row['sources_unchanged']is True and row['devices_after']['all_idle']is True
 files=sorted(p for p in source.rglob('*')if p.is_file())
 assert files and not any(p.is_symlink()for p in source.rglob('*'))
 assert not any(str(p)in pinned for p in files),'Never archive a frozen source/evidence input'
 index={str(p.relative_to(DONOR)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in files}
 original_bytes=sum(v['bytes']for v in index.values())
 destination=DONOR/'private-evidence-archives';destination.mkdir(exist_ok=True)
 assert destination.resolve().is_relative_to(Path('/root/SpecSLO'))
 archive=destination/f'{TAG}.tar.gz';temporary=archive.with_suffix('.gz.inprogress')
 assert not archive.exists()and not temporary.exists()
 with tarfile.open(temporary,'w:gz',compresslevel=6)as t:t.add(source,arcname=str(source.relative_to(DONOR)),recursive=True)
 with tarfile.open(temporary,'r:gz')as t:
  members=[m for m in t.getmembers()if m.isfile()]
  assert len(members)==len(index)and {m.name for m in members}==set(index)
  for m in members:
   value=index[m.name]
   assert m.size==value['bytes']and hashlib.sha256(t.extractfile(m).read()).hexdigest()==value['sha256']
 assert all(sha(DONOR/k)==v['sha256']for k,v in index.items())
 temporary.rename(archive)
 repro=[]
 for name in ('ROOT425_TERMINAL_SUMMARY.json','ROOT425_TERMINAL_CURRENT_STATE_FIELD_FAILURE_AUDIT.json'):
  f=B/'budget-diagnostics'/name
  if f.exists():repro.append(dict(path=str(f),sha256=sha(f)))
 metadata=dict(schema='ROOT435.negative425-verified-private-archive.v1',source=str(source),
  source_terminal_status=row['status'],receipt=str(receipt),receipt_sha256=sha(receipt),
  original_argv=row.get('child',{}).get('argv'),original_readiness=row.get('readiness'),
  original_scope=row.get('qualification_scope'),additional_reproduction_records=repro,
  archive=str(archive),archive_sha256=sha(archive),original_file_count=len(files),
  original_file_bytes=original_bytes,archive_bytes=archive.stat().st_size,per_file=index,
  all_archive_members_size_and_SHA_verified=True,original_files_rehashed_before_removal=True,
  frozen_source_and_evidence_input_intersections=0,public_storage_used=False,models_or_datasets_moved=False,
  current435_and_best376405_and_control398_untouched=True,
  restore_base=str(DONOR),restore_argv=['tar','-xzf',str(archive),'-C',str(DONOR)])
 full_index=destination/f'{TAG}.index.json.gz'
 assert not full_index.exists();full_index.write_bytes(gzip.compress(json.dumps(metadata,indent=2).encode(),mtime=0))
 assert json.loads(gzip.decompress(full_index.read_bytes()))['per_file']==index
 errors={p.name:p.read_bytes()for p in source.glob('rank*-last-error.json')}
 shutil.rmtree(source);source.mkdir()
 for name,value in errors.items():(source/name).write_bytes(value)
 marker={k:v for k,v in metadata.items()if k!='per_file'}
 marker.update(full_index=str(full_index),full_index_sha256=sha(full_index),retained_small_errors=sorted(errors))
 (source/'ARCHIVED.json').write_text(json.dumps(marker,indent=2)+'\n')
 result=dict(status='COMPLETE_VERIFIED_PRIVATE_NEGATIVE425_ARCHIVAL',run=marker,
  saved_bytes=original_bytes-archive.stat().st_size,current_NPU_job_changed=False,
  frozen_source_paths_modified=False,public_storage_used=False,models_or_datasets_moved=False)
 (B/'budget-diagnostics/ROOT435_NEGATIVE425_PRIVATE_ARCHIVAL_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],original_bytes=original_bytes,archive_bytes=archive.stat().st_size,saved_bytes=result['saved_bytes'])),flush=True)
if __name__=='__main__':main()
