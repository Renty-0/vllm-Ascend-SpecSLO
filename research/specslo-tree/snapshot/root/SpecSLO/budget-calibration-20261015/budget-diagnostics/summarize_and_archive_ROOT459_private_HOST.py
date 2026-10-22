"""Summarize actual paired samples and losslessly archive terminal459 privately."""
from pathlib import Path
import datetime,gzip,hashlib,json,shutil,statistics,tarfile
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='adaptive-own-physical-root-v459-693-node-and-paired-cost-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=R/'actual'/TAG;receipt=R/f'receipt-{TAG}.json'
    assert sha(receipt)=='12746563b82559ed3d2fcad5d1cde4b8a4a3fa3fdfeaf29a5312ccc2c10a5dba'
    original=json.loads(receipt.read_text());audit=D/'ROOT459_TERMINAL_INDEPENDENT_NODE_AUDIT_HOST.json'
    assert original['status']=='complete'and original['sources_unchanged']is True and original['owned_remaining']==[]and original['devices_after']['all_idle']is True
    assert json.loads(audit.read_text())['status']=='PASS_HOST459_SOURCE_AND_ALL693_OWN_PHYSICAL_NODE_RECORDS'
    rows=[];raw_pins={}
    for bank in('mixed','mid','short'):
        ranks=[]
        for rank in(1,2,3):
            path=source/bank/f'rank{rank}-cost-and-prefix-protection.json';raw_pins[str(path)]=sha(path)
            panel=json.loads(path.read_text())['paired_physical_cost'];assert panel['rounds']==20 and len(panel['pairs'])==20
            ranks.append(panel)
        for arm in ranks[0]['arms']:
            values={mode:[]for mode in('fixed120','adaptive80_120')}
            for iteration in range(20):
                actual=[]
                for panel in ranks:
                    pair=panel['pairs'][iteration];assert pair['iteration']==iteration
                    sample=next(v for v in pair['samples']if v['arm']==arm['name']);actual.append(sample['milliseconds'])
                for mode in values:values[mode].append(max(v[mode]for v in actual))
            old=statistics.median(values['fixed120']);new=statistics.median(values['adaptive80_120'])
            rows.append(dict(bank=bank,arm=arm['name'],fixed120_ms=old,adaptive_ms=new,speedup=old/new,time_reduction_percent=100*(1-new/old),max_rank_samples=values))
    summary=dict(status='PASS_HOST459_ACTUAL_PAIRED_MAX_TARGET_RANK_CALL_COSTS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_receipt=str(receipt),source_receipt_sha256=sha(receipt),independent_terminal_audit=str(audit),raw_panel_SHA256=raw_pins,
        method='Twenty alternating paired synchronized full target calls; maximum of3 TP ranks per iteration, then median',
        rows=rows,NPU_used=False,new_NPU_run=False,full463=False,Raw=None,Goodput=None)
    with(D/'ROOT459_PAIRED_MAX_RANK_CALL_COSTS_HOST.json').open('x')as f:json.dump(summary,f,indent=2);f.write('\n')
    pinned=set()
    def collect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if isinstance(key,str)and key.startswith('/')and Path(key).is_file():pinned.add(key)
                collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
        elif isinstance(value,str)and value.startswith('/')and Path(value).is_file():pinned.add(value)
    for freeze in B.glob('*/FREEZE*.json'):collect(json.loads(freeze.read_text()))
    files=sorted(p for p in source.rglob('*')if p.is_file());assert files and not(source/'ARCHIVED.json').exists()
    assert not any(p.is_symlink()for p in source.rglob('*'))and not any(str(p)in pinned for p in files),'Frozen evidence inputs must remain in place'
    index={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in files}
    destination=R/'private-evidence-archives';destination.mkdir(exist_ok=True)
    assert destination.resolve().is_relative_to(Path('/root/SpecSLO'))
    archive=destination/f'{TAG}.tar.gz';temporary=archive.with_suffix('.gz.inprogress');assert not archive.exists()and not temporary.exists()
    with tarfile.open(temporary,'w:gz',compresslevel=6)as saved:saved.add(source,arcname=str(source.relative_to(R)),recursive=True)
    with tarfile.open(temporary,'r:gz')as saved:
        members=[m for m in saved.getmembers()if m.isfile()];assert len(members)==len(index)and{m.name for m in members}==set(index)
        for member in members:
            wanted=index[member.name];assert member.size==wanted['bytes']and hashlib.sha256(saved.extractfile(member).read()).hexdigest()==wanted['sha256']
    assert all(sha(R/name)==row['sha256']for name,row in index.items())
    temporary.rename(archive)
    metadata=dict(schema='ROOT460.completed459-private-lossless-archive.v1',source=str(source),source_terminal_status='complete',
        receipt=str(receipt),receipt_sha256=sha(receipt),original_argv=original['child']['argv'],original_readiness=original['readiness'],original_scope=original['qualification_scope'],
        additional_reproduction_records=[dict(path=str(path),sha256=sha(path))for path in(audit,D/'ROOT459_PAIRED_MAX_RANK_CALL_COSTS_HOST.json')],
        archive=str(archive),archive_sha256=sha(archive),original_file_count=len(files),original_file_bytes=sum(v['bytes']for v in index.values()),archive_bytes=archive.stat().st_size,
        per_file=index,all_archive_members_size_and_SHA_verified=True,original_files_rehashed_before_removal=True,frozen_source_and_evidence_input_intersections=0,
        public_storage_used=False,models_or_datasets_moved=False,restore_base=str(R),restore_argv=['tar','-xzf',str(archive),'-C',str(R)])
    full_index=destination/f'{TAG}.index.json.gz';assert not full_index.exists();full_index.write_bytes(gzip.compress(json.dumps(metadata,indent=2).encode(),mtime=0))
    assert json.loads(gzip.decompress(full_index.read_bytes()))['per_file']==index
    shutil.rmtree(source);source.mkdir();marker={key:value for key,value in metadata.items()if key!='per_file'}
    marker.update(full_index=str(full_index),full_index_sha256=sha(full_index));(source/'ARCHIVED.json').write_text(json.dumps(marker,indent=2)+'\n')
    result=dict(status='COMPLETE_VERIFIED_PRIVATE_COMPLETED459_ARCHIVAL',run=marker,saved_bytes=metadata['original_file_bytes']-archive.stat().st_size,
                frozen_sources_modified=False,public_storage_used=False,new_NPU_run=False)
    with(D/'ROOT460_COMPLETED459_PRIVATE_ARCHIVAL_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],saved_bytes=result['saved_bytes'],archive_bytes=archive.stat().st_size)),flush=True)
if __name__=='__main__':
    from private_maintenance_lock_HOST import exclusive_private_maintenance
    with exclusive_private_maintenance():main()
