"""Remove only old unreferenced compiler cache and exact duplicate checkpoints."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
CACHE=B/'tree-M165-N32-independent-target-math-cost-v317/triton-cache'
TAGS=('M160-S32-C128-root-v329-B1to8-532-math-cost-v1',
      'M160-FIA160-MC2projection165-root-v331-B1to8-532-math-cost-v1')

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    sys.path.insert(0,str(B/'tree-FIA160-projection165-MC2-audit-adapter-v332'))
    import guard332
    guard332.PIN='b14c31542a98d6cffb742a1632210ea235a61bf86028ea70d4ca710157fc6eea'
    _,frozen,_=guard332.prepared()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:env=(proc/'environ').read_bytes().split(b'\0')
        except OSError:continue
        for variable in env:
            if variable.startswith(b'TRITON_CACHE_DIR='):
                value=variable.split(b'=',1)[1].decode(errors='replace')
                if value==str(CACHE)or value.startswith(str(CACHE)+'/'):
                    raise RuntimeError('Old cache still referenced by a live process '+proc.name)
    checkpoints=[]
    for tag in TAGS:
        receipt=json.loads((O/f'receipt-{tag}.json').read_text())
        assert receipt['owned_remaining']==[]
        if receipt['child']['exit_code'] != 0:
            assert tag == TAGS[1] and receipt['status'] == 'failed'
            assert receipt['child']['exit_code'] == -15
            assert len(receipt['errors']) == 2
            assert all('aggregate private bound' in error for error in receipt['errors'])
            assert (O/'actual'/tag/'completed.json').is_file()
        assert receipt['sources_unchanged']is True and receipt['post_verifier']['exit_code']==0
        assert receipt['devices_after']['all_idle']is True
        for rank in range(4):
            path=O/'actual'/tag/f'rank{rank}-completed-math-before-cost.json'
            final=O/'actual'/tag/f'rank{rank}-numeric.json'
            assert path.is_file()and not path.is_symlink()and str(path)not in frozen
            value=json.loads(final.read_text())
            payload=dict(rank=rank,cases=532,math=value['math'],bank=value['bank'],
                actual_static_graph_calls=value['actual_static_graph_calls'],
                dispatch_audit=value['dispatch_audit'],performance=False)
            raw=json.dumps(payload,indent=2).encode()
            assert raw==path.read_bytes()
            checkpoints.append(dict(path=str(path),bytes=len(raw),sha256=sha(raw),
                final_report=str(final),final_report_sha256=sha(final.read_bytes()),
                reconstruction_fields_in_order=list(payload),rank=rank,cases=532,performance=False))
    assert CACHE.is_dir()and not CACHE.is_symlink()
    cache_files=[]
    for path in sorted(CACHE.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():
            assert str(path)not in frozen
            raw=path.read_bytes();cache_files.append(dict(path=str(path),bytes=len(raw),sha256=sha(raw)))
    record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='verified_before_cleanup',
        cache_root=str(CACHE),cache_files=cache_files,checkpoints=checkpoints,
        total_bytes=sum(x['bytes']for x in cache_files+checkpoints),
        original_receipts_and_all_final_numeric_reports_retained=True,
        frozen_source_or_model_or_dataset_removed=False,public_storage_used=False,
        cache_regeneration='Automatically compiled on first use by preserved frozen317/318/320 source and retained CANN9.0/torch-npu2.10.0/Triton environment; cache itself is not measurement evidence. Use a fresh diagnostic tag and original freeze pins. M160 continues to use its separate326 cache.',
        checkpoint_reconstruction='Read final_report, select rank,cases=532,math,bank,actual_static_graph_calls,dispatch_audit,performance=False in recorded order; json.dumps(payload,indent=2).encode() with no trailing newline. Verify sha256 against this record before restoring.',
        deleted=False)
    manifest=D/'ROOT329_331_REDUNDANT_CHECKPOINT_AND_OLD_CACHE_REGENERATION_20261019.json'
    with manifest.open('x')as stream:json.dump(record,stream,indent=2);stream.write('\n')
    for row in checkpoints:Path(row['path']).unlink()
    shutil.rmtree(CACHE)
    record.update(status='complete_exact_reconstruction_verified',deleted=True,
                  deleted_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],cache_files=len(cache_files),
        duplicate_checkpoints=len(checkpoints),reclaimed_bytes=record['total_bytes'],manifest=str(manifest))))

if __name__=='__main__':main()
