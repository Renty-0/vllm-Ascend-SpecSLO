"""Verified private archival of superseded355/356; current405 and best/control untouched."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
DONOR = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAGS = ('M120-B8-closed-control-root-v355-full463-ownAR-v1', 'M120-B8-protected-tight-normal-root-v356-full463-ownAR-v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pinned = set()
    for path in B.glob('*/FREEZE*.json'):
        value = json.loads(path.read_text()).get('files', {})
        if isinstance(value, dict):
            pinned.update(value)
    destination = DONOR / 'private-evidence-archives'
    destination.mkdir(exist_ok=True)
    summaries = []
    for tag in TAGS:
        source = DONOR / 'actual' / tag
        assert source.parent == DONOR / 'actual' and source.is_dir()
        assert not (source / 'ARCHIVED.json').exists()
        receipt = DONOR / f'receipt-{tag}.json'
        row = json.loads(receipt.read_text())
        assert row['status'] != 'running' and row.get('finished_UTC')
        assert row.get('owned_remaining') == []
        files = sorted(p for p in source.rglob('*') if p.is_file())
        assert files and not any(p.is_symlink() for p in source.rglob('*'))
        assert not any(str(p) in pinned for p in files)
        index = {str(p.relative_to(DONOR)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files}
        original_bytes = sum(value['bytes'] for value in index.values())
        archive = destination / f'{tag}.tar.gz'
        assert not archive.exists()
        temporary = archive.with_suffix('.gz.inprogress')
        assert not temporary.exists()
        with tarfile.open(temporary, 'w:gz', compresslevel=6) as saved:
            saved.add(source, arcname=str(source.relative_to(DONOR)), recursive=True)
        with tarfile.open(temporary, 'r:gz') as saved:
            members = [m for m in saved.getmembers() if m.isfile()]
            assert len(members) == len(index) and {m.name for m in members} == set(index)
            for member in members:
                expected = index[member.name]
                assert member.size == expected['bytes']
                assert hashlib.sha256(saved.extractfile(member).read()).hexdigest() == expected['sha256']
        assert all(sha(DONOR / relative) == expected['sha256'] for relative, expected in index.items())
        temporary.rename(archive)
        full_index = destination / f'{tag}.index.json.gz'
        metadata = dict(schema='ROOT409.superseded355356-verified-private-evidence-archive.v1',
                        source=str(source), source_terminal_status=row['status'],
                        receipt=str(receipt), receipt_sha256=sha(receipt),
                        archive=str(archive), archive_sha256=sha(archive),
                        original_file_count=len(files), original_file_bytes=original_bytes,
                        archive_bytes=archive.stat().st_size, per_file=index,
                        all_archive_members_size_and_SHA_verified=True,
                        original_files_rehashed_before_removal=True,
                        intersection_with_all_current_FREEZE_file_paths=0,
                        public_storage_used=False, model_or_dataset_moved=False,
                        current_running_and_best376_and_control398_untouched=True,
                        restore_base=str(DONOR),
                        restore_argv=['tar', '-xzf', str(archive), '-C', str(DONOR)])
        full_index.write_bytes(gzip.compress(json.dumps(metadata, indent=2).encode(), mtime=0))
        assert json.loads(gzip.decompress(full_index.read_bytes()))['per_file'] == index
        errors = {p.name: p.read_bytes() for p in source.glob('rank*-last-error.json')}
        shutil.rmtree(source)
        source.mkdir()
        for name, data in errors.items():
            (source / name).write_bytes(data)
        marker = {k: v for k, v in metadata.items() if k != 'per_file'}
        marker.update(full_index=str(full_index), full_index_sha256=sha(full_index),
                      retained_small_error_files=sorted(errors))
        (source / 'ARCHIVED.json').write_text(json.dumps(marker, indent=2) + '\n')
        summaries.append(marker)
        print(json.dumps(dict(tag=tag, original_bytes=original_bytes,
                              archive_bytes=archive.stat().st_size,
                              saved_bytes=original_bytes - archive.stat().st_size)), flush=True)
    report = B / 'budget-diagnostics/ROOT409_SUPERSEDED355356_PRIVATE_ARCHIVAL_RESULT.json'
    report.write_text(json.dumps(dict(status='COMPLETE_VERIFIED_PRIVATE_ARCHIVAL',
        runs=summaries, current_NPU_job_changed=False, source_frozen_paths_modified=False,
        models_or_datasets_moved=False, public_storage_used=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
