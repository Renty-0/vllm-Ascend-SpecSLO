"""Lossless private archival of explicit terminal superseded experiments."""
from pathlib import Path
import datetime
import gzip
import hashlib
import json
import shutil
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
R = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAGS = (
    'M160-FIA160-MC2projection165-root-v331-B1to8-532-math-cost-v1',
    'M160-S32-C128-root-v329-B1to8-532-math-cost-v1',
    'span-prefix-dual-cache-concrete-device-v296-B1to8-408-math-cost-v1',
    'prefix-grid-replay-root-v280-B1to8-408-math-and-paired-cost-v1',
    'static-M120-root-v260-spare-B5-groups-full463-ownAR-v1',
    'measured-target-cost-risk-root-v316-B1to8-full463-ownAR-v1',
    'active64-whole-ticket-S24-segments-root-v398-full463-ownAR-v1',
    'Q5-KV-bucket-all594-and-paired-FULL-root-v361-target-component-v1',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def pinned_paths():
    pins = set()

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and key.startswith('/'):
                    pins.add(key)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and value.startswith('/'):
            pins.add(value)

    for freeze in B.glob('*/FREEZE*.json'):
        collect(json.loads(freeze.read_text()))
    collect(json.loads((D / 'ROOT456_ORIGINAL455_SOURCE_CLOSURE_HOST.json').read_text()))
    return pins


def archive_one(tag, declared):
    assert declared['archival_eligible'] is True
    source = R / 'actual' / tag
    receipt = R / f'receipt-{tag}.json'
    assert sha(receipt) == declared['receipt_SHA256']
    row = json.loads(receipt.read_text())
    assert row['status'] in ('complete', 'failed') and row['owned_remaining'] == []
    assert row['sources_unchanged'] is True and row['devices_after']['all_idle'] is True
    assert not (source / 'ARCHIVED.json').exists()
    files = sorted(path for path in source.rglob('*') if path.is_file())
    assert len(files) == declared['files'] and sum(path.stat().st_size for path in files) == declared['bytes']
    assert not any(path.is_symlink() for path in source.rglob('*'))
    pins = pinned_paths()
    assert not any(str(path) in pins for path in files), 'Frozen source/evidence inputs cannot be archived'
    index = {str(path.relative_to(R)): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in files}
    destination = R / 'private-evidence-archives'
    destination.mkdir(exist_ok=True)
    assert destination.resolve().is_relative_to(Path('/root/SpecSLO'))
    archive = destination / f'{tag}.tar.gz'
    temporary = archive.with_suffix('.gz.inprogress')
    assert not archive.exists() and not temporary.exists()
    with tarfile.open(temporary, 'w:gz', compresslevel=6) as saved:
        saved.add(source, arcname=str(source.relative_to(R)), recursive=True)
    with tarfile.open(temporary, 'r:gz') as saved:
        members = [member for member in saved.getmembers() if member.isfile()]
        assert len(members) == len(index) and {m.name for m in members} == set(index)
        for member in members:
            wanted = index[member.name]
            assert member.size == wanted['bytes']
            assert hashlib.sha256(saved.extractfile(member).read()).hexdigest() == wanted['sha256']
    assert all(sha(R / name) == value['sha256'] for name, value in index.items())
    archive_sha = sha(temporary)
    temporary.rename(archive)
    metadata = dict(
        schema='ROOT462.superseded-verified-private-archive.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), source=str(source),
        source_terminal_status=row['status'], receipt=str(receipt), receipt_sha256=sha(receipt),
        original_argv=row['child']['argv'], original_readiness=row['readiness'],
        original_scope=row['qualification_scope'], archive=str(archive), archive_sha256=archive_sha,
        original_file_count=len(files), original_file_bytes=declared['bytes'], archive_bytes=archive.stat().st_size,
        per_file=index, all_archive_members_size_and_SHA_verified=True,
        original_files_rehashed_before_removal=True, frozen_source_and_evidence_input_intersections=0,
        public_storage_used=False, restore_base=str(R),
        restore_argv=['tar', '-xzf', str(archive), '-C', str(R)], current462_changed=False)
    full_index = destination / f'{tag}.index.json.gz'
    assert not full_index.exists()
    full_index.write_bytes(gzip.compress(json.dumps(metadata, indent=2).encode(), mtime=0))
    assert json.loads(gzip.decompress(full_index.read_bytes()))['per_file'] == index
    errors = {path.name: path.read_bytes() for path in source.glob('rank*-last-error.json')}
    shutil.rmtree(source)
    source.mkdir()
    for name, raw in errors.items():
        (source / name).write_bytes(raw)
    marker = {key: value for key, value in metadata.items() if key != 'per_file'}
    marker.update(full_index=str(full_index), full_index_sha256=sha(full_index), retained_errors=sorted(errors))
    (source / 'ARCHIVED.json').write_text(json.dumps(marker, indent=2) + '\n')
    return dict(tag=tag, saved_bytes=declared['bytes'] - archive.stat().st_size,
                archive=str(archive), archive_sha256=archive_sha, full_index=str(full_index),
                full_index_sha256=sha(full_index), files=len(files))


def main():
    inventory = json.loads((D / 'ROOT462_TERMINAL_PRIVATE_ARCHIVAL_INVENTORY_HOST.json').read_text())
    declared = {row['tag']: row for row in inventory['rows']}
    results = []
    for tag in TAGS:
        result = archive_one(tag, declared[tag])
        results.append(result)
        print(json.dumps(result), flush=True)
        partial = dict(status='IN_PROGRESS_VERIFIED_PRIVATE_SUPERSEDED_ARCHIVAL', results=results,
                       current462_changed=False, public_storage_used=False)
        (D / 'ROOT462_SUPERSEDED_PRIVATE_ARCHIVAL_PROGRESS.json').write_text(json.dumps(partial, indent=2) + '\n')
    value = dict(status='COMPLETE_VERIFIED_PRIVATE_SUPERSEDED_ARCHIVAL', results=results,
                 saved_bytes=sum(row['saved_bytes'] for row in results),
                 current462_changed=False, frozen_sources_modified=False, public_storage_used=False)
    with (D / 'ROOT462_SUPERSEDED_PRIVATE_ARCHIVAL_RESULT.json').open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=value['status'], saved_bytes=value['saved_bytes'])), flush=True)


if __name__ == '__main__':
    from private_maintenance_lock_HOST import exclusive_private_maintenance
    with exclusive_private_maintenance():
        main()
