"""Freeze additive control sources and all exact348 dependencies privately."""
import ast
import hashlib
import json
import tarfile
from pathlib import Path


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
DEST = BASE / 'tree-admission-closed-control-full-v355'
PARENT = BASE / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348'
HOST = BASE / 'tree-priority-protected-live-refill-host-v354'
DIAG = BASE / 'budget-diagnostics'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = DEST / 'FREEZE355.json'
    archive = DEST / 'SOURCE355.tar.gz'
    assert not manifest.exists() and not archive.exists()
    assert json.loads((DEST / 'HOST355_RESULT.json').read_text())['status'].startswith('PASS_HOST_')
    assert json.loads((HOST / 'HOST354_RESULT.json').read_text())['random_mode_cases'] == 720
    parent_manifest = PARENT / 'FREEZE348.json'
    parent_sha = '09119c918ad6d28d137a731cf041e69f7a0135dc302169455fa944df6caab5c8'
    assert sha(parent_manifest) == parent_sha
    parent = json.loads(parent_manifest.read_text())
    files = dict(parent['files'])
    files[str(parent_manifest)] = parent_sha
    for directory in (DEST, HOST):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in ('.py', '.sh', '.json'):
                if path.suffix == '.py':
                    ast.parse(path.read_text(), filename=str(path))
                files[str(path)] = sha(path)
    for name in ('build_ROOT355_closed_control_HOST.py', 'freeze_ROOT355_closed_control_HOST.py'):
        path = DIAG / name
        files[str(path)] = sha(path)
    for name, expected in files.items():
        assert sha(name) == expected, name
    value = dict(files=dict(sorted(files.items())), parent348_freeze_sha256=parent_sha,
                 parent348_archive_sha256=sha(PARENT / 'SOURCE348.tar.gz'),
                 admission_mode='closed',
                 scope='Exact348 M120/FULL2560,305/312,B1to8; closed control admission. Fresh original startup/components,small16,target408,draft45,recovery20,warm/twozero/seal/full463 and ownAR required.',
                 device_qualification_inherited=False)
    with manifest.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    sources = dict(files)
    sources[str(manifest)] = sha(manifest)
    with tarfile.open(archive, 'x:gz') as saved:
        for name in sorted(sources):
            saved.add(name, arcname=name.lstrip('/'), recursive=False)
    with tarfile.open(archive, 'r:gz') as saved:
        members = saved.getmembers()
        assert len(members) == len(sources)
        for member in members:
            assert member.isfile()
            assert hashlib.sha256(saved.extractfile(member).read()).hexdigest() == sources['/' + member.name]
    print(json.dumps(dict(freeze_sha256=sha(manifest), archive_sha256=sha(archive),
                          source_members=len(sources), archive_bytes=archive.stat().st_size,
                          admission_mode='closed', NPU_used=False)))


if __name__ == '__main__':
    main()
