"""Freeze the dual route without borrowing any previous model permission."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-global64-S24-S32-dual-serving-v409'
PARENTS = (
    ('tree-normal24-derived-callback-repair-v405', 405),
    ('tree-frugal-S32-startup-import-repair-v406', 406),
    ('tree-S24-S32-whole-ticket-routing-host-v407', 407),
    ('tree-S24-S32-physical-dispatch-host-v408', 408),
    ('tree-S32-actual-typed-allocation-host-v404', 404),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (P / 'FREEZE409.json').exists()
    files = {}
    parents = {}
    for folder, version in PARENTS:
        path = B / folder / f'FREEZE{version}.json'
        value = json.loads(path.read_text())
        for name, digest in value['files'].items():
            assert sha(name) == digest, name
            assert name not in files or files[name] == digest
            files[name] = digest
        files[str(path)] = sha(path)
        parents[str(version)] = sha(path)
    for path in sorted(P.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.log'):
            if path.suffix == '.py':
                ast.parse(path.read_text())
            files[str(path)] = sha(path)
    for name in ('run_ROOT409_actual_startup_HOST.py',
                 'ROOT406_TERMINAL_INDEPENDENT_FINITE_AUDIT.json',
                 'ROOT406_FINAL_FINITE_RESULT.json'):
        path = B / 'budget-diagnostics' / name
        files[str(path)] = sha(path)
    files[str(Path(__file__).resolve())] = sha(__file__)
    freeze = dict(schema='ROOT409.current-dual-route-full-source-freeze.v1',
                  files=files, parents=parents, global_active_limit=64,
                  normal_draft_limit=32,
                  narrow=dict(N=24, C=96, M=120, S=24),
                  wide=dict(N=32, C=88, M=120, S=32, B_max=3, B3_owners=24),
                  qualification_inherited=False,
                  original_components_227_681_plus_fresh200_required=True,
                  original_warm_twozero_seal_full463_ownAR_required=True)
    path = P / 'FREEZE409.json'
    path.write_text(json.dumps(freeze, indent=2) + '\n')
    files_with_freeze = dict(files, **{str(path): sha(path)})
    archive = P / 'SOURCE409.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=6) as saved:
        for name in sorted(files_with_freeze):
            saved.add(name, arcname=name.lstrip('/'), recursive=False)
    with tarfile.open(archive, 'r:gz') as saved:
        members = saved.getmembers()
        assert len(members) == len(files_with_freeze)
        for member in members:
            assert member.isfile()
            assert hashlib.sha256(saved.extractfile(member).read()).hexdigest() == files_with_freeze['/' + member.name]
    assert all(sha(name) == digest for name, digest in files_with_freeze.items())
    receipt = dict(schema='ROOT409.source-archive-receipt.v1',
                   freeze_sha256=sha(path), archive_sha256=sha(archive),
                   archive_bytes=archive.stat().st_size, members=len(files_with_freeze),
                   no_NPU_launched=True, no_inherited_model_permission=True)
    (P / 'SOURCE409_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
