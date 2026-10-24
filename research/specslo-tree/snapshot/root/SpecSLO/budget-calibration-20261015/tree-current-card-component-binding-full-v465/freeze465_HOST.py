"""Seal corrected inner device binding, retaining all frozen463 algorithm gates."""
from pathlib import Path
import ast
import datetime
import fcntl
import gzip
import hashlib
import io
import json
import tarfile

P = Path(__file__).resolve().parent
D = P.parent / 'budget-diagnostics'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    control = json.loads((P / 'HOST465_RESULT.json').read_text())
    if control['status'] != 'PASS_HOST465_ACTUAL_COMPONENT_CURRENT_CARDS' or control['NN_permission'] is not False:
        raise RuntimeError('Actual current source controls required')
    for path, wanted in control['sources'].items():
        if sha(path) != wanted:
            raise RuntimeError('Changed HOST-tested source: ' + path)
    mapping = json.loads((P.parent / 'tree-current-card-rebind-full-v464/actual-retained-device-mapping.json').read_text())
    if mapping['devices'] != [dict(user=i, logical=i, physical=p) for i, p in enumerate([0, 1, 2, 7])]:
        raise RuntimeError('Current actual ACL device mapping required')
    if mapping['acl_set_device_called'] is not False:
        raise RuntimeError('Mapping query must not run a model')
    verified = D / 'ROOT464_CURRENT_CARDS_RETAINED_ENV_VERIFIER.log'
    result = json.loads(verified.read_text())
    if result != dict(baseline_id='nano-pearl-ascend-v1', status='pass', checked_frozen_files=10496,
            checked_external_files=15, checked_packages=212,
            public_model_check='resolved path/size/mtime match previously SHA-verified public files', errors=[]):
        raise RuntimeError('Actual unchanged full environment verifier required')
    paths = [p for p in P.iterdir() if p.is_file() and p.suffix in ('.py', '.sh', '.json')
             and not p.name.startswith(('FREEZE', 'SOURCE'))]
    paths += [verified, D / 'ROOT465_INHERITED_DEVICE_CONSTANT_INVENTORY_HOST.json', D / 'ROOT464_TERMINAL_DEVICE_ASSERTION_FAILURE_HOST.json']
    for path in paths:
        if path.is_symlink():
            raise RuntimeError('Regular source files required')
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
    files = {str(path): sha(path) for path in sorted(paths)}
    freeze = P / 'FREEZE465.json'
    value = dict(schema='ROOT465.current-card-source.v1', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=files, parent464_freeze='aadc2090b2cb5ffc22243ab19fdbc007481b79a08c7b5c1b806867e3835c12b1',
        parent464_archive='c1e891ef13827d0439f0f56728612b25e0ea00ff194bed87d386026b6a004ae3',
        physical_cards=[0, 1, 2, 7], previous_physical_cards=[2, 5, 6, 7],
        global_active_limit=64, original463_algorithm_unchanged=True, NN_permission_inherited=False,
        every_original_and_additional_gate_required=True)
    with freeze.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    inventory = dict(files)
    inventory[str(freeze)] = sha(freeze)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name in sorted(inventory):
            data = Path(name).read_bytes()
            info = tarfile.TarInfo(name.lstrip('/'))
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
    archive = P / 'SOURCE465.tar.gz'
    with archive.open('xb') as stream:
        stream.write(gzip.compress(buffer.getvalue(), mtime=0))
    result = dict(status='HOST465_SOURCE_FROZEN_NO_NN_PERMISSION', source=str(P),
        source_freeze_SHA256=sha(freeze), source_archive_SHA256=sha(archive),
        own_source_members=len(inventory), original463_algorithm_unchanged=True,
        physical_cards=[0, 1, 2, 7], NN_permission_inherited=False, NPU_model_launched=False)
    with (P / 'SOURCE465_RECEIPT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    with (P.parent / '.ROOT_NPU_EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
