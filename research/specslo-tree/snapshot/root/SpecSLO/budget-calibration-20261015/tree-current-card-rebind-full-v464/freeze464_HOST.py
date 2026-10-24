"""Seal current device control, retaining the exact frozen463 algorithm."""
from pathlib import Path
import ast
import datetime
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
    control = json.loads((P / 'HOST464_RESULT.json').read_text())
    if control['status'] != 'PASS_HOST464_ONLY_DEVICE_CONTROL_REBIND' or control['NN_permission'] is not False:
        raise RuntimeError('Actual current source controls required')
    for path, wanted in control['sources'].items():
        if sha(path) != wanted:
            raise RuntimeError('Changed HOST-tested source: ' + path)
    mapping = json.loads((P / 'actual-retained-device-mapping.json').read_text())
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
    paths += [verified, D / 'ROOT463_INTERRUPTED_RESTART_20261023_HOST.json']
    for path in paths:
        if path.is_symlink():
            raise RuntimeError('Regular source files required')
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
    files = {str(path): sha(path) for path in sorted(paths)}
    freeze = P / 'FREEZE464.json'
    value = dict(schema='ROOT464.current-card-source.v1', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=files, parent463_freeze='d27c7eb676ad7268ad7911c455e86177b3691b7e8b21cc4c28799c5f04de19a5',
        parent463_archive='e312cdeaf16066b101b565e67e0a1258eb4cfa6683632af6d279527f741a630c',
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
    archive = P / 'SOURCE464.tar.gz'
    with archive.open('xb') as stream:
        stream.write(gzip.compress(buffer.getvalue(), mtime=0))
    result = dict(status='HOST464_SOURCE_FROZEN_NO_NN_PERMISSION', source=str(P),
        source_freeze_SHA256=sha(freeze), source_archive_SHA256=sha(archive),
        own_source_members=len(inventory), original463_algorithm_unchanged=True,
        physical_cards=[0, 1, 2, 7], NN_permission_inherited=False, NPU_model_launched=False)
    with (P / 'SOURCE464_RECEIPT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
