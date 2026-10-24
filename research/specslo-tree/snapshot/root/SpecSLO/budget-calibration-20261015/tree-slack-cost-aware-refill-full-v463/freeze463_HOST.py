"""Seal this declared host-policy candidate; never inherit device permission."""

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
    for name in ('HOST463_POLICY_RESULT.json', 'HOST463_INTEGRATION_RESULT.json', 'HOST463_SHELL_RESULT.json'):
        value = json.loads((P / name).read_text())
        if not value['status'].startswith('PASS_HOST463') or value['NPU_used'] is not False:
            raise RuntimeError('Actual source controls required before freeze: ' + name)
        for path, wanted in value.get('sources', {}).items():
            if sha(path) != wanted:
                raise RuntimeError('HOST control source changed after execution: ' + path)
    prior = json.loads((D / 'ROOT462_V2_FINAL_COMMON_CLOCK_HOST.json').read_text())
    if prior['metrics']['all']['Goodput'] != 460.81910634078093 or prior['goal_achieved'] is not False:
        raise RuntimeError('Original qualified462 performance checkpoint required')
    paths = [p for p in P.iterdir() if p.is_file() and p.suffix in ('.py', '.sh', '.json')
             and not p.name.startswith(('FREEZE', 'SOURCE'))]
    paths += [D / 'ROOT462_PROSPECTIVE_PREFILL_COST_MODEL_HOST.json',
              D / 'ROOT462_V2_FINAL_COMMON_CLOCK_HOST.json',
              D / 'ROOT462_V2_TERMINAL_FULL_AUDIT_HOST.json']
    for path in paths:
        if path.is_symlink():
            raise RuntimeError('Regular original source files required')
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
    files = {str(p): sha(p) for p in sorted(paths)}
    value = dict(schema='ROOT463.credit-cost-refill-source.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=files,
        numerical_parent462_freeze='9f79697914d74f79cc94fe4605c8f20665c44ee2f6b33fe99b59b73d7b4e2280',
        numerical_parent462_archive='cebb1af09f459d8e0a64b9a6b7561721f07e787cca5fb809fd09552e483bfbd9',
        global_active_limit=64, prefill_cohort_maximum=32,
        NN_permission_inherited=False, every_original_and_additional_own_physical_gate_required=True)
    freeze = P / 'FREEZE463.json'
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
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    packed = gzip.compress(buffer.getvalue(), mtime=0)
    archive = P / 'SOURCE463.tar.gz'
    with archive.open('xb') as stream:
        stream.write(packed)
    result = dict(status='HOST463_SOURCE_FROZEN_NO_NN_PERMISSION', source=str(P),
        source_freeze_SHA256=sha(freeze), source_archive_SHA256=sha(archive),
        own_source_members=len(inventory), model_or_NPU_used=False,
        NN_permission_inherited=False, NPU_launched=False)
    with (P / 'SOURCE463_RECEIPT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
