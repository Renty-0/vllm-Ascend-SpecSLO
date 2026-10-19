"""Freeze additive361 sources; source retention only, never a device launch."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-Q5-bucket-completion-and-paired-mode-cost-v361'
OLD = B / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'
D = B / 'budget-diagnostics'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze_path = P / 'FREEZE361.json'
    archive_path = P / 'SOURCE361.tar.gz'
    receipt_path = P / 'SOURCE361_RECEIPT.json'
    assert not any(path.exists() for path in (freeze_path, archive_path, receipt_path))
    for name in ('HOST361_COST_PROTOCOL_RESULT.json', 'HOST361_RUNTIME_RESULT.json',
                 'HOST361_MODE_AUDIT_RESULT.json', 'HOST361_GUARD_SOURCE_RESULT.json'):
        value = json.loads((P / name).read_text())
        assert value['status'].startswith('PASS_HOST') and value['NPU_used'] is False
    selected = sorted([path for path in P.iterdir() if path.suffix in ('.py', '.sh', '.json')]
                      + [D / 'build_ROOT361_bucket_completion_guard_HOST.py', Path(__file__).resolve()])
    for path in selected:
        if path.suffix == '.py':
            ast.parse(path.read_text())
        elif path.suffix == '.sh':
            subprocess.run(['/bin/bash', '-n', str(path)], check=True)
    pins = json.loads((OLD / 'SOURCE344_RECEIPT.json').read_text())
    assert sha(OLD / 'FREEZE344.json') == pins['freeze_sha256']
    assert sha(OLD / 'SOURCE344.tar.gz') == pins['archive_sha256']
    value = dict(schema='ROOT361.additive-KV-bucket-completion-source.v1',
                 parent_freeze_sha256=pins['freeze_sha256'], parent_archive_sha256=pins['archive_sha256'],
                 files={str(path): sha(path) for path in selected},
                 target_cases_each=594, independent_prefixes_each_target=216,
                 old344_numeric_body_retained=True, new_paired_FULL_bucket_mode_costs=True,
                 original_budget_costs_retained=True, complete_cost_panels_durable=True,
                 deadline_seconds=3600, private_tag_limit_MiB=192, aggregate_limit_MiB=3584,
                 model_or_performance_qualification_inherited=False,
                 full594_qualified=False, NPU_used=False, Raw=None, Goodput=None)
    with freeze_path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    inventory = dict(value['files'], **{str(freeze_path): sha(freeze_path)})
    with archive_path.open('xb') as output:
        with gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for filename in sorted(inventory):
                    path = Path(filename)
                    info = archive.gettarinfo(str(path), arcname=filename.lstrip('/'))
                    info.uid = info.gid = 0
                    info.uname = info.gname = 'root'
                    info.mtime = 0
                    with path.open('rb') as source:
                        archive.addfile(info, source)
    receipt = dict(freeze_sha256=sha(freeze_path), archive_sha256=sha(archive_path),
                   own_archive_members=len(inventory), own_archive_bytes=archive_path.stat().st_size,
                   parent344_source_only_pins={key: pins[key] for key in ('freeze_sha256', 'archive_sha256')},
                   NPU_used=False, model_or_performance_qualification_inherited=False, Goodput=None)
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
