"""Freeze source-only single-device FIA diagnostic; never launch devices."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-FIA-multi-owner-page-offset-invariance-v363'
OLD = B / 'tree-FIA-Q4-group-and-query-isolation-v335'
PARENT = B / 'tree-M160-C128-F64-global192-full-v333'
D = B / 'budget-diagnostics'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze = P / 'FREEZE363.json'
    archive = P / 'SOURCE363.tar.gz'
    receipt = P / 'SOURCE363_RECEIPT.json'
    assert not any(path.exists() for path in (freeze, archive, receipt))
    for name in ('HOST363_RESULT.json', 'HOST363_AUDIT_RESULT.json', 'HOST363_GUARD_RESULT.json'):
        result = json.loads((P / name).read_text())
        assert result['status'].startswith('PASS_HOST') and result['NPU_used'] is False
    host = json.loads((P / 'HOST363_RESULT.json').read_text())
    assert host['probe_sha256'] == sha(P / 'probe363.py')
    selected = sorted([path for path in P.iterdir() if path.suffix in ('.py', '.sh', '.json')]
                      + [OLD / 'guard335.py', OLD / 'startup335.sh',
                         D / 'build_ROOT363_synthetic_offset_guard_HOST.py', Path(__file__).resolve()])
    for path in selected:
        if path.suffix == '.py':
            ast.parse(path.read_text())
    pins = json.loads((PARENT / 'SOURCE333_RECEIPT.json').read_text())
    assert sha(PARENT / 'FREEZE333.json') == pins['freeze_sha256']
    assert sha(PARENT / 'SOURCE333.tar.gz') == pins['archive_sha256']
    value = dict(schema='ROOT363.synthetic-FIA-page-offset-source.v1',
                 parent_freeze_sha256=pins['freeze_sha256'], parent_archive_sha256=pins['archive_sha256'],
                 files={str(path): sha(path) for path in selected},
                 synthetic_cases=640, ABA_rows=1920, physical_M=120, groups=24,
                 KV_heads=[2, 3], Q_heads=[16, 24], mask_capacities=[10, 11],
                 uniform_extents=[2560, 4096], original_reference_extent=2560,
                 deadline_seconds=600, private_tag_limit_MiB=64, aggregate_limit_MiB=256,
                 original_requested_cards=[2, 5, 6, 7], logical_device=1,
                 models_loaded=False, model_qualified=False, performance=False,
                 qualification_inherited=False, NPU_used=False, Raw=None, Goodput=None)
    with freeze.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    inventory = dict(value['files'], **{str(freeze): sha(freeze)})
    with archive.open('xb') as output:
        with gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|') as bundle:
                for filename in sorted(inventory):
                    path = Path(filename)
                    info = bundle.gettarinfo(filename, arcname=filename.lstrip('/'))
                    info.uid = info.gid = 0
                    info.uname = info.gname = 'root'
                    info.mtime = 0
                    with path.open('rb') as source:
                        bundle.addfile(info, source)
    result = dict(freeze_sha256=sha(freeze), archive_sha256=sha(archive),
                  own_archive_members=len(inventory), own_archive_bytes=archive.stat().st_size,
                  parent333_source_only_pins={key: pins[key] for key in ('freeze_sha256', 'archive_sha256')},
                  models_loaded=False, qualification_inherited=False, NPU_used=False, Goodput=None)
    with receipt.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
