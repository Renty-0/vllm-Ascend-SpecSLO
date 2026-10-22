"""Freeze only finite400 additions; retain the exact398 parent source archive."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-frugal399-guard-metadata-repair-v400'
PREVIOUS = B/'tree-M120-S32-frugal-independent-target-host-v399'
PROJECTOR = B/'tree-M120-S32-frugal-budget-projection-host-v397'
PARENT_FREEZE = '2b358bc162f42527eeeeeb9738fc9655a541e1a9e45353a0980530a95d0b0b37'
PARENT_ARCHIVE = '51fd4d110516e4d4516f520158a79060ad256568158d994f52ee83f47591d97d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs = [P/name for name in ('FREEZE400.json', 'SOURCE400.tar.gz', 'SOURCE400_RECEIPT.json')]
    assert not any(path.exists() for path in outputs)
    assert sha(PREVIOUS/'FREEZE399.json') == PARENT_FREEZE
    assert sha(PREVIOUS/'SOURCE399.tar.gz') == PARENT_ARCHIVE
    host = json.loads((P/'HOST400_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST400') and host['NPU_used'] is False
    assert host['models_constructed'] == host['device_tools_called'] == 0
    sources = {}
    selected = [path for root in (P,) for path in root.iterdir()
                if path.is_file() and path.suffix in ('.py', '.sh', '.json')]
    selected.append(Path(__file__).resolve())
    previous_files = json.loads((PREVIOUS/'FREEZE399.json').read_text())['files']
    for path in selected:
        assert str(path) not in previous_files
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        sources[str(path)] = sha(path)
    value = dict(schema='ROOT400.finite399-guard-metadata-repair-source.v1', files=sources,
                 parent399_freeze_sha256=PARENT_FREEZE, parent399_archive_sha256=PARENT_ARCHIVE,
                 numerical_sources_changed=False, original399_runtime_unchanged=True,
                 removed_duplicate_receipt_keywords=['action','numeric_source_obligations_only'],
                 configured_max_num_seqs=64, finite_normal_N=32, finite_C=88,
                 fresh200_cross_backend_cases_still_required=True,
                 NPU_used=False, models_qualified=False, full463=False,
                 production_permission=False, Raw=None, Goodput=None)
    freeze, archive, receipt = outputs
    with freeze.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    inventory = dict(sources, **{str(freeze):sha(freeze)})
    with archive.open('xb') as output:
        with gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as saved:
                for name in sorted(inventory):
                    path = Path(name)
                    info = saved.gettarinfo(str(path), arcname=name.lstrip('/'))
                    info.uid = info.gid = 0
                    info.uname = info.gname = 'root'
                    info.mtime = 0
                    with path.open('rb') as stream:
                        saved.addfile(info, stream)
    result = dict(freeze_sha256=sha(freeze), archive_sha256=sha(archive),
                  archive_bytes=archive.stat().st_size, archive_members=len(inventory),
                  parent_archive_reused=True, NPU_used=False,
                  models_qualified=False, Raw=None, Goodput=None)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
