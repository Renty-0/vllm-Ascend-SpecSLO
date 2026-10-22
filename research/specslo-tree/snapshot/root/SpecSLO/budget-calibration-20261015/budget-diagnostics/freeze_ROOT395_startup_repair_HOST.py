"""Freeze a launch-only successor; preserve all391 numerical source bytes."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-parent4-current-M120-import-repair-v395'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs = [P / name for name in ['FREEZE395.json', 'SOURCE395.tar.gz', 'SOURCE395_RECEIPT.json']]
    assert not any(path.exists() for path in outputs)
    sys.path.insert(0, str(P))
    import guard395
    guard395.parent.EXPECTED_FREEZE = guard395.PARENT_FREEZE
    guard395.parent.EXPECTED_ARCHIVE = guard395.PARENT_ARCHIVE
    _, original_sources, _ = guard395.parent.prepared()
    host = json.loads((P / 'HOST395_STARTUP_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST395') and host['NPU_used'] is False
    assert host['model_constructors_called'] == 0 and host['configured_max_num_seqs'] == 64
    source_names = [path for path in P.iterdir() if path.suffix in ['.py', '.sh', '.json']]
    source_names.append(Path(__file__).resolve())
    sources = {}
    for path in source_names:
        assert str(path) not in original_sources
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        sources[str(path)] = sha(path)
    value = dict(schema='ROOT395.current-M120-parent4-startup-path-only.v1', files=sources,
                 parent391_freeze_sha256=guard395.PARENT_FREEZE,
                 parent391_archive_sha256=guard395.PARENT_ARCHIVE,
                 original391_numerical_bodies_changed=False,
                 added_dependency_directories=json.loads((P / 'REPAIR395.json').read_text())['added_search_directories'],
                 configured_max_num_seqs=64, finite_normal_N=16, parent_N=16,
                 parent_B=4, child_B=4, E=[1, 2, 3, 4],
                 fresh_original173_baseline_required=True, original391_current_M120_checks_required=True,
                 early_observer392_retained=True, Native_qualification_flags_assigned=False,
                 full463=False, production_permission=False, NPU_used=False, Raw=None, Goodput=None)
    freeze, archive, receipt = outputs
    with freeze.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    inventory = dict(sources, **{str(freeze): sha(freeze)})
    with archive.open('xb') as output:
        with gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as saved:
                for name in sorted(inventory):
                    path = Path(name)
                    entry = saved.gettarinfo(str(path), arcname=name.lstrip('/'))
                    entry.uid = entry.gid = 0
                    entry.uname = entry.gname = 'root'
                    entry.mtime = 0
                    with path.open('rb') as stream:
                        saved.addfile(entry, stream)
    result = dict(freeze_sha256=sha(freeze), archive_sha256=sha(archive),
                  archive_members=len(inventory), archive_bytes=archive.stat().st_size,
                  NPU_used=False, models_qualified=False, Raw=None, Goodput=None)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
