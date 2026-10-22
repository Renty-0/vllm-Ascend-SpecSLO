"""Freeze a finite N24 role/repair component; device qualification is not inherited."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-parent4-normal24-SLO-mixed-protocol-v419'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs = [P / name for name in ['FREEZE419.json', 'SOURCE419.tar.gz', 'SOURCE419_RECEIPT.json']]
    assert not any(path.exists() for path in outputs)
    sys.path.insert(0, str(P))
    import guard419
    guard419.parent.EXPECTED_FREEZE = guard419.PARENT_FREEZE
    guard419.parent.EXPECTED_ARCHIVE = guard419.PARENT_ARCHIVE
    _, original_sources, _ = guard419.parent.prepared()
    host = json.loads((P / 'HOST419_SCOPE_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST419') and host['NPU_used'] is False
    assert host['Torch_imported'] is False and host['global_max_num_seqs'] == 64
    assert len(host['cases']) == 25
    source_names = [path for path in P.iterdir() if path.suffix in ['.py', '.sh', '.json']]
    source_names.append(Path(__file__).resolve())
    source_names.append(B/'budget-diagnostics/check_ROOT419_real_startup_construction_HOST.py')
    sources = {}
    for path in source_names:
        assert str(path) not in original_sources
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        sources[str(path)] = sha(path)
    value = dict(schema='ROOT419.finite-N24-E4-R0to24-current-M120-protocol.v1', files=sources,
                 parent395_freeze_sha256=guard419.PARENT_FREEZE,
                 parent395_archive_sha256=guard419.PARENT_ARCHIVE,
                 configured_max_num_seqs=64, finite_normal_N=24, parent_N=24,
                 parent_B=4, child_B=4, E=[4], R=list(range(25)),
                 fresh_original173_baseline_required=True, original391_current_M120_checks_required=True,
                 early_observer392_retained=True, canonical_source_bound419=True, SLO_priority_actual_mixed_required=True,
                 Native_qualification_flags_assigned=False,
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
