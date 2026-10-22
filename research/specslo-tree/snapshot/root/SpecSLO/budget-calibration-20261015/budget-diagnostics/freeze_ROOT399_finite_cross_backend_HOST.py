"""Freeze only finite399 additions; retain the exact398 parent source archive."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-M120-S32-frugal-independent-target-host-v399'
PREVIOUS = B/'tree-active64-whole-proposal-segment-limit-v398'
PROJECTOR = B/'tree-M120-S32-frugal-budget-projection-host-v397'
PARENT_FREEZE = 'dfb6cfd44d2f524d7ad28892ec055d50d1035910fb71dbbc5486dc060cb5e73d'
PARENT_ARCHIVE = '5c79ae661393d9bb347dd6f53ff447d34e09fa121bc90d4d43844cdb63357f6a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs = [P/name for name in ('FREEZE399.json', 'SOURCE399.tar.gz', 'SOURCE399_RECEIPT.json')]
    assert not any(path.exists() for path in outputs)
    assert sha(PREVIOUS/'FREEZE398.json') == PARENT_FREEZE
    assert sha(PREVIOUS/'SOURCE398.tar.gz') == PARENT_ARCHIVE
    scope = json.loads((P/'HOST399_RESULT.json').read_text())
    startup = json.loads((P/'HOST399_PRECONSTRUCTOR_RESULT.json').read_text())
    assert scope['status'].startswith('PASS_HOST399') and scope['cases'] == 200
    assert scope['NPU_used'] is startup['NPU_used'] is False
    assert startup['model_constructor_calls'] == startup['NPU_lazy_init_calls'] == 0
    sources = {}
    selected = [path for root in (P, PROJECTOR) for path in root.iterdir()
                if path.is_file() and path.suffix in ('.py', '.sh', '.json')]
    selected.append(Path(__file__).resolve())
    previous_files = json.loads((PREVIOUS/'FREEZE398.json').read_text())['files']
    for path in selected:
        assert str(path) not in previous_files
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        sources[str(path)] = sha(path)
    value = dict(schema='ROOT399.finite-S32-canonical-S24-math-source.v1', files=sources,
                 parent398_freeze_sha256=PARENT_FREEZE, parent398_archive_sha256=PARENT_ARCHIVE,
                 source_qualification_inherited=False, configured_max_num_seqs=64,
                 finite_N=list(range(25, 33)), physical_M=120, physical_S=32,
                 independent_reference_M=120, independent_reference_S=24,
                 independent_neural_prefix_owners=96,
                 cases_per_target=200, KV_extents=[512,1024,1536,2048,2560],
                 original_full_components_executed=False, full463=False, whole_AR=False,
                 Native_qualification_flags_assigned=False, production_permission=False,
                 NPU_used=False, Raw=None, Goodput=None)
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
