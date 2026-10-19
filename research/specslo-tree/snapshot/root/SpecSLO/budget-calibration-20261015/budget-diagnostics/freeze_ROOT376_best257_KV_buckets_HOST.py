"""Source-only freeze for best257 plus uniform target KV extents."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-best257-uniform-KV-buckets-full-v376'
OLD = B / 'tree-static-q5-M120-SLO-first-full463-own-AR-v257'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    freeze, archive, receipt = (P / name for name in ('FREEZE376.json', 'SOURCE376.tar.gz', 'SOURCE376_RECEIPT.json'))
    assert not any(path.exists() for path in (freeze, archive, receipt))
    for name in ('HOST376_RESULT.json', 'HOST376_SCOPE_RESULT.json', 'HOST376_CONSTRUCTION_RESULT.json'):
        value = json.loads((P / name).read_text())
        assert value['status'].startswith('PASS_HOST376') and value['NPU_used'] is False
    assert sha(OLD / 'FREEZE257.json') == 'b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
    assert sha(OLD / 'SOURCE257.tar.gz') == 'f268e73e73b334f29dba4da8413a9a9efc6f4bbcabdbd975554c801f6e34e8b0'
    original = json.loads((OLD / 'FREEZE257.json').read_text())
    files = dict(original['files'])
    files[str(OLD / 'FREEZE257.json')] = sha(OLD / 'FREEZE257.json')
    selected = [path for path in P.iterdir() if path.suffix in ('.py', '.sh', '.json')]
    selected.extend([Path(__file__).resolve(), B / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344/domain344.py'])
    for path in selected:
        if path.suffix == '.py':
            ast.parse(path.read_text())
        elif path.suffix == '.sh':
            subprocess.run(['/bin/bash', '-n', str(path)], check=True)
        files[str(path)] = sha(path)
    for name, expected in files.items():
        assert sha(name) == expected, name
    # The configuration of the NN models/operators remains the exact best257
    # shell configuration; only source lookup/pinning changes are permitted.
    def model_environment(path):
        return [line for line in path.read_text().splitlines()
                if line.startswith(('export VLLM_', 'unset VLLM_'))]
    assert model_environment(P / 'startup376.sh') == model_environment(OLD / 'startup257.sh')
    value = dict(schema='ROOT376.best257-plus-target-uniform-KV-source.v1',
        files=files, original_best257_freeze_sha256=sha(OLD / 'FREEZE257.json'),
        original_best257_archive_sha256=sha(OLD / 'SOURCE257.tar.gz'),
        original227_fixed_FULL_cases=True, additional_current_target_cases=681,
        original_optimized_RootSeed_FastV5_draft_unchanged=True,
        original_allocator_admission_arrivals_publication_clocks_unchanged=True,
        original_warm_twozero_seal_and_full463_all3_ownAR_required=True,
        Native_qualification_flag_assignments=False, deadline_seconds=10800,
        NPU_used=False, model_or_performance_qualification_inherited=False, Raw=None, Goodput=None)
    with freeze.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    inventory = dict(files, **{str(freeze): sha(freeze)})
    with archive.open('xb') as output:
        with gzip.GzipFile(fileobj=output, filename='', mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as saved:
                for name in sorted(inventory):
                    path = Path(name)
                    info = saved.gettarinfo(str(path), arcname=name.lstrip('/'))
                    info.uid = info.gid = 0
                    info.uname = info.gname = 'root'
                    info.mtime = 0
                    with path.open('rb') as source:
                        saved.addfile(info, source)
    result = dict(freeze_sha256=sha(freeze), archive_sha256=sha(archive),
        archive_members=len(inventory), archive_bytes=archive.stat().st_size,
        NPU_used=False, model_qualified=False, Raw=None, Goodput=None)
    with receipt.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
