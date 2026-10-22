"""Freeze active64 successor with all immutable376 sources and fresh HOST checks."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-active64-whole-proposal-segment-limit-v398'
OLD = B / 'tree-best257-uniform-KV-buckets-full-v376'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs = [P / name for name in ['FREEZE398.json', 'SOURCE398.tar.gz', 'SOURCE398_RECEIPT.json']]
    assert not any(path.exists() for path in outputs)
    assert sha(OLD / 'FREEZE376.json') == 'ae723ddfddeb6d47f04605a3842de0c3690196b0addf39fa17c5bd1ca248765c'
    assert sha(OLD / 'SOURCE376.tar.gz') == 'd0102997fe19bc7c40bff8a0c91f128a553058c55f4d22488baf7be2b2fdeeea'
    for name in ['HOST398_RESULT.json', 'HOST398_SCOPE_RESULT.json',
                 'HOST398_CAPACITY_RESULT.json', 'HOST398_CONSTRUCTION_RESULT.json',
                 'HOST398_PACKING_RESULT.json', 'HOST398_LIVE_BINDING_RESULT.json']:
        value = json.loads((P / name).read_text())
        assert value['status'].startswith('PASS_HOST398') and value['NPU_used'] is False
    # Only module-local names changed in the numerical implementations.
    for name in ['bucket', 'dispatch', 'oracle', 'prefixes', 'projection', 'qualify']:
        previous = (OLD / f'{name}376.py').read_text()
        current = (P / f'{name}398.py').read_text()
        assert current == previous.replace('376', '398'), name
    parent = json.loads((OLD / 'FREEZE376.json').read_text())
    sources = dict(parent['files'])
    sources[str(OLD / 'FREEZE376.json')] = sha(OLD / 'FREEZE376.json')
    selected = [path for path in P.iterdir() if path.suffix in ['.py', '.sh', '.json', '.md']]
    selected.extend([Path(__file__).resolve()])
    for path in selected:
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        sources[str(path)] = sha(path)
    for name, expected in sources.items():
        assert sha(name) == expected, name
    value = dict(schema='ROOT398.best376-active64-source.v1', files=sources,
                 parent376_freeze_sha256=sha(OLD / 'FREEZE376.json'),
                 parent376_archive_sha256=sha(OLD / 'SOURCE376.tar.gz'),
                 active_request_ceiling=64, previous_active_request_ceiling=64, historical_best_tree_active_request_ceiling=48,
                 configured_max_num_seqs=64, target_max_requests=24,
                 target_candidate_budget=96, physical_target_rows=120, max_primary_side_attention_segments=24,
                 target_whole_proposal_packing_changed=True, original_controller_class_unmodified=True,
                 numerical_body_changes=False, original227_and_additional681_required=True,
                 original_warm_twozero_seal_and_full463_all3_ownAR_required=True,
                 Native_qualification_flags_assigned=False, deadline_seconds=10800,
                 model_or_performance_qualification_inherited=False, NPU_used=False,
                 Raw=None, Goodput=None)
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
                  NPU_used=False, model_qualified=False, Raw=None, Goodput=None)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
