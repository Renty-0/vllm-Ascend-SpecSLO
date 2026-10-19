"""Own bounded launch; preserve original209 output, verifier and cleanup roots."""
import argparse
import ast
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
B = HERE.parent
PARENT = B/'tree-Q5-bucket-causal-inner1-target-math-cost-v341'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path.insert(0, str(PARENT))
import guard341 as parent
import guard336 as numerical_guard
import guard333 as original_owner
require, sha = parent.require, parent.sha
LOCK = B/'.ROOT_NPU_EXECUTION.lock'
MAX_SECONDS = 1800
MAX_TAG_BYTES = 96*1024**2
MAX_AGGREGATE_BYTES = 2560*1024**2
EXPECTED_FREEZE = None
EXPECTED_ARCHIVE = None


def prepared():
    require(sha(HERE/'FREEZE342.json') == EXPECTED_FREEZE, 'explicit342 additive source pin')
    own = json.loads((HERE/'FREEZE342.json').read_text())
    for path, expected in own['files'].items():
        require(sha(path) == expected, 'immutable342 source '+path)
    pins = json.loads((PARENT/'SOURCE341_RECEIPT.json').read_text())
    require(own['parent_freeze_sha256'] == pins['freeze_sha256']
            and own['parent_archive_sha256'] == pins['archive_sha256'], 'unchanged frozen336 numeric parent')
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = pins['freeze_sha256'], pins['archive_sha256']
    _, sources, previous = parent.prepared()
    additive = dict(own['files'])
    additive[str(HERE/'FREEZE342.json')] = EXPECTED_FREEZE
    require(sha(HERE/'SOURCE342.tar.gz') == EXPECTED_ARCHIVE, 'explicit342 additive archive pin')
    with tarfile.open(HERE/'SOURCE342.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(additive) and {m.name for m in members}
                == {p.lstrip('/') for p in additive}, 'exact342 additive archive inventory')
        for member in members:
            require(member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                    == additive['/'+member.name], 'exact342 archived source bytes')
    require(not set(sources) & set(additive), 'disjoint342 additive sources')
    sources.update(additive)
    return own, sources, dict(own_freeze_sha256=EXPECTED_FREEZE, own_archive_sha256=EXPECTED_ARCHIVE,
        own_archive_members=len(additive), source_members=len(sources),
        parent_numeric_source_verified_in_place=previous,
        device_qualification_inherited=False, output_root=str(DONOR))


def storage(tag=None):
    original = original_owner.donor_guard()
    actual = 0 if tag is None else original.bytes_under(DONOR/'actual'/tag)
    total = original.bytes_under(DONOR)+original.bytes_under(HERE)
    require(actual <= MAX_TAG_BYTES and total <= MAX_AGGREGATE_BYTES, 'private bounded342 diagnostic')
    return dict(actual_tag_bytes=actual, aggregate_private_bytes=total)


def launch_argv(tag):
    return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash',
        'LANG=C.UTF-8','PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        'SPECSLO_V342_ROOT_FREEZE_SHA='+EXPECTED_FREEZE,
        'SPECSLO_V218_ROOT_FREEZE_SHA=fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260',
        '/bin/bash',str(B/'serial-v1/runtime.sh'),'/bin/bash',str(HERE/'startup342.sh'),tag]


def actual_audit(directory):
    from audit342 import audit
    return audit(directory)


def execution_namespace():
    source = inspect.getsource(numerical_guard.execution_namespace)
    projected = source.replace('ROOT.V336.linear-target-shape-supervisor.v1',
                               'ROOT.V342.original-root-linear-target-supervisor.v1')
    projected = projected.replace('ROOT336 supervisor received', 'ROOT342 supervisor received')
    tree = ast.parse(projected)
    fields = [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == 'models_loaded']
    require(len(fields) == 1, 'one planned-model declaration, not actual startup evidence')
    fields[0].arg = 'models_requested'
    for field in ast.walk(tree):
        if isinstance(field,ast.keyword)and field.arg=='finite_cases':field.value=ast.Constant(28)
    namespace = dict(vars(numerical_guard), HERE=HERE, __file__=__file__, prepared=prepared, actual_audit=actual_audit,
        launch_argv=launch_argv, storage=storage, MAX_SECONDS=MAX_SECONDS,
        MAX_TAG_BYTES=MAX_TAG_BYTES, MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES)
    exec(compile(ast.fix_missing_locations(tree), __file__+'::retained-owned-guard-factory', 'exec'), namespace)
    result = namespace['execution_namespace']()
    result.update(P=DONOR, G=HERE, prepared=prepared, storage=storage, launch_argv=launch_argv,
                  MAX_SECONDS=MAX_SECONDS, MAX_TAG_BYTES=MAX_TAG_BYTES,
                  MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES)
    return result


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = args.expected_freeze_sha256, args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None
            and all(re.fullmatch('[0-9a-f]{64}', s) for s in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)), 'safe342 tag/pins')
    prepared()
    namespace = execution_namespace()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
