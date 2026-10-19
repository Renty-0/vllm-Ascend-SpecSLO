"""Single-owner bounded synthetic diagnostic with original environment checks."""
import argparse
import ast
import copy
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
B = HERE.parent
PARENT = B/'tree-FIA-Q4-group-and-query-isolation-v335'
sys.path.insert(0, str(PARENT))
import guard335 as parent

LOCK = B/'.ROOT_NPU_EXECUTION.lock'
MAX_SECONDS = 300
MAX_TAG_BYTES = 16*1024**2
MAX_AGGREGATE_BYTES = 128*1024**2
ACTION = 'synthetic-Q5-causal-uniform-KV-raw-screen'
EXPECTED_FREEZE = None
EXPECTED_ARCHIVE = None
require = parent.require
sha = parent.sha


def prepared():
    require(sha(HERE/'FREEZE340.json') == EXPECTED_FREEZE, 'exact additive340 source pin')
    own = json.loads((HERE/'FREEZE340.json').read_text())
    for path, expected in own['files'].items():
        require(sha(path) == expected, 'immutable340 input '+path)
    parent_pins = json.loads((PARENT/'SOURCE335_RECEIPT.json').read_text())
    require(own['parent_freeze_sha256'] == parent_pins['freeze_sha256']
            and own['parent_archive_sha256'] == parent_pins['archive_sha256'], 'explicit parent333 source-only pins')
    parent.EXPECTED_FREEZE = parent_pins['freeze_sha256']
    parent.EXPECTED_ARCHIVE = parent_pins['archive_sha256']
    _, sources, parent_ready = parent.prepared()
    own_sources = dict(own['files'])
    own_sources[str(HERE/'FREEZE340.json')] = EXPECTED_FREEZE
    require(sha(HERE/'SOURCE340.tar.gz') == EXPECTED_ARCHIVE, 'exact additive340 archive')
    with tarfile.open(HERE/'SOURCE340.tar.gz', 'r:gz') as saved:
        members = saved.getmembers()
        require({m.name for m in members} == {p.lstrip('/') for p in own_sources}
                and len(members) == len(own_sources), 'exact additive archive inventory')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest()
                    == own_sources['/'+member.name], 'exact additive archive bytes')
    require(not (set(sources) & set(own_sources)), 'disjoint immutable additive files')
    sources.update(own_sources)
    return own, sources, dict(own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, own_archive_members=len(own_sources),
        parent_source_archive_verified_in_place=parent_ready,
        source_members=len(sources), model_qualification_inherited=False)


def storage(tag=None):
    helper = parent.parent.donor_guard()
    actual = 0 if tag is None else helper.bytes_under(HERE/'actual'/tag)
    total = helper.bytes_under(HERE)
    require(actual <= MAX_TAG_BYTES and total <= MAX_AGGREGATE_BYTES, 'private synthetic data bound')
    return dict(actual_tag_bytes=actual, aggregate_private_bytes=total)


def launch_argv(tag):
    return ['/usr/bin/env', '-i', 'HOME=/root', 'USER=root', 'LOGNAME=root',
        'SHELL=/bin/bash', 'LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        '/bin/bash', str(B/'serial-v1/runtime.sh'), '/bin/bash', str(HERE/'startup340.sh'), tag]


def actual_audit(directory):
    from audit340 import audit
    result = audit(Path(directory)/'result.json', expected_source=sha(HERE/'probe340.py'))
    return result, 'completed_synthetic_no_model_qualification', dict(all_rows_and_costs_recomputed=True)


def execution_namespace():
    original = parent.parent.donor_guard()
    namespace = dict(vars(original), P=HERE, G=HERE, ACTION=ACTION, prepared=prepared,
        storage=storage, launch_argv=launch_argv, actual_audit=actual_audit,
        MAX_SECONDS=MAX_SECONDS, MAX_TAG_BYTES=MAX_TAG_BYTES,
        MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(models_loaded=False, full463=False, formal_Goodput=False, adopted=False,
                   synthetic_Q5=True, groups=[24], model_or_service_performance=False))
    source = (B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    changes = (
        ('V154.root-owned-local-component-supervisor.v1', 'ROOT.V340.synthetic-FIA-supervisor.v1'),
        ('V154 supervisor received', 'ROOT340 supervisor received'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'synthetic Q5 FULL/causal and uniform shorter KV arithmetic/cost; no models or service metrics'),
        ('bounded1800second', 'bounded300second'), ('within1800seconds', 'within300seconds'),
        ("'independent_native_discovery_arithmetic'", "'independent_synthetic_arithmetic'"))
    for before, after in changes:
        require(source.count(before) == 1, 'exact original metadata anchor '+before)
        source = source.replace(before, after)
    names = {'verifier_report', 'child_limits', 'execute_locked'}
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, 'original owned supervision bodies retained')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-owned-supervision', 'exec'), namespace)
    return namespace


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = args.expected_freeze_sha256, args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None
            and all(re.fullmatch('[0-9a-f]{64}', s) for s in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)), 'explicit safe tag/pins')
    prepared()
    namespace = execution_namespace()
    lock = LOCK.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
