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
PARENT = B/'tree-M160-C128-F64-global192-full-v333'
sys.path.insert(0, str(PARENT))
import guard333 as parent
from probe335 import configurations

LOCK = B/'.ROOT_NPU_EXECUTION.lock'
MAX_SECONDS = 300
MAX_TAG_BYTES = 16*1024**2
MAX_AGGREGATE_BYTES = 128*1024**2
ACTION = 'synthetic-FIA-Q4-Q5-group-query-invariance'
EXPECTED_FREEZE = None
EXPECTED_ARCHIVE = None
require = parent.require
sha = parent.sha


def prepared():
    require(sha(HERE/'FREEZE335.json') == EXPECTED_FREEZE, 'exact additive335 source pin')
    own = json.loads((HERE/'FREEZE335.json').read_text())
    for path, expected in own['files'].items():
        require(sha(path) == expected, 'immutable335 input '+path)
    parent_pins = json.loads((PARENT/'SOURCE333_RECEIPT.json').read_text())
    require(own['parent_freeze_sha256'] == parent_pins['freeze_sha256']
            and own['parent_archive_sha256'] == parent_pins['archive_sha256'], 'explicit parent333 source-only pins')
    parent.EXPECTED_FREEZE = parent_pins['freeze_sha256']
    parent.EXPECTED_ARCHIVE = parent_pins['archive_sha256']
    _, sources, parent_ready = parent.prepared()
    own_sources = dict(own['files'])
    own_sources[str(HERE/'FREEZE335.json')] = EXPECTED_FREEZE
    require(sha(HERE/'SOURCE335.tar.gz') == EXPECTED_ARCHIVE, 'exact additive335 archive')
    with tarfile.open(HERE/'SOURCE335.tar.gz', 'r:gz') as saved:
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
    helper = parent.donor_guard()
    actual = 0 if tag is None else helper.bytes_under(HERE/'actual'/tag)
    total = helper.bytes_under(HERE)
    require(actual <= MAX_TAG_BYTES and total <= MAX_AGGREGATE_BYTES, 'private synthetic data bound')
    return dict(actual_tag_bytes=actual, aggregate_private_bytes=total)


def launch_argv(tag):
    return ['/usr/bin/env', '-i', 'HOME=/root', 'USER=root', 'LOGNAME=root',
        'SHELL=/bin/bash', 'LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        '/bin/bash', str(B/'serial-v1/runtime.sh'), '/bin/bash', str(HERE/'startup335.sh'), tag]


def actual_audit(directory):
    path = Path(directory)/'result.json'
    value = json.loads(path.read_text())
    require(value['schema'] == 'ROOT335.synthetic-FIA-group-query-invariance.v1'
            and value['seed'] == 335 and value['source_sha256'] == sha(HERE/'probe335.py'), 'actual source-bound synthetic result')
    require(value['models_loaded'] is value['model_qualified'] is value['performance'] is False
            and value['Goodput'] is None, 'no model or service claim')
    rows = value['rows']
    expected = [(*item, phase) for item in configurations() for phase in ('A1', 'B', 'A2')]
    actual = [tuple(row[k] for k in ('width', 'groups', 'mode', 'mask_capacity', 'inner_precise', 'prefix', 'phase'))
              for row in rows]
    require(actual == expected and len(rows) == 864, 'all planned synthetic cases and ABA order')
    summaries = {}
    for row in rows:
        count = row['groups'] if row['mode'] == 'roots' else row['width']*row['groups']
        indices, differences = row['different_query_indices'], row['max_abs_by_query']
        require(row['M'] == row['width']*row['groups'] and row['compared_queries'] == count
                and row['identical_Q_K_V_visibility_and_page_table'] is True, 'actual synthetic metadata')
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in differences),
                'finite nonnegative actual differences')
        require(len(differences) == count and indices == [i for i, v in enumerate(differences) if v > 0]
                and row['all_compared_raw16_equal'] is (not indices), 'raw16 difference arithmetic')
        require(row['whole_output_restored'] is (True if row['phase'] == 'A2' else None), 'whole-output ABA restoration')
        key = f"Q{row['width']}-S{row['groups']}-{row['mode']}"
        summary = summaries.setdefault(key, dict(cases=0, noninvariant=0))
        summary['cases'] += 1
        summary['noninvariant'] += int(bool(indices))
    result = dict(status='completed_synthetic_diagnostic', rows=len(rows), groups=summaries,
        result_sha256=sha(path), models_loaded=False, model_qualified=False,
        performance=False, Goodput=None, adopted=False)
    return result, 'completed_synthetic_no_model_qualification', dict(ABA_complete=True,
        query_and_group_differences_recounted=True, Native_counters_claimed=False)


def execution_namespace():
    original = parent.donor_guard()
    namespace = dict(vars(original), P=HERE, G=HERE, ACTION=ACTION, prepared=prepared,
        storage=storage, launch_argv=launch_argv, actual_audit=actual_audit,
        MAX_SECONDS=MAX_SECONDS, MAX_TAG_BYTES=MAX_TAG_BYTES,
        MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(models_loaded=False, full463=False, formal_Goodput=False, adopted=False,
                   synthetic_Q4_and_Q5=True, groups=[24, 32, 33], rows=864))
    source = (B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    changes = (
        ('V154.root-owned-local-component-supervisor.v1', 'ROOT.V335.synthetic-FIA-supervisor.v1'),
        ('V154 supervisor received', 'ROOT335 supervisor received'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'synthetic Q4/Q5 group/query invariance; no models, service requests or performance'),
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
