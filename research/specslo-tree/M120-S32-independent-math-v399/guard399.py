"""Owned finite new-target diagnostic; serving qualifications are never inherited."""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
B = HERE.parent
PREVIOUS = B/'tree-active64-whole-proposal-segment-limit-v398'
sys.path.insert(0, str(PREVIOUS))
import guard398 as parent

PARENT_FREEZE = 'dfb6cfd44d2f524d7ad28892ec055d50d1035910fb71dbbc5486dc060cb5e73d'
PARENT_ARCHIVE = '5c79ae661393d9bb347dd6f53ff447d34e09fa121bc90d4d43844cdb63357f6a'
ACTION399 = 'frugal-C88-M120-S32-cross-S24-node-component'
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None
require = parent.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def own_archive():
    freeze, archive = HERE/'FREEZE399.json', HERE/'SOURCE399.tar.gz'
    require(sha(freeze) == EXPECTED_FREEZE and sha(archive) == EXPECTED_ARCHIVE,
            'explicit399 frozen source and archive pins')
    value = json.loads(freeze.read_text())
    files = dict(value['files'], **{str(freeze):EXPECTED_FREEZE})
    with tarfile.open(archive, 'r:gz') as saved:
        members = saved.getmembers()
        require(len(members) == len(files) and len({m.name for m in members}) == len(files),
                'exact399 unique archive members')
        expected = {name.lstrip('/'):digest for name, digest in files.items()}
        require({m.name for m in members} == set(expected), 'complete399 bounded member names')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest()
                    == expected[member.name] == sha('/'+member.name), 'actual399 archived source '+member.name)
    return value, files


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, sources, inherited = parent.prepared()
    own, extra = own_archive()
    require(own['parent398_freeze_sha256'] == PARENT_FREEZE
            and own['parent398_archive_sha256'] == PARENT_ARCHIVE, 'exact399 parent source')
    require(not set(sources) & set(extra), 'disjoint399 finite extra source closure')
    sources.update(extra)
    return own, sources, dict(parent=inherited, source_members=len(sources),
                             own_freeze_sha256=EXPECTED_FREEZE, own_archive_sha256=EXPECTED_ARCHIVE,
                             device_qualification_inherited=False, new_finite_cases_per_target=200)


def launch_argv(tag):
    args = parent.launch_argv(tag)
    old = str(PREVIOUS/'launch398.sh')
    require(args.count(old) == 1, 'one exact399 parent launcher')
    action = 'causal80-N9-to15-allB4-empty-active32-full463'
    require(args.count(action) == 1, 'one exact399 action replacement')
    return [str(HERE/'launch399.sh') if value == old else ACTION399 if value == action else value
            for value in args]


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    require(value['aggregate_private_bytes'] <= 3584*1024**2, 'finite399 unchanged aggregate storage bound')
    return value


def actual_audit(directory):
    sys.path.insert(0, str(HERE))
    from audit399 import audit
    result = audit(directory)
    completed = json.loads((Path(directory)/'completed.json').read_text())
    require(completed['audit'] == result, 'actual399 producer and independent retained audits agree')
    require(completed['replies'] == [['calibrated', r, dict(numeric_cases=200, performance=False)]
                                    for r in range(4)], 'all four real finite workers complete')
    require(json.loads((Path(directory)/'worker-exit.json').read_text())
            == dict(closed=True, all_exited=True), 'actual399 owned worker exit')
    return result, 'finite_S32_to_independent_S24_math_and_prepared_graph_cost_only', dict(
        full463=False, whole_workload_AR=False, hidden_KV_reexecuted_by_HOST=False)


def execution_namespace():
    namespace = parent.execution_namespace()
    namespace.update(prepared=prepared, launch_argv=launch_argv, storage=storage,
                     actual_audit=actual_audit, G=HERE, ACTION=ACTION399)
    namespace['SCOPE'] = dict(action=ACTION399, numeric_source_obligations_only=True,
        active_request_limit=64, configured_max_num_seqs=64, effective_requests=32,
        configured_requests=32, effective_candidates=88, configured_candidates=96,
        physical_M=120, physical_S=32, independent_reference_S=24,
        finite_cases_per_target=200, original201_44_114_and_all_original_components_required=False,
        original_full_components_executed=False, model_qualification_inherited=False,
        full463=False, postfull_own_AR_required=False, production_permission=False,
        Raw=None, Goodput=None, formal_Goodput=False, adopted=False)
    source = (B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    edits = (
        ('V154.root-owned-local-component-supervisor.v1',
         'ROOT.V399.finite-frugal-S32-independent-S24-supervisor.v1'),
        ('V154 supervisor received', 'ROOT399 supervisor received'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'finite200 S32 cases against independently prefilled canonical S24 paths; strict hidden/all128KV/greedy; prepared-graph costs only; no original full components, full463, ownAR or production qualification'),
        ('bounded1800second', 'bounded10800second'),
        ('within1800seconds', 'within10800seconds'),
        ("'actual_component_audit'", "'actual_finite399_audit'"),
        ("'actual_local_coverage_status'", "'actual_finite399_status'"),
    )
    for before, after in edits:
        require(source.count(before) == 1, 'exact399 original supervision metadata anchor '+before)
        source = source.replace(before, after)
    names = {'verifier_report', 'child_limits', 'execute_locked'}
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, 'all399 original owned execution/cleanup bodies')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::finite-original-owned-supervision', 'exec'), namespace)
    require(namespace['execute_locked'].__globals__ is namespace, 'actual399 owned supervision globals')
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
            and all(re.fullmatch('[0-9a-f]{64}', v) for v in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)),
            'safe399 fresh tag and explicit source pins')
    prepared()
    namespace = execution_namespace()
    lock_path = B/'.ROOT_NPU_EXECUTION.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
