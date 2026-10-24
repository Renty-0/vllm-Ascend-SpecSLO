"""Owned full lifecycle with corrected completed.audit reader and cohort48."""
import argparse
import ast
import copy
import fcntl
import hashlib
import inspect
import re
import tarfile
from check466 import P, BASE, PARENT_FREEZE, PARENT_ARCHIVE, PHYSICAL, ready, sha
import guard465 as parent

DONOR, ACTION, LOCK = parent.DONOR, parent.ACTION, parent.LOCK
MAX_SECONDS = parent.MAX_SECONDS
MAX_TAG_BYTES = parent.MAX_TAG_BYTES
MAX_AGGREGATE_BYTES = parent.MAX_AGGREGATE_BYTES
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, files, previous = parent.prepared()
    own = ready(EXPECTED_FREEZE)
    extra = dict(own['files'])
    extra[str(P/'FREEZE466.json')] = EXPECTED_FREEZE
    if sha(P/'SOURCE466.tar.gz') != EXPECTED_ARCHIVE:
        raise RuntimeError('Exact466 source archive required')
    with tarfile.open(P/'SOURCE466.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(extra) or {m.name for m in members} != {p.lstrip('/') for p in extra}:
            raise RuntimeError('Exact466 archive inventory required')
        for member in members:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != extra['/'+member.name]:
                raise RuntimeError('Exact466 archive bytes required')
    extra[str(P/'SOURCE466.tar.gz')] = EXPECTED_ARCHIVE
    for path, wanted in extra.items():
        if path in files and files[path] != wanted:
            raise RuntimeError('Identical inherited source required')
        files[path] = wanted
    return own, files, dict(parent465=previous, own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, physical_cards=list(PHYSICAL),
        maximum_new_cohort=48, global_active_limit=64, original_model_callbacks_unchanged=True,
        maximum_empty_prefill_cohort=48, maximum_nonempty_prefill_cohort=32,
        terminal_audit_interfaces_corrected=True, NN_permission_inherited=False)


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += sum(f.stat().st_size for f in P.rglob('*') if f.is_file())
    if value['aggregate_private_bytes'] > MAX_AGGREGATE_BYTES:
        raise RuntimeError('Unchanged private storage bound exceeded')
    return value


def launch_argv(tag):
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    args = parent.launch_argv(tag)
    old = str(BASE/'launch465.sh')
    if args.count(old) != 1:
        raise RuntimeError('One original465 launch entry required')
    args = [str(P/'launch466.sh') if value == old else value for value in args]
    args.insert(args.index('/bin/bash'), 'SPECSLO_V466_ROOT_FREEZE_SHA='+EXPECTED_FREEZE)
    return args


def constructed_audit():
    tree = ast.parse(inspect.getsource(parent.actual_audit))
    before = ast.dump(tree, include_attributes=False)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and ast.unparse(n.targets[0]) == 'completed']
    if len(assignments) != 1:
        raise RuntimeError('Exact completed-document consumer required')
    node = assignments[0]
    value = node.value
    node.value = ast.Subscript(value=value, slice=ast.Constant('audit'), ctx=ast.Load())
    changed = copy.deepcopy(tree)
    node.value = value
    if ast.dump(tree, include_attributes=False) != before:
        raise RuntimeError('Full guard audit body inverse required')
    namespace = dict(parent.actual_audit.__globals__)
    exec(compile(ast.fix_missing_locations(changed), __file__+'::completed-audit-field', 'exec'), namespace)
    return namespace['actual_audit']


def actual_audit(directory):
    result = constructed_audit()(directory)
    rows = result['ROOT463_actual_credit_cost_admission']
    if not rows or any(row['admission']['maximum_new_cohort'] != 48 for row in rows):
        raise RuntimeError('Every actual trace must use declared cohort48')
    return result


def execution_namespace():
    namespace = parent.execution_namespace()
    namespace.update(prepared=prepared, storage=storage, launch_argv=launch_argv, actual_audit=actual_audit)
    namespace['execute_locked'] = parent.parent.replace_constants(namespace['execute_locked'], (
        ('ROOT.V465.current-component-credit-cost-full-own-AR-supervisor.v1',
         'ROOT.V466.current-card-cohort48-full-own-AR-supervisor.v1'),
    ), namespace)
    namespace['SCOPE'] = dict(namespace['SCOPE'], maximum_new_cohort=48,
                              maximum_new_prefill_cohort=48,
                              maximum_empty_prefill_cohort=48,
                              maximum_nonempty_prefill_cohort=32,
                              original463_algorithm_unchanged=False,
                              original_credit_formula_and_nonempty_ceiling_unchanged=True,
                              terminal_audit_interfaces_corrected=True)
    if namespace['execute_locked'].__globals__ is not namespace:
        raise RuntimeError('Actual owned lifecycle namespace required')
    return namespace


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = args.expected_freeze_sha256, args.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) or any(
            not re.fullmatch('[a-f0-9]{64}', value) for value in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)):
        raise RuntimeError('Safe fresh tag and explicit source pins required')
    prepared()
    namespace = execution_namespace()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
