"""Original463 lifecycle, checks and cleanup on current physical0/1/2/7."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import re
import tarfile
import types

from check464 import P, B, BASE, PARENT_FREEZE, PARENT_ARCHIVE, PHYSICAL, ready, sha
import guard463 as parent

DONOR, ACTION, LOCK = parent.DONOR, parent.ACTION, parent.LOCK
MAX_SECONDS, MAX_TAG_BYTES, MAX_AGGREGATE_BYTES = parent.MAX_SECONDS, parent.MAX_TAG_BYTES, parent.MAX_AGGREGATE_BYTES
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, files, previous = parent.prepared()
    own = ready(EXPECTED_FREEZE)
    extra = dict(own['files'])
    extra[str(P / 'FREEZE464.json')] = EXPECTED_FREEZE
    if sha(P / 'SOURCE464.tar.gz') != EXPECTED_ARCHIVE:
        raise RuntimeError('Exact464 source archive required')
    with tarfile.open(P / 'SOURCE464.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(extra) or {m.name for m in members} != {p.lstrip('/') for p in extra}:
            raise RuntimeError('Exact464 archive inventory required')
        for member in members:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != extra['/' + member.name]:
                raise RuntimeError('Exact464 archive bytes required')
    for path, wanted in extra.items():
        if path in files and files[path] != wanted:
            raise RuntimeError('Identical inherited source required')
        files[path] = wanted
    return own, files, dict(parent463=previous, own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, physical_cards=list(PHYSICAL),
        original463_algorithm_unchanged=True, NN_permission_inherited=False,
        global_active_limit=64, complete_fresh_qualification_required=True)


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += sum(f.stat().st_size for f in P.rglob('*') if f.is_file())
    if value['aggregate_private_bytes'] > MAX_AGGREGATE_BYTES:
        raise RuntimeError('Unchanged aggregate private bound exceeded')
    return value


def launch_argv(tag):
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    args = parent.launch_argv(tag)
    old = str(BASE / 'launch463.sh')
    if args.count(old) != 1:
        raise RuntimeError('One original463 launch entry required')
    args = [str(P / 'launch464.sh') if value == old else value for value in args]
    args.insert(args.index('/bin/bash'), 'SPECSLO_V464_ROOT_FREEZE_SHA=' + EXPECTED_FREEZE)
    return args


def replace_constants(function, changes, namespace=None):
    """Change only declared control metadata; executable bytecode is identical."""
    constants = list(function.__code__.co_consts)
    for old, new in changes:
        matches = [i for i, value in enumerate(constants) if type(value) is type(old) and value == old]
        if len(matches) != 1:
            raise RuntimeError('One declared control constant required: ' + repr(old))
        constants[matches[0]] = new
    code = function.__code__.replace(co_consts=tuple(constants))
    return types.FunctionType(code, function.__globals__ if namespace is None else namespace,
        function.__name__, function.__defaults__, function.__closure__)


def bind_helpers(original):
    helpers = original()
    old_helper = helpers['helper']

    def helper():
        value = old_helper()
        if value.PHYSICAL != (2, 5, 6, 7):
            raise RuntimeError('Original isolated physical-card helper required')
        value.PHYSICAL = PHYSICAL
        return value

    helpers['helper'] = helper
    helpers['verify'] = replace_constants(helpers['verify'], (
        ('serial-v1/runtime.sh', 'tree-current-card-rebind-full-v464/environment464.sh'),
    ))
    return helpers


actual_audit = parent.actual_audit


def execution_namespace():
    ns = parent.execution_namespace()
    original_helpers = ns['original_helpers']
    ns.update(prepared=prepared, storage=storage, launch_argv=launch_argv,
        actual_audit=actual_audit, original_helpers=lambda: bind_helpers(original_helpers))
    ns['execute_locked'] = replace_constants(ns['execute_locked'], (
        ('ROOT.V463.credit-cost-admission-full-own-AR-supervisor.v1',
         'ROOT.V464.current-card-credit-cost-full-own-AR-supervisor.v1'),
        ((2, 5, 6, 7), PHYSICAL),
    ), ns)
    ns['SCOPE'].update(current_physical_cards=list(PHYSICAL), previous_physical_cards=[2, 5, 6, 7],
        original463_algorithm_unchanged=True, cross_card_history_not_a_matched_comparison=True)
    if ns['execute_locked'].__globals__ is not ns:
        raise RuntimeError('Actual owned lifecycle namespace required')
    return ns


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
        raise RuntimeError('Safe tag and explicit source pins required')
    prepared()
    ns = execution_namespace()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ns['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
