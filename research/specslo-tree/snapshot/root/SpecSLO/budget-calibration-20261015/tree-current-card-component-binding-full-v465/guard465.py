"""Unchanged464 owned lifecycle, with the corrected inner component route."""
import argparse
import fcntl
import hashlib
import re
import tarfile
from check465 import P, B, BASE, PARENT_FREEZE, PARENT_ARCHIVE, PHYSICAL, ready, sha
import guard464 as parent

DONOR, ACTION, LOCK = parent.DONOR, parent.ACTION, parent.LOCK
MAX_SECONDS, MAX_TAG_BYTES, MAX_AGGREGATE_BYTES = parent.MAX_SECONDS, parent.MAX_TAG_BYTES, parent.MAX_AGGREGATE_BYTES
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, files, previous = parent.prepared()
    own = ready(EXPECTED_FREEZE)
    extra = dict(own['files'])
    extra[str(P / 'FREEZE465.json')] = EXPECTED_FREEZE
    if sha(P / 'SOURCE465.tar.gz') != EXPECTED_ARCHIVE:
        raise RuntimeError('Exact465 source archive required')
    with tarfile.open(P / 'SOURCE465.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(extra) or {m.name for m in members} != {p.lstrip('/') for p in extra}:
            raise RuntimeError('Exact465 archive inventory required')
        for member in members:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != extra['/' + member.name]:
                raise RuntimeError('Exact465 archive bytes required')
    for path, wanted in extra.items():
        if path in files and files[path] != wanted:
            raise RuntimeError('Identical inherited source required')
        files[path] = wanted
    return own, files, dict(parent464=previous, own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, physical_cards=list(PHYSICAL),
        original463_algorithm_unchanged=True, current_internal_component_binding=True,
        NN_permission_inherited=False, global_active_limit=64, complete_fresh_qualification_required=True)


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += sum(f.stat().st_size for f in P.rglob('*') if f.is_file())
    if value['aggregate_private_bytes'] > MAX_AGGREGATE_BYTES:
        raise RuntimeError('Unchanged aggregate private bound exceeded')
    return value


def launch_argv(tag):
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    args = parent.launch_argv(tag)
    old = str(BASE / 'launch464.sh')
    if args.count(old) != 1:
        raise RuntimeError('One original464 launch entry required')
    args = [str(P / 'launch465.sh') if value == old else value for value in args]
    args.insert(args.index('/bin/bash'), 'SPECSLO_V465_ROOT_FREEZE_SHA=' + EXPECTED_FREEZE)
    return args


actual_audit = parent.actual_audit


def execution_namespace():
    ns = parent.execution_namespace()
    ns.update(prepared=prepared, storage=storage, launch_argv=launch_argv, actual_audit=actual_audit)
    ns['execute_locked'] = parent.replace_constants(ns['execute_locked'], (
        ('ROOT.V464.current-card-credit-cost-full-own-AR-supervisor.v1',
         'ROOT.V465.current-component-credit-cost-full-own-AR-supervisor.v1'),
    ), ns)
    ns['SCOPE'].update(current_internal_component_binding=True)
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
