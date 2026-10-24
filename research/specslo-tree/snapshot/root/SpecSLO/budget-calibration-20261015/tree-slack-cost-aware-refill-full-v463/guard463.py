"""Fresh complete qualification with unchanged21600s and private limits."""

from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import re
import tarfile

from check463 import P, B, BASE, PARENT_FREEZE, PARENT_ARCHIVE, ready, sha
import guard462 as parent
from projection425 import project_function

DONOR, ACTION, LOCK = parent.DONOR, parent.ACTION, parent.LOCK
MAX_SECONDS = 21600
MAX_TAG_BYTES = 256 * 1024**2
MAX_AGGREGATE_BYTES = 3584 * 1024**2
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None


def prepared():
    parent.EXPECTED_FREEZE = PARENT_FREEZE
    parent.EXPECTED_ARCHIVE = PARENT_ARCHIVE
    _, files, previous = parent.prepared()
    own = ready(EXPECTED_FREEZE)
    extra = dict(own['files'])
    extra[str(P / 'FREEZE463.json')] = EXPECTED_FREEZE
    if sha(P / 'SOURCE463.tar.gz') != EXPECTED_ARCHIVE:
        raise RuntimeError('ROOT463 exact current archive required')
    with tarfile.open(P / 'SOURCE463.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(extra) or {m.name for m in members} != {p.lstrip('/') for p in extra}:
            raise RuntimeError('ROOT463 exact archive inventory required')
        for member in members:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != extra['/' + member.name]:
                raise RuntimeError('ROOT463 exact original archive bytes required')
    for path, wanted in extra.items():
        if path in files and files[path] != wanted:
            raise RuntimeError('ROOT463 identical inherited source required')
        files[path] = wanted
    return own, files, dict(parent462=previous, own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, NN_permission_inherited=False,
        host_admission_change_only=True, global_active_limit=64)


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += parent.parent.parent.base.donor_guard().bytes_under(P)
    if value['aggregate_private_bytes'] > MAX_AGGREGATE_BYTES:
        raise RuntimeError('ROOT463 unchanged private aggregate bound exceeded')
    return value


def launch_argv(tag):
    parent.EXPECTED_FREEZE = PARENT_FREEZE
    parent.EXPECTED_ARCHIVE = PARENT_ARCHIVE
    args = parent.launch_argv(tag)
    old = str(BASE / 'launch462.sh')
    if args.count(old) != 1:
        raise RuntimeError('ROOT463 one explicit actual full launch required')
    args = [str(P / 'launch463.sh') if a == old else a for a in args]
    args.insert(args.index('/bin/bash'), 'SPECSLO_V463_ROOT_FREEZE_SHA=' + EXPECTED_FREEZE)
    return args


def actual_audit(directory):
    result = parent.actual_audit(directory)
    completed = json.loads((Path(directory) / 'completed.json').read_text())
    records = completed.get('ROOT463_actual_credit_cost_admission')
    if not records or completed.get('ROOT463_all_original_NN_AR_clocks_and_cleanup_obligations_retained') is not True:
        raise RuntimeError('ROOT463 complete actual new admission audit required')
    result['ROOT463_actual_credit_cost_admission'] = records
    result['ROOT463_fixed_prefill_model'] = completed['ROOT463_fixed_prefill_model']
    return result


def execution_namespace():
    factory = project_function(parent.execution_namespace, (
        ('ROOT.V462.qualification-margin-adaptive-full-own-AR-supervisor.v1',
         'ROOT.V463.credit-cost-admission-full-own-AR-supervisor.v1'),
        ('ROOT462 supervisor received', 'ROOT463 supervisor received'),
    ), dict(P=P, prepared=prepared, storage=storage, launch_argv=launch_argv, actual_audit=actual_audit))
    ns = factory()
    ns['SCOPE'].update(credit_cost_refill_policy463=True, maximum_new_prefill_cohort=32,
        global_active_limit=64, numerical_parent460_unchanged=True,
        original_all_NN_and_own_AR_checks_required=True)
    if ns['execute_locked'].__globals__ is not ns:
        raise RuntimeError('ROOT463 actual owned namespace required')
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
            not re.fullmatch('[a-f0-9]{64}', h) for h in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)):
        raise RuntimeError('ROOT463 safe tag and explicit source hashes required')
    prepared()
    ns = execution_namespace()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ns['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
