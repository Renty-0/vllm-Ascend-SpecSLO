"""Original425 owned supervisor; additive428 plus original commit binding."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import re
import tarfile
from check433 import P, B, PARENT_FREEZE, PARENT_ARCHIVE, ready, sha
import guard430 as parent
import guard425 as base

EXPECTED_FREEZE = EXPECTED_ARCHIVE = None
require = base.require
DONOR, ACTION, LOCK = base.DONOR, base.ACTION, base.LOCK


def prepared():
    parent.EXPECTED_FREEZE = PARENT_FREEZE
    parent.EXPECTED_ARCHIVE = PARENT_ARCHIVE
    _, sources, previous = parent.prepared()
    own = ready(EXPECTED_FREEZE)
    extra = dict(own['files'])
    extra[str(P / 'FREEZE433.json')] = EXPECTED_FREEZE
    require(not set(extra).intersection(sources), 'disjoint additive433 source')
    require(sha(P / 'SOURCE433.tar.gz') == EXPECTED_ARCHIVE, 'exact own433 source archive')
    with tarfile.open(P / 'SOURCE433.tar.gz', 'r:gz') as saved:
        members = saved.getmembers()
        require(len(members) == len(extra) and {m.name for m in members} == {p.lstrip('/') for p in extra},
                'all own433 archive members')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest() == extra['/' + member.name],
                    'exact433 archived bytes')
    sources.update(extra)
    return own, sources, dict(parent430=previous, own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, original199_commit_binding_restored=True,
        device_qualification_inherited=False)


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += base.donor_guard().bytes_under(P)
    require(value['aggregate_private_bytes'] <= base.MAX_AGGREGATE_BYTES, 'unchanged aggregate private cap')
    return value


def launch_argv(tag):
    args = parent.launch_argv(tag)
    old = str(parent.P / 'launch430.sh')
    require(args.count(old) == 1, 'one actual430 shell entry')
    return [str(P / 'launch433.sh') if v == old else v for v in args]


def actual_audit(directory):
    from projection425 import project_function
    actual = project_function(base.actual_audit, (('from audit425 import audit', 'from audit433 import audit'),))
    return actual(directory)


def execution_namespace():
    from projection425 import project_function
    factory = project_function(base.execution_namespace, (
        ('ROOT.V425.best257-uniform-KV-full-own-AR-supervisor.v1', 'ROOT.V433.best257-uniform-KV-full-own-AR-supervisor.v1'),
        ('ROOT425 supervisor received', 'ROOT433 supervisor received'),
    ), dict(HERE=P, prepared=prepared, storage=storage, launch_argv=launch_argv, actual_audit=actual_audit))
    ns = factory()
    ns['SCOPE'] = dict(ns['SCOPE'], current_backend_state_field='uniform_KV_bucket425_enabled',
        original199_prestartup_commit_identity_bound=True, all428_backend_and_math_retained=True, original218_shadow_closes_after_all75=True)
    require(ns['execute_locked'].__globals__ is ns, 'actual433 owned callbacks')
    return ns


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = args.expected_freeze_sha256, args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None
            and all(re.fullmatch('[a-f0-9]{64}', v) is not None for v in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)),
            'safe433 tag and explicit pins')
    prepared()
    ns = execution_namespace()
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ns['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
