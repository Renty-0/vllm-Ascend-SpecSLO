"""Exact391 component protocol, owned supervision and additive import repair."""
import argparse
import fcntl
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(B / 'tree-parent4-current-M120-target-protocol-v391'))
import guard391 as parent

PARENT_FREEZE = 'a9a4bfaa81f9bc8f95a75a533612fb1bd8b8202638a6f65c6d35b8185fe8fa98'
PARENT_ARCHIVE = 'e97edf491453ea415c815a1f7b372511c0e35184fdb32b9265a184a38f207ed8'
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None
source_owner = parent.parent.parent.parent
original_owner = source_owner.original
require = parent.require


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, sources, previous = parent.prepared()
    own, extra = source_owner.checked_archive(HERE, 395, EXPECTED_FREEZE, EXPECTED_ARCHIVE)
    require(own['parent391_freeze_sha256'] == PARENT_FREEZE
            and own['parent391_archive_sha256'] == PARENT_ARCHIVE, 'exact391 numerical sources')
    require(not set(sources) & set(extra), 'disjoint395 startup repair sources')
    sources.update(extra)
    return own, sources, dict(parent=previous, own_freeze_sha256=EXPECTED_FREEZE,
                             own_archive_sha256=EXPECTED_ARCHIVE, source_members=len(sources),
                             device_qualification_inherited=False, numerical_bodies_changed=False)


def launch_argv(tag):
    args = parent.launch_argv(tag)
    old = str(B / 'tree-parent4-current-M120-target-protocol-v391/startup391.sh')
    require(args.count(old) == 1, 'one exact391 startup')
    return [str(HERE / 'startup395.sh') if value == old else value for value in args]


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += parent.parent.bytes_under(HERE)
    require(value['aggregate_private_bytes'] <= original_owner.MAX_AGGREGATE_BYTES,
            'unchanged finite protocol storage limit')
    return value


def execution_namespace():
    namespace = parent.execution_namespace()
    namespace.update(prepared=prepared, launch_argv=launch_argv, storage=storage, G=HERE)
    namespace['SCOPE'] = dict(namespace['SCOPE'],
                            startup_dependency_repair_only=True,
                            original391_numerical_bodies_unchanged=True,
                            production_global64_requirement_retained=True,
                            full463=False, Raw=None, Goodput=None)
    require(namespace['execute_locked'].__globals__ is namespace, 'actual395 owned callbacks')
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
            and all(re.fullmatch('[0-9a-f]{64}', value)
                    for value in [EXPECTED_FREEZE, EXPECTED_ARCHIVE]), 'safe395 tag and explicit pins')
    prepared()
    namespace = execution_namespace()
    with original_owner.LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
