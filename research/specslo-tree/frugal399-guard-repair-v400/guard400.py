"""Launch finite399 with duplicate receipt keywords removed before execution.

No399 model run has occurred. Its frozen neural/runtime sources stay untouched;
this additive supervision layer fixes only the prelaunch receipt metadata.
"""
import argparse
import fcntl
import inspect
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
B = HERE.parent
PREVIOUS = B/'tree-M120-S32-frugal-independent-target-host-v399'
sys.path.insert(0, str(PREVIOUS))
import guard399 as parent

PARENT_FREEZE = '2b358bc162f42527eeeeeb9738fc9655a541e1a9e45353a0980530a95d0b0b37'
PARENT_ARCHIVE = '51fd4d110516e4d4516f520158a79060ad256568158d994f52ee83f47591d97d'
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None
require = parent.require


def own_archive():
    source = inspect.getsource(parent.own_archive)
    changes = (('FREEZE399.json', 'FREEZE400.json'), ('SOURCE399.tar.gz', 'SOURCE400.tar.gz'))
    result = source
    for before, after in changes:
        require(result.count(before) == 1, 'one actual399 archive filename '+before)
        result = result.replace(before, after)
    inverse = result
    for before, after in reversed(changes):
        inverse = inverse.replace(after, before)
    require(inverse == source, 'exact400 archive-reader source inverse')
    namespace = dict(vars(parent), HERE=HERE, EXPECTED_FREEZE=EXPECTED_FREEZE,
                     EXPECTED_ARCHIVE=EXPECTED_ARCHIVE)
    exec(compile(result, __file__+'::same-archive-reader-new-own-files', 'exec'), namespace)
    return namespace['own_archive']()


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, sources, previous = parent.prepared()
    own, extra = own_archive()
    require(own['parent399_freeze_sha256'] == PARENT_FREEZE
            and own['parent399_archive_sha256'] == PARENT_ARCHIVE, 'unchanged399 numerical source pins')
    require(not set(sources) & set(extra), 'disjoint400 metadata-only sources')
    sources.update(extra)
    return own, sources, dict(parent=previous, own_freeze_sha256=EXPECTED_FREEZE,
                             own_archive_sha256=EXPECTED_ARCHIVE,
                             source_members=len(sources), numerical_bodies_changed=False,
                             model_qualification_inherited=False)


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += sum(path.stat().st_size for path in HERE.rglob('*') if path.is_file())
    require(value['aggregate_private_bytes'] <= 3584*1024**2, 'same bounded400 private storage')
    return value


def execution_namespace():
    namespace = parent.execution_namespace()
    scope = dict(namespace['SCOPE'])
    require(scope.pop('action') == parent.ACTION399
            and scope.pop('numeric_source_obligations_only') is True,
            'remove only two duplicate explicit receipt keywords')
    scope['supervision_metadata_repair400'] = True
    namespace.update(prepared=prepared, storage=storage, G=HERE, SCOPE=scope)
    require(namespace['execute_locked'].__globals__ is namespace, 'same actual399 supervision body')
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
            and all(re.fullmatch('[0-9a-f]{64}', value) for value in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)),
            'safe400 tag and explicit source pins')
    prepared()
    namespace = execution_namespace()
    with (B/'.ROOT_NPU_EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
