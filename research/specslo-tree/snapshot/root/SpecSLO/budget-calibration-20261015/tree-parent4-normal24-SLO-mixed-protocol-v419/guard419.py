"""Finite N24/E4 component supervision; never grants online qualification."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(B / 'tree-parent4-current-M120-import-repair-v395'))
import guard395 as parent

PARENT_FREEZE = 'd3ccc5f6547ec6844df3308132a5f759beb23c08bec93a4befe6d585d7716516'
PARENT_ARCHIVE = 'd7c3f53b7564c308d5957f20aade43fa6e863b65bcbeb476a82bc8fe1ce5828a'
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None
require = parent.require


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, sources, previous = parent.prepared()
    own, extra = parent.source_owner.checked_archive(HERE, 419, EXPECTED_FREEZE, EXPECTED_ARCHIVE)
    require(own['parent395_freeze_sha256'] == PARENT_FREEZE
            and own['parent395_archive_sha256'] == PARENT_ARCHIVE, 'exact frozen395 startup parent')
    require(not set(sources) & set(extra), 'disjoint419 numerical scope source')
    sources.update(extra)
    return own, sources, dict(parent=previous, own_freeze_sha256=EXPECTED_FREEZE,
                             own_archive_sha256=EXPECTED_ARCHIVE, source_members=len(sources),
                             device_qualification_inherited=False, finite_N=24, finite_E=4,
                             finite_R=list(range(25)))


def launch_argv(tag):
    args = parent.launch_argv(tag)
    old = str(B / 'tree-parent4-current-M120-import-repair-v395/startup395.sh')
    require(args.count(old) == 1, 'one exact395 startup')
    return [str(HERE / 'startup419.sh') if value == old else value for value in args]


def storage(tag=None):
    value = parent.storage(tag)
    value['aggregate_private_bytes'] += parent.parent.parent.bytes_under(HERE)
    require(value['aggregate_private_bytes'] <= parent.original_owner.MAX_AGGREGATE_BYTES,
            'unchanged finite component storage limit')
    return value


def actual_audit(directory):
    sys.path.insert(0, str(HERE))
    from audit419 import audit
    result = audit(directory)
    done = json.loads((Path(directory) / 'completed.json').read_text())
    require(done['audit'] == result, 'actual419 worker and independent HOST audit agree')
    require(done['replies'] == [['calibrated', i, dict(bounded199=True, production_permission=False)]
                               for i in range(4)], 'all four real workers completed')
    require(json.loads((Path(directory) / 'worker-exit.json').read_text())
            == dict(closed=True, all_exited=True), 'owned worker exit complete')
    return result, 'finite_N24_E4_R0to24_current_M120_no_production', dict(
        hidden_KV_recomputed_by_HOST=False, Raw=None, Goodput=None)


def execution_namespace():
    namespace = parent.execution_namespace()
    namespace.update(prepared=prepared, launch_argv=launch_argv, storage=storage,
                     actual_audit=actual_audit, G=HERE)
    namespace['SCOPE'] = dict(namespace['SCOPE'], finite_normalN419=24, finite_E419=4,
                            finite_R419=list(range(25)), new_canonical_Fast_source419=True,
                            fresh_original173_still_required=True,
                            production_global64_requirement_retained=True,
                            full463=False, Raw=None, Goodput=None)
    require(namespace['execute_locked'].__globals__ is namespace, 'actual419 owned callbacks')
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
                    for value in [EXPECTED_FREEZE, EXPECTED_ARCHIVE]), 'safe419 tag and explicit pins')
    prepared()
    namespace = execution_namespace()
    with parent.original_owner.LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
