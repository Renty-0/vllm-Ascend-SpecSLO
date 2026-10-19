"""Prepare additive launch/guardian sources without modifying frozen348 bytes."""
from pathlib import Path
import ast
import hashlib
import json


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
PARENT = BASE / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348'
DEST = BASE / 'tree-admission-closed-control-full-v355'
PARENT_SHA = '09119c918ad6d28d137a731cf041e69f7a0135dc302169455fa944df6caab5c8'


def changed(source, changes):
    result = source
    for before, after, count in changes:
        assert result.count(before) == count, (before, result.count(before))
        result = result.replace(before, after)
    inverse = result
    for before, after, count in reversed(changes):
        assert inverse.count(after) == count, (after, inverse.count(after))
        inverse = inverse.replace(after, before)
    assert inverse == source
    return result


def main():
    assert hashlib.sha256((PARENT / 'FREEZE348.json').read_bytes()).hexdigest() == PARENT_SHA
    guard = changed((PARENT / 'guard348.py').read_text(), (
        ('from check348 import', 'from check355 import', 1),
        ('FREEZE348.json', 'FREEZE355.json', 1),
        ('SOURCE348.tar.gz', 'SOURCE355.tar.gz', 1),
        ('launch348.sh', 'launch355.sh', 1),
        ('from audit348 import audit', 'from audit355 import audit', 1),
        ("'SPECSLO_V348_ROOT_FREEZE_SHA='+EXPECTED_FREEZE",
         "'SPECSLO_V355_ROOT_FREEZE_SHA='+EXPECTED_FREEZE, 'SPECSLO_V348_ROOT_FREEZE_SHA=" + PARENT_SHA + "'", 1),
        ('actual_live_refill=True, refill_minimum=4',
         "actual_live_refill=False, admission_mode='closed', refill_minimum=4", 1),
        ('ROOT.V348.canonical-full-own-AR-supervisor.v1', 'ROOT.V355.closed-control-full-own-AR-supervisor.v1', 1),
        ('ROOT348 supervisor received', 'ROOT355 supervisor received', 1),
    ))
    ast.parse(guard)
    (DEST / 'guard355.py').write_text(guard)
    launch = changed((PARENT / 'launch348.sh').read_text(), (
        (str(PARENT / 'startup348.sh'), str(DEST / 'startup355.sh'), 1),
    ))
    anchor = '  SPECSLO_V348_ROOT_FREEZE_SHA="${SPECSLO_V348_ROOT_FREEZE_SHA:?explicit ROOT348 freeze required}" \\\n'
    assert launch.count(anchor) == 1
    launch = launch.replace(anchor, anchor +
        '  SPECSLO_V355_ROOT_FREEZE_SHA="${SPECSLO_V355_ROOT_FREEZE_SHA:?explicit ROOT355 freeze required}" \\\n')
    (DEST / 'launch355.sh').write_text(launch)
    startup = changed((PARENT / 'startup348.sh').read_text(), (
        (str(PARENT / 'check348.py'), str(DEST / 'check355.py'), 1),
        (str(PARENT / 'runtime348.py'), str(DEST / 'runtime355.py'), 1),
    ))
    anchor = 'export PYTHONPATH='
    assert startup.count(anchor) == 1
    startup = startup.replace(anchor, anchor + str(DEST) + ':' +
        str(BASE / 'tree-priority-protected-live-refill-host-v354') + ':')
    (DEST / 'startup355.sh').write_text(startup)
    old_pin = 'SPECSLO_V218_ROOT_FREEZE_SHA=fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'
    assert launch.count(old_pin) == 1
    result = dict(status='HOST_additive_launch_and_guard_sources_prepared',
                  original348_freeze_sha256=PARENT_SHA, original218_launch_pin_preserved=True,
                  launch_shell_environment_preserved_except_own_source_pin_and_lookup_paths=True,
                  old_model_and_kernel_configuration_unchanged=True,
                  original_full_NN_and_owned_cleanup_gates_retained=True,
                  mode='closed', NPU_used=False, frozen=False, Goodput=None)
    (DEST / 'HOST_BUILD355_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
