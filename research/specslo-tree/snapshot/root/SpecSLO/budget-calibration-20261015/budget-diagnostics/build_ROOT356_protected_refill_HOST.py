"""Copy only additive355 delegates; preserve all64-digit numerical-source pins."""
from pathlib import Path
import hashlib
import json
import re


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
SOURCE = BASE / 'tree-admission-closed-control-full-v355'
DEST = BASE / 'tree-priority-protected-refill-full-v356'


def translate(text):
    original_pins = re.findall(r'\b[0-9a-f]{64}\b', text)
    changes = [(str(SOURCE), str(DEST)), (SOURCE.name, DEST.name),
               ('SPECSLO_V355_ROOT_FREEZE_SHA', 'SPECSLO_V356_ROOT_FREEZE_SHA'),
               ('FREEZE355.json', 'FREEZE356.json'), ('SOURCE355.tar.gz', 'SOURCE356.tar.gz'),
               ('ROOT.V355.', 'ROOT.V356.'), ('ROOT355', 'ROOT356')]
    changes.extend((name + '355', name + '356') for name in (
        'runtime', 'check', 'guard', 'generation', 'trace', 'extension', 'audit', 'startup', 'launch'))
    result = text
    for before, after in changes:
        result = result.replace(before, after)
    assert re.findall(r'\b[0-9a-f]{64}\b', result) == original_pins
    return result


def main():
    assert hashlib.sha256((SOURCE / 'FREEZE355.json').read_bytes()).hexdigest() == \
        '1f6bd48a2a1972e5f4e1efbba692302ea7dc2c43ce6a94f9aaec2b9d3628149a'
    DEST.mkdir(exist_ok=False)
    names = ('check355.py', 'runtime355.py', 'generation355.py', 'trace355.py',
             'extension355.py', 'guard355.py', 'launch355.sh', 'startup355.sh')
    for name in names:
        original = (SOURCE / name).read_text()
        result = translate(original)
        if name == 'check355.py':
            assert result.count("value['admission_mode'] == 'closed'") == 1
            result = result.replace("value['admission_mode'] == 'closed'",
                                    "value['admission_mode'] == 'protect_tight_normal'")
            result = result.replace('predeclared closed control;', 'predeclared protected refill;')
        elif name == 'generation355.py':
            assert result.count("MODE = 'closed'") == 1
            result = result.replace("MODE = 'closed'", "MODE = 'protect_tight_normal'")
            result = result.replace('empty_active_only=True', 'empty_active_only=False')
            result = result.replace('Closed-batch control', 'Priority-protected refill')
        elif name == 'runtime355.py':
            anchor = "report['ROOT356_admission_mode']='closed'"
            assert result.count(anchor) == 1
            result = result.replace(anchor, "report['ROOT356_admission_mode']='protect_tight_normal'")
        elif name == 'extension355.py':
            assert result.count("ROOT356_admission_mode='closed'") == 1
            result = result.replace("ROOT356_admission_mode='closed'",
                                    "ROOT356_admission_mode='protect_tight_normal'")
        elif name == 'guard355.py':
            anchor = "actual_live_refill=False, admission_mode='closed'"
            assert result.count(anchor) == 1
            result = result.replace(anchor, "actual_live_refill=True, admission_mode='protect_tight_normal'")
            result = result.replace('ROOT.V356.closed-control-full-own-AR-supervisor.v1',
                                    'ROOT.V356.protected-refill-full-own-AR-supervisor.v1')
        target = DEST / name.replace('355', '356')
        target.write_text(result)
    result = dict(status='HOST_additive_protected_refill_sources_prepared',
                  mode='protect_tight_normal', template355_source_bytes_retained=True,
                  all_original64hex_source_pins_preserved=True,
                  original348_model_and_actual_budget_code_unchanged=True,
                  fresh_all_original_numerical_and_ownAR_gates_required=True,
                  NPU_used=False, frozen=False, Goodput=None)
    (DEST / 'HOST_BUILD356_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
