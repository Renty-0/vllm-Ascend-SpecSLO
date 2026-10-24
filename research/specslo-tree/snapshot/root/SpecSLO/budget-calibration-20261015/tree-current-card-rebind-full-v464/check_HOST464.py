"""Executable ownership and complete inverse controls for the device rebind."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys

P = Path(__file__).resolve().parent
B = P.parent
OLD = B / 'tree-slack-cost-aware-refill-full-v463'
sys.path.insert(0, str(P))
import guard464
import guard463
import runtime463


def inverse(path, original, changes):
    text = path.read_text()
    for old, new in reversed(changes):
        assert text.count(new) == 1, (path, new)
        text = text.replace(new, old)
    assert text == original.read_text(), path


def main():
    inverse(P / 'map464.py', B / 'serial-v1/map-current.py', [
        ('enumerate([2, 5, 6, 7])', 'enumerate([0, 1, 2, 7])'),
    ])
    inverse(P / 'environment464.sh', B / 'serial-v1/runtime.sh', [
        ('${SPECSLO_VISIBLE_DEVICES:-2,5,6,7}', '${SPECSLO_VISIBLE_DEVICES:-0,1,2,7}'),
        (str(B / 'serial-v1/map-current.py'), str(P / 'map464.py')),
    ])
    old = '  /bin/bash ' + str(B / 'serial-v1/runtime.sh') + ' \\\n'
    new = '  SPECSLO_V464_ROOT_FREEZE_SHA="${SPECSLO_V464_ROOT_FREEZE_SHA:?explicit ROOT464 freeze required}" \\\n  /bin/bash ' + str(P / 'environment464.sh') + ' \\\n'
    inverse(P / 'launch464.sh', OLD / 'launch463.sh', [
        (old, new), (str(OLD / 'startup463.sh'), str(P / 'startup464.sh')),
    ])
    inverse(P / 'startup464.sh', OLD / 'startup463.sh', [
        ('"$python" -B "' + str(OLD / 'check463.py') + '"',
         '"$python" -B "' + str(P / 'check464.py') + '"'),
        ('export PYTHONPATH=', 'export PYTHONPATH=' + str(P) + ':'),
        (str(B / 'serial-v1/map-current.py') + ' 2,5,6,7', str(P / 'map464.py') + ' 0,1,2,7'),
    ])
    previous, current = guard463.execution_namespace(), guard464.execution_namespace()
    a, z = previous['execute_locked'].__code__, current['execute_locked'].__code__
    assert a.co_code == z.co_code
    assert a.co_names == z.co_names and a.co_varnames == z.co_varnames
    assert len(a.co_consts) == len(z.co_consts)
    deltas = [(x, y) for x, y in zip(a.co_consts, z.co_consts) if x != y]
    assert deltas == [
        ('ROOT.V463.credit-cost-admission-full-own-AR-supervisor.v1',
         'ROOT.V464.current-card-credit-cost-full-own-AR-supervisor.v1'),
        ((2, 5, 6, 7), (0, 1, 2, 7)),
    ], deltas
    for key in ('MAX_SECONDS', 'MAX_TAG_BYTES', 'MAX_AGGREGATE_BYTES'):
        assert current[key] == previous[key]
    assert current['MAX_SECONDS'] == 21600
    assert current['SCOPE']['global_active_limit'] == 64
    assert all(current['SCOPE'][key] == value for key, value in previous['SCOPE'].items())
    assert current['actual_audit'] is previous['actual_audit']
    old_helpers, helpers = previous['original_helpers'](), current['original_helpers']()
    assert helpers['verify'].__code__.co_code == old_helpers['verify'].__code__.co_code
    assert [(a, z) for a, z in zip(old_helpers['verify'].__code__.co_consts,
        helpers['verify'].__code__.co_consts) if a != z] == [
            ('serial-v1/runtime.sh', 'tree-current-card-rebind-full-v464/environment464.sh')]
    assert helpers['owned_cleanup'].__code__ == old_helpers['owned_cleanup'].__code__
    assert helpers['helper']().PHYSICAL == (0, 1, 2, 7)
    assert old_helpers['helper']().PHYSICAL == (2, 5, 6, 7)
    negative_controls = 0
    for changes in [(('absent-control', 0),), (((2, 5, 6, 7), (0, 1, 2, 7)), ((2, 5, 6, 7), (0, 1, 2, 7)))]:
        try:
            guard464.replace_constants(previous['execute_locked'], changes)
        except RuntimeError:
            negative_controls += 1
        else:
            raise AssertionError('Missing or repeated replacement was accepted')
    for path in P.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    for path in P.glob('*.py'):
        ast.parse(path.read_bytes(), filename=str(path))
    value = dict(status='PASS_HOST464_ONLY_DEVICE_CONTROL_REBIND',
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir()
                 if p.is_file() and p.suffix in ('.py', '.sh', '.json') and not p.name.startswith(('HOST', 'FREEZE', 'SOURCE'))},
        original463_runtime_source_SHA256=hashlib.sha256(runtime463.source().encode()).hexdigest(),
        supervisor_executable_bytecode_identical=True, original463_actual_audit_same_object=True,
        original_ownership_cleanup_bytecode_identical=True, all_original_scope_checks_retained=True,
        complete_shell_inverse=True, negative_controls=negative_controls,
        model_imported=False, NPU_model_used=False, NN_permission=False)
    with (P / 'HOST464_RESULT.json').open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
