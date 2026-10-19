"""Source-only lifecycle/env identity and bounded guard checks; no model use."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys


P = Path(__file__).resolve().parent
OLD = P.parent / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'


def main():
    paths = [P / 'HOST361_RUNTIME_RESULT.json', P / 'HOST361_GUARD_SOURCE_RESULT.json']
    assert not any(path.exists() for path in paths) and 'torch' not in sys.modules
    import runtime361
    actual = runtime361.worker_source()
    old = runtime361.original.worker_source()
    restored = actual.replace('from check361 import ready', 'from check344 import ready').replace(
        'from probe361 import run', 'from probe344 import run')
    assert restored == old and actual.count('from check361 import ready') == 2
    assert actual.count('from probe361 import run') == 1 and actual.count('numeric_cases=594') == 2
    compile(actual, 'actual_original344_worker_with361_imports', 'exec')
    old_nodes = ast.parse(old)
    new_nodes = ast.parse(actual)
    for name in ('worker', 'main'):
        a = next(node for node in ast.walk(old_nodes) if isinstance(node, ast.FunctionDef) and node.name == name)
        b = next(node for node in ast.walk(new_nodes) if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(a.args) == ast.dump(b.args)
    parent_startup = (OLD / 'startup344.sh').read_text()
    startup = (P / 'startup361.sh').read_text()
    lines = lambda source: [line for line in source.splitlines() if 'VLLM_ASCEND_' in line]
    assert lines(startup) == lines(parent_startup)
    assert 'PYTHONPATH=' + str(P) + ':' in startup
    assert 'map-current.py 2,5,6,7' in startup
    assert str(P / 'runtime361.py') in startup and str(P / 'check361.py') in startup
    assert sorted(re.findall(r'\b[0-9a-f]{64}\b', actual)) == sorted(re.findall(r'\b[0-9a-f]{64}\b', old))
    runtime_result = dict(status='PASS_HOST_original344_full_worker_and_ENV_source_identity',
        entire_worker_inverse=True, original_signatures_and_594_cases_preserved=True,
        all_VLLM_ENV_lines_identical=True, actual_fixture_directory=str(runtime361.original.original.P),
        new_numeric_result_inherited=False, NPU_used=False, Goodput=None)
    guard = (P / 'guard361.py').read_text()
    tree = ast.parse(guard)
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in ('MAX_SECONDS',)}
    assert values['MAX_SECONDS'] == 3600
    assert 'MAX_TAG_BYTES = 192*1024**2' in guard
    assert 'MAX_AGGREGATE_BYTES = 3584*1024**2' in guard
    assert "PARENT = B/'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'" in guard
    assert 'import guard344 as parent' in guard and 'from audit361 import audit' in guard
    assert 'SPECSLO_V361_ROOT_FREEZE_SHA' in guard and 'SPECSLO_V344_ROOT_FREEZE_SHA' in guard
    assert 'flock' in guard and 'execute_locked' in guard
    assert sorted(re.findall(r'\b[0-9a-f]{64}\b', guard)) == sorted(re.findall(
        r'\b[0-9a-f]{64}\b', (OLD / 'guard344.py').read_text()))
    compile(guard, str(P / 'guard361.py'), 'exec')
    guard_result = dict(status='PASS_HOST_guard361_source_limits_parent_pins_and_owned_lifecycle',
        seconds=3600, tag_private_limit_MiB=192, aggregate_private_limit_MiB=3584,
        original218_literal_pin_identical=True, original344_parent_verified=True,
        original_owned_guard_factory_and_cleanup_unchanged=True,
        actual_prepared_execution_pending_freeze=True, NPU_used=False, Goodput=None)
    for path, result in zip(paths, (runtime_result, guard_result), strict=True):
        with path.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(dict(runtime=runtime_result['status'], guard=guard_result['status'], NPU_used=False)))


if __name__ == '__main__':
    main()
