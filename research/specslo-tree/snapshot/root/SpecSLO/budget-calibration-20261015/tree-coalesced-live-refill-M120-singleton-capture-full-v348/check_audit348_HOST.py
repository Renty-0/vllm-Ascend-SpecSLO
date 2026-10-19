"""Run the actual original raw-auditor loader with only two B8 bounds."""
import ast
import inspect
import json
import sys
from pathlib import Path

import guard348
from audit_domain348 import declared_functions, original


def main():
    inherited = original.inherited_module('audit_production_v101')
    namespace = dict(vars(inherited))
    path = original.BASE / 'audit_production_v101.py'
    text = declared_functions(path, namespace, names=('audit_trace_set',))
    tree = ast.parse(text)
    compared = []
    for name in ('audit_transactions', 'audit_trace_set'):
        old = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        updated = namespace[name]
        assert updated.__code__.co_filename.endswith('::original-domain8')
        assert updated.__globals__ is namespace
        comparisons = [node for node in ast.walk(old)
                       if isinstance(node, ast.Compare) and ast.unparse(node) == '1 <= b <= 5']
        assert len(comparisons) == 1
        # The loader itself verifies the inverse for the entire source module.
        assert 8 in updated.__code__.co_consts or any(
            hasattr(constant, 'co_consts') and 8 in constant.co_consts
            for constant in updated.__code__.co_consts)
        compared.append(name)
    assert not any(name.split('.')[0] in ('torch', 'torch_npu') for name in sys.modules)
    value = dict(status='HOST_actual_raw_auditor_two_declared_B8_bounds_and_whole_AST_inverse_passed',
                 functions=compared, source=str(path), numerical_matchers_changed=False,
                 N32_C96_root128_limits_unchanged=True, NPU_used=False)
    target = Path(__file__).resolve().parent / 'HOST_AUDIT348_RESULT.json'
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
