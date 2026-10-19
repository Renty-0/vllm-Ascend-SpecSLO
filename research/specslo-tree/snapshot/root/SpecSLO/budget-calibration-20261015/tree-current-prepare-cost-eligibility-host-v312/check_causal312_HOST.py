"""Run the unchanged causal request/phase-change test with the new allocator."""
import ast
import copy
import json
import sys
import types
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B / 'tree-capacity-request-risk-span-full-v299'),
               str(B / 'tree-spare-capacity-budget-host-v290')]
import check299


def main():
    path = B / 'tree-capacity-request-risk-span-full-v299/check_risk_allocation299_HOST.py'
    original = ast.parse(path.read_text())
    function = copy.deepcopy(next(node for node in original.body
                                  if isinstance(node, ast.FunctionDef) and node.name == 'main'))
    imports = [node for node in ast.walk(function)
               if isinstance(node, ast.ImportFrom) and node.module == 'budget299']
    outputs = [node for node in ast.walk(function)
               if isinstance(node, ast.Constant) and node.value == 'HOST_RISK_ALLOCATION299_RESULT.json']
    assert len(imports) == len(outputs) == 1
    imports[0].module = 'budget312'
    outputs[0].value = 'HOST_CAUSAL312_RESULT.json'
    changed = copy.deepcopy(function)
    imports[0].module = 'budget299'
    outputs[0].value = 'HOST_RISK_ALLOCATION299_RESULT.json'
    expected = next(node for node in original.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
    assert ast.dump(function) == ast.dump(expected)
    namespace = dict(globals())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[])),
                 str(path) + '::same-test-cost312-only', 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
