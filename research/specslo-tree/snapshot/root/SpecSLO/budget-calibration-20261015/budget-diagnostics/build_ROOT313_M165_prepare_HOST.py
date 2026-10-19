"""Prepare M165 descriptors without changing any running or frozen source."""
import ast
import copy
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-M165-single-prepare-host-v313'


def project(source, names=None, strings=None, imports=None):
    tree = ast.parse(source.read_text())
    original = copy.deepcopy(tree)
    edits = []

    def change(node, field, value):
        old = getattr(node, field)
        if old != value:
            edits.append((node, field, old))
            setattr(node, field, value)

    def rename(value):
        for old, new in (names or {}).items():
            value = value.replace(old, new)
        return value

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            change(node, 'id', rename(node.id))
        elif isinstance(node, ast.arg):
            change(node, 'arg', rename(node.arg))
        elif isinstance(node, ast.Attribute):
            change(node, 'attr', rename(node.attr))
        elif isinstance(node, ast.alias):
            change(node, 'name', rename(node.name))
        elif isinstance(node, ast.ImportFrom):
            module = (imports or {}).get(node.module, node.module)
            if node.module == 'descriptors304' and [n.name for n in node.names] == ['separate']:
                module = 'descriptors313'
            change(node, 'module', module)
        elif isinstance(node, ast.Constant):
            if type(node.value) is int and node.value == 120:
                change(node, 'value', 165)
            elif isinstance(node.value, str):
                value = node.value
                for old, new in (strings or {}).items():
                    value = value.replace(old, new)
                change(node, 'value', value)
    encoded = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    for node, field, old in reversed(edits):
        setattr(node, field, old)
    assert ast.dump(tree) == ast.dump(original)
    return encoded, len(edits)


def main():
    P.mkdir(exist_ok=False)
    records = []
    source = B / 'tree-single-validated-prefix-prepare-host-v305/static_state305.py'
    code, count = project(source,
        names={'305': '313', 'branch_tail255': 'branch_tail285', 'static_state255': 'static_base285'},
        strings={'305': '313', '120': '165'},
        imports={'static_state255': 'static_base285', 'layout269': 'layout285',
                 'padding269': 'padding285', 'kernels288': 'write313'})
    (P / 'static_state313.py').write_text(code)
    records.append(dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        output='static_state313.py', edits=count, complete_AST_inverse=True))
    source = B / 'tree-single-validated-prefix-descriptors-host-v304/descriptors304.py'
    original = ast.parse(source.read_text())
    function = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == 'separate')
    selected = copy.deepcopy(function)
    integers = [node for node in ast.walk(selected) if isinstance(node, ast.Constant)
                and type(node.value) is int and node.value == 120]
    assert len(integers) == 2
    for node in integers:
        node.value = 165
    code = ast.unparse(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))) + '\n'
    for node in integers:
        node.value = 120
    assert ast.dump(selected) == ast.dump(function)
    (P / 'descriptors313.py').write_text(
        '"""Same full immutable304 proof; physical writer count is165."""\n'
        'from descriptors304 import CheckedRows\n' + code)
    source = B / 'tree-prefix-precopy-dual-cachewrite-host-v288/kernels288.py'
    original = ast.parse(source.read_text())
    function = copy.deepcopy(next(node for node in original.body
                                  if isinstance(node, ast.FunctionDef) and node.name == 'dual_write'))
    edits = []
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 120:
            edits.append(node)
            node.value = 165
    assert len(edits) == 4
    code = ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))) + '\n'
    for node in edits:
        node.value = 120
    expected = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == 'dual_write')
    assert ast.dump(function) == ast.dump(expected)
    (P / 'write313.py').write_text(
        '"""Same raw288 per-row device body,165 independent row programs; unqualified."""\n'
        'from kernels288 import torch,torch_npu,triton,_dual_cache_write288\n' + code)
    for path in P.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    result = dict(status='SOURCE_ONLY_M165_prepare_not_frozen_not_NPU_executed',
                  original_geometry='M120/S24/C96', proposed_geometry='M165/S33/C128/N32/F12',
                  descriptor_full1792_contract_preserved=True,
                  all64_prefix_span_device_body_unchanged=True,
                  raw_dual_write_device_body_unchanged_only_grid165=True,
                  actual_NPU_launch_or_model_math_not_qualified=True,
                  unchanged_BF16_no_quantization=True,
                  future_global_budget192_preservation_required=True,
                  inherited_model_qualification=False, source_projections=records,
                  NPU_used=False, Goodput=None, adopted=False)
    (P / 'SOURCE_ONLY313_REVIEW.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
