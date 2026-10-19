"""Real165 protocol and all visible KV paths, plus span/alias inverse checks."""
import ast
import copy
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B / name) for name in (
    'tree-uniform-q5-M160-S32-C128-host-v326',
    'tree-single-validated-prefix-descriptors-host-v304')]
import check_base326_HOST as original


def verify_descriptors(engine):
    state = engine._branch_tail_state
    slots = state.static_current['physical'][2].slot_mapping.tolist()
    assert len(slots) == 160
    if not state.static_current['copy_enabled']:
        assert state.split326_last_HOST is None
        return
    contract = state.copy326_contract
    expected = sorted((s, d) for s, d in zip(contract.source, contract.destination, strict=True) if s != d)
    prefix = [(s + j, d + j) for s, d, count in state.prefix326_spans.tolist() for j in range(count)]
    aliases = [(slots[row], dest) for row, dest in enumerate(state.alias326_destination.tolist()) if dest >= 0]
    assert sorted(prefix + aliases) == expected
    assert len({d for _, d in prefix + aliases}) == len(expected)
    assert state.split326_last_HOST['query_slots'] == slots
    assert len(prefix) == state.split326_last_HOST['prefix_copies']
    assert len(aliases) == state.split326_last_HOST['current_aliases']
    from descriptors326 import separate
    for invalid in (slots[:-1], [-1] + slots[1:]):
        try:
            separate(contract, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid165 physical writer list accepted')


def main():
    source = Path(original.__file__)
    tree = ast.parse(source.read_text())
    function = copy.deepcopy(next(node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name == 'main'))
    setup = [node for node in ast.walk(function)
             if isinstance(node, ast.ImportFrom) and node.module == 'fixture326_HOST']
    assert len(setup) == 1
    setup[0].module = 'fixture326_HOST'
    appends = [node for node in ast.walk(function)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and ast.unparse(node.value.func) == 'results.append']
    assert len(appends) == 1
    target = appends[0]

    def insert(nodes):
        for node in ast.walk(nodes):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list) and any(item is target for item in value):
                    index = next(i for i, item in enumerate(value) if item is target)
                    value[index:index] = ast.parse('verify_descriptors(e)').body
                    return
        raise AssertionError('Original post-ancestry result boundary absent')

    insert(function)
    outputs = [node for node in ast.walk(function)
               if isinstance(node, ast.Constant) and node.value == 'HOST326_ANCESTRY_RESULT.json']
    assert len(outputs) == 1
    outputs[0].value = 'HOST326_ANCESTRY_AND_SPLIT_RESULT.json'
    namespace = dict(vars(original), P=P, verify_descriptors=verify_descriptors)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source) + '::same-ancestry-test-new326-prepare', 'exec'), namespace)
    namespace['main']()
    result = json.loads((P / 'HOST326_ANCESTRY_AND_SPLIT_RESULT.json').read_text())
    assert result['cases'] == 1020 and result['NPU_used'] is False
    assert 'torch_npu' not in sys.modules


if __name__ == '__main__':
    main()
