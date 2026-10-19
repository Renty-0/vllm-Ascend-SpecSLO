"""Batch contiguous index ranges; retain all original layout checks.

HOST proposal only. The private269 factory first performs its original AST
inverse check. Two index-construction loops are then replaced privately.
"""
import ast
import builtins
import copy
import types

_LAYOUT = None


def _compile_layout(source, filename, mode):
    if not filename.endswith('layout269.py::original255-with-alternate-chain') or mode != 'exec':
        raise RuntimeError('ROOT308 exact original269 compilation required')
    tree = ast.parse(source)
    original = copy.deepcopy(tree)
    function = tree.body[0]
    owner_loop = next(node for node in function.body
                      if isinstance(node, ast.For) and ast.unparse(node.target) == '(plan, owner)')
    copy_index = next(i for i, node in enumerate(owner_loop.body)
                      if isinstance(node, ast.For) and ast.unparse(node.iter) == 'enumerate(tail)')
    identity_index = next(i for i, node in enumerate(function.body)
                          if isinstance(node, ast.For) and ast.unparse(node.iter) == 'range(padding)')
    old_copy = copy.deepcopy(owner_loop.body[copy_index])
    old_identity = copy.deepcopy(function.body[identity_index])
    expected_copy = ast.parse('''
for j, pos in enumerate(tail):
    physical_page = table[pos // BLOCK]
    if not 0 <= physical_page < 544:
        raise ValueError('branch reads only original allocated KV')
    source.append(physical_page * BLOCK + pos % BLOCK)
    dest.append(own[j // BLOCK] * BLOCK + j % BLOCK)
''').body[0]
    expected_identity = ast.parse('''
for j in range(padding):
    slot = identity[j // BLOCK] * BLOCK + j % BLOCK
    source.append(slot)
    dest.append(slot)
''').body[0]
    if (ast.dump(old_copy) != ast.dump(expected_copy)
            or ast.dump(old_identity) != ast.dump(expected_identity)):
        raise RuntimeError('ROOT308 two exact original row loops required')
    new_copy = ast.parse('''
for physical_page in {table[pos // BLOCK] for pos in tail}:
    if not 0 <= physical_page < 544:
        raise ValueError('branch reads only original allocated KV')
source.extend(table[pos // BLOCK] * BLOCK + pos % BLOCK for pos in tail)
for start in range(0, len(tail), BLOCK):
    first = own[start // BLOCK] * BLOCK
    dest.extend(range(first, first + min(BLOCK, len(tail) - start)))
''').body
    new_identity = ast.parse('''
for start in range(0, padding, BLOCK):
    first = identity[start // BLOCK] * BLOCK
    run = range(first, first + min(BLOCK, padding - start))
    source.extend(run)
    dest.extend(run)
''').body
    owner_loop.body[copy_index:copy_index + 1] = new_copy
    function.body[identity_index:identity_index + 1] = new_identity
    changed = copy.deepcopy(tree)
    owner_loop.body[copy_index:copy_index + len(new_copy)] = [old_copy]
    function.body[identity_index:identity_index + len(new_identity)] = [old_identity]
    if ast.dump(tree) != ast.dump(original):
        raise RuntimeError('ROOT308 complete AST inverse required')
    return builtins.compile(ast.fix_missing_locations(changed), __file__ + '::range-packed-layout', mode)


def prepared():
    global _LAYOUT
    if _LAYOUT is None:
        import layout269
        original = layout269.prepared
        factory = types.FunctionType(original.__code__,
                                     dict(original.__globals__, _LAYOUT=None, compile=_compile_layout),
                                     original.__name__, original.__defaults__, original.__closure__)
        _LAYOUT = factory()
    return _LAYOUT


def layout(*args, **kwargs):
    return prepared()(*args, **kwargs)
