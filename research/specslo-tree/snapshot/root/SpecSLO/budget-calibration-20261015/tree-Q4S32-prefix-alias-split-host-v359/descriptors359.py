"""Privately widen only the original validated split's writer vector to128.

The complete original anti-alias contract and twelve-span limit are retained.
This HOST helper is not installed in any serving or active experiment process.
"""
import ast
import copy
import inspect

from descriptors304 import CheckedRows
import descriptors304 as original


def prepared():
    source = inspect.getsource(original.separate)
    tree = ast.parse(source)
    before = copy.deepcopy(tree)
    constants = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
                 and type(node.value) is int and node.value == 120]
    if len(constants) != 2:
        raise RuntimeError('exactly two original120 writer-vector dimensions required')
    for node in constants:
        node.value = 128
    projected = copy.deepcopy(tree)
    for node in constants:
        node.value = 120
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('complete original split source inverse required')
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(projected), __file__ + '::original-split-with128-writers', 'exec'), namespace)
    result = namespace['separate']
    if inspect.signature(result) != inspect.signature(original.separate):
        raise RuntimeError('original descriptor API signature required')
    return result


separate = prepared()
