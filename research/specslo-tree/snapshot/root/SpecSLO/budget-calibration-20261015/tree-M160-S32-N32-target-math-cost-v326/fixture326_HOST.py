"""Build the original165 CPU fixture before its private layout replacement."""
import ast
import copy
from pathlib import Path


def build():
    import fixture_base326_HOST as original
    source = Path(original.__file__)
    tree = ast.parse(source.read_text())
    function = copy.deepcopy(next(node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name == 'build'))
    cut = next(i for i, node in enumerate(function.body)
               if isinstance(node, ast.ImportFrom) and node.module == 'layout326')
    function.body = function.body[:cut] + [ast.Return(value=ast.Name(id='env', ctx=ast.Load()))]
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(source) + '::original165-before-layout-hook', 'exec'), namespace)
    env = namespace['build']()
    import branch_tail326
    from static_state326 import bind_checked_prepare
    bind_checked_prepare(env['engine']._branch_tail_state, branch_tail326)
    return env
