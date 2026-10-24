"""Change only the two cohort-ceiling bindings in the complete frozen LIVE."""
import ast
import copy
from admission466 import MAXIMUM_NEW
import generation463 as original


def project(source,maximum_new):
    if type(maximum_new) is not int or maximum_new not in MAXIMUM_NEW:
        raise ValueError('Predeclared cohort arm required')
    tree=ast.parse(source);before=ast.dump(tree,include_attributes=False)
    function=next(node for node in ast.walk(tree)if isinstance(node,ast.FunctionDef)
                  and node.name=='admit_available')
    original_body=copy.deepcopy(function.body)
    imports=[node for node in ast.walk(function)if isinstance(node,ast.ImportFrom)
             and node.module=='admission463']
    if len(imports)!=1:raise RuntimeError('One actual credit-cost selector import required')
    imports[0].module='admission466'
    ceilings=[node for node in ast.walk(function)if isinstance(node,ast.Assign)
              and len(node.targets)==1 and ast.unparse(node.targets[0])=='vacant_capacity']
    if (len(ceilings)!=1 or not isinstance(ceilings[0].value,ast.Call)
            or ast.unparse(ceilings[0].value.func)!='min' or len(ceilings[0].value.args)!=2
            or ast.unparse(ceilings[0].value.args[1])!='32'):
        raise RuntimeError('Exact original minimum(vacancy,32) ceiling required')
    ceilings[0].value.args[1]=ast.Constant(maximum_new)
    calls=[node for node in ast.walk(function)if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name)and node.func.id=='_choose463']
    if len(calls)!=1:raise RuntimeError('One original policy invocation required')
    calls[0].keywords.append(ast.keyword(arg='maximum_new',value=ast.Constant(maximum_new)))
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    function.body=original_body
    if ast.dump(tree,include_attributes=False)!=before:
        raise RuntimeError('Complete LIVE source inverse required')
    return result


def patch(source):
    return project(original.patch(source),48)
