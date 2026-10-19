"""Dedent inspected class methods while retaining the exact guarded linear delta."""
import ast
import copy
import inspect
import textwrap


def install():
    import hybrid382
    tree=ast.parse(inspect.getsource(hybrid382.project_linear))
    before=ast.dump(tree,include_attributes=False)
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)
           and n.func.value.id=='ast'and n.func.attr=='parse']
    if len(calls)!=1 or len(calls[0].args)!=1:raise RuntimeError('ROOT384 exact one original method source parser')
    call=calls[0];old=copy.deepcopy(call.args[0])
    call.args[0]=ast.Call(func=ast.Name(id='_ROOT384_dedent',ctx=ast.Load()),args=[copy.deepcopy(old)],keywords=[])
    projected=copy.deepcopy(tree);call.args[0]=old
    if ast.dump(tree,include_attributes=False)!=before:raise RuntimeError('ROOT384 full parser delta AST inverse')
    namespace=dict(vars(hybrid382),_ROOT384_dedent=textwrap.dedent)
    exec(compile(ast.fix_missing_locations(projected),__file__+'::dedent-actual-native-method-source','exec',dont_inherit=True),namespace)
    hybrid382.project_linear=namespace['project_linear']


def run(engine,native,stage,output,fixture):
    stage('ROOT384/retain-actual-native-method-source-indentation',lambda:install())
    from probe382 import run as original
    return original(engine,native,stage,output,fixture)
