"""Original Native allocator/forward CPU fixture, specialized Runner ABI only."""
import ast,copy,importlib.util,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
SOURCE=B/'tree-static-q5-M120-full463-own-AR-v256/check_full256_HOST.py'

def build():
    spec=importlib.util.spec_from_file_location('_ROOT326_actual_CPU_fixture',SOURCE)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    function=copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
    cut=next(i for i,n in enumerate(function.body)if isinstance(n,ast.ImportFrom)and n.module in('static_state255','static_state256'))
    function.name='_ROOT326_fixture';function.body=function.body[:cut]
    runner=next(n for n in function.body if isinstance(n,ast.ClassDef)and n.name=='Runner')
    for n in ast.walk(runner):
        if isinstance(n,ast.Constant)and type(n.value)is int and n.value in(24,121):n.value={24:32,121:161}[n.value]
    function.body+=ast.parse('from static_base326 import install\ninstall(engine,native)').body
    function.body.append(ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])))
    ns=dict(vars(fixture));exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(SOURCE)+'::CPU-original-prefix-specialized-physical165','exec'),ns)
    env=ns[function.name]();engine=env['engine']
    from layout326 import layout
    state=engine._branch_tail_state;outer=state.prepare;cells=dict(zip(outer.__code__.co_freevars,outer.__closure__))
    prior=cells['prepare'].cell_contents;fn=prior.__func__
    private=types.FunctionType(fn.__code__,dict(fn.__globals__,layout=layout),fn.__name__,fn.__defaults__,fn.__closure__)
    def cell(value):return(lambda:value).__closure__[0]
    closure=tuple(cell(types.MethodType(private,state))if name=='prepare'else cells[name]for name in outer.__code__.co_freevars)
    state.prepare=types.FunctionType(outer.__code__,outer.__globals__,outer.__name__,outer.__defaults__,closure)
    return env
