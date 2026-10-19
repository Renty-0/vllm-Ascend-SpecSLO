"""All227 logical trees; adaptive candidate versus independent FULL120 reference."""
import ast
import oracle255 as original
from source377 import function
from geometry377 import adjust


def cases():
    return [adjust(case, enabled=True) for case in original.cases()]


def candidate_function():
    anchor = '    from metadata255 import checked'
    extra = '''    if name == 'candidate':
        hits = [n for n in ast.walk(node) if isinstance(n,ast.Constant) and type(n.value)is int and n.value==120]
        if len(hits)!=3:raise RuntimeError('ROOT377 three original candidate physical dimension checks')
        for old in hits:
            old.value = None
        class PhysicalDimension(ast.NodeTransformer):
            def visit_Constant(self,n):
                if n in hits:return ast.copy_location(ast.Subscript(value=ast.Name(id='case',ctx=ast.Load()),slice=ast.Constant(value='M'),ctx=ast.Load()),n)
                return n
        node=PhysicalDimension().visit(node)
'''
    specialized = function(original.specialized, ((anchor,extra+anchor),))
    return specialized('candidate')


def candidate(*args, **kwargs):
    return candidate_function()(*args, **kwargs)


def negative(engine, case, bank, plans, rows, stage, output):
    from baseline255 import run
    state = engine._branch_tail_state
    enabled = state.adaptive377_enabled
    state.adaptive377_enabled = False
    try:
        return run(engine, adjust(case, enabled=False), bank, plans, rows, stage, output)
    finally:
        state.adaptive377_enabled = enabled


def specialized_run():
    anchor = '    from metadata255 import checked'
    extra = '''    if name == 'run':
        compare=next(n for n in ast.walk(node) if isinstance(n,ast.FunctionDef) and n.name=='compare')
        at=next(i for i,n in enumerate(compare.body) if isinstance(n,ast.Return))
        compare.body[at:at]=ast.parse("need(torch.equal(actual['hidden'],own['hidden']) and all(torch.equal(a,b)for a,b in zip(actual['KV'],own['KV'],strict=True)) and actual['graph_eager_KV_bitwise_equal'], 'ROOT377 strict independent FULL120 hidden and all128 KV bits')").body
        for n in ast.walk(node):
            if isinstance(n,ast.ImportFrom)and n.module=='baseline255':
                for alias in n.names:
                    if alias.name=='run':
                        n.module='oracle377';alias.name='negative'
'''
    projected = function(original.specialized, ((anchor,extra+anchor),))
    result = projected('run')
    result.__globals__.update(cases=cases,candidate=candidate,reference=original.reference)
    return result


def run(engine,native,stage,bank,output):
    body = specialized_run()
    body.__globals__['prepare'] = lambda case: original.prepare(case,bank['prefixes'])
    return body(engine,native,stage,bank,output=output)
