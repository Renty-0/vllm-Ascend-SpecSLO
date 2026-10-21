"""All original227 node cases; require strict hidden and KV bits as well."""
import ast
import oracle255 as original
from projection409 import project_function


def specialized():
    anchor = '    from metadata255 import checked'
    extra = '''    if name == 'run':
        compare = next(n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.name == 'compare')
        at = next(i for i, n in enumerate(compare.body) if isinstance(n, ast.Return))
        compare.body[at:at] = ast.parse("need(torch.equal(actual['hidden'], own['hidden']) and all(torch.equal(a,b) for a,b in zip(actual['KV'],own['KV'],strict=True)) and actual['graph_eager_KV_bitwise_equal'], 'ROOT409 strict hidden and all128 KV bits')").body
'''
    projected = project_function(original.specialized, ((anchor, extra + anchor),))
    return projected('run')


def run(engine, native, stage, bank, output):
    function = specialized()
    function.__globals__['prepare'] = lambda case: original.prepare(case, bank['prefixes'])
    return function(engine, native, stage, bank, output=output)
