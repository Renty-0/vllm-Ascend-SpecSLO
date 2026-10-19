"""Fresh original wholeprompt/own-root/nextdraft checks for refill sizes1,3,4,8."""
import ast
import copy
from pathlib import Path

import prefill209 as original

P = Path(__file__).resolve().parent
_NAMESPACE = None


def project():
    source = Path(original.__file__).read_text()
    changes = [
        ('from subset209 import prepare,guard_rest,finish', 'from subset345 import prepare,guard_rest,finish', 1),
        ('multiowner-eager-prefill-own-current-process', 'live-refill-small-cohort-own-current-process', 1),
        ('for n in (2,32)', 'for n in (1,3,4,8)', 3),
        ('representative_sizes=[2,32]', 'representative_sizes=[1,3,4,8]', 2),
        ("len(set(lens))>1", "(n==1 or len(set(lens))>1)", 1),
        ('pn=1 if n==2 else 16', 'pn=min(16,(n+1)//2)', 1),
        ('4*2*2*21', '4*4*2*21', 1),
        ('all336 new component zero subtractions', 'all672 new component zero subtractions', 1),
        ('actual2/32 plus two full zero passes', 'actual1/3/4/8 plus two full zero passes', 1),
        ('V209.current-multiowner', 'ROOT349.current-small-refill', 2),
        ('V209.independent-multiowner', 'ROOT349.independent-small-refill', 1),
        (" protected=lane('incumbent-protected-prefix',protect)\n saved,restore=", " protected=lane('incumbent-protected-prefix',protect)\n lane('original-exact-singleton-qualification-before-tap',lambda:warm_singleton(engine,native,bank))\n saved,restore=", 1),
        ('eager_original_multiowner=True,\n   singleton201_authorization_used=False', 'eager_original_multiowner=(n>1 or engine.is_draft),\n   actual_prefill_capture_routes=list(saved.routes),\n   singleton201_authorization_used=(n==1 and not engine.is_draft)', 1),
        ("row['singleton201_authorization_used'] is False", "row['singleton201_authorization_used'] is (n==1 and rank!=0) and row['actual_prefill_capture_routes']==(['original_exact_singleton_graph']if n==1 and rank!=0 else['original_eager_model'])", 1),
        ('original_helper_eager=True,', 'original_helper_eager=False,original_singleton_graph_and_multiowner_eager=True,', 1),
        (" inventory=stage('multiowner209/current-entry-objects-before'", " warm_directory=Path(output)/'small-refill349-prewarm'\n stage('small-refill349/create-warm-output',lambda:warm_directory.mkdir(parents=True,exist_ok=False)if engine.rank==0 else None)\n prewarm=[run_case(engine,native,stage,rows,n,0,warm_directory)for n in(1,3,4,8)]\n inventory=stage('multiowner209/current-entry-objects-before'", 1),
        ('rank=engine.rank,cases=result,errors=[]', 'rank=engine.rank,cases=result,warmup_cases=prewarm,errors=[]', 1),
    ]
    projected = source
    for before, after, count in changes:
        if projected.count(before) != count:
            raise RuntimeError('exact small-refill source anchor '+before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after, count in reversed(changes):
        if inverse.count(after) != count:
            raise RuntimeError('exact small-refill inverse anchor '+after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('whole original multiowner component source inverse')
    return projected


def namespace():
    global _NAMESPACE
    if _NAMESPACE is None:
        names = {'run_case','run','audit'}
        nodes = [copy.deepcopy(n) for n in ast.parse(project()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        if len(nodes) != len(names):
            raise RuntimeError('all actual original numerical bodies required')
        from capture349 import capture_begin, warm_singleton
        value = dict(vars(original),capture_begin=capture_begin,warm_singleton=warm_singleton)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     __file__+'::original-component-with-declared-small-refills', 'exec'), value)
        _NAMESPACE = value
    return _NAMESPACE


def run(engine, *args, **kwargs):
    previous = getattr(engine, '_v209_actual_prefill', None)
    try:
        value = namespace()['run'](engine, *args, **kwargs)
    finally:
        engine._v209_actual_prefill = previous
    engine._ROOT349_actual_small_refill = value
    return value
def audit(*args, **kwargs): return namespace()['audit'](*args, **kwargs)
