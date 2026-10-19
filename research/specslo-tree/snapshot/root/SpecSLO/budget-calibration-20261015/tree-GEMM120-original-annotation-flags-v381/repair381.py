"""Retain original function annotation compilation when rebuilding the MC2 control."""
import ast
import copy
import inspect
import __future__


def install():
    import control380
    tree=ast.parse(inspect.getsource(control380.project_mc2))
    before=ast.dump(tree,include_attributes=False)
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name)and n.func.id=='compile']
    if len(calls)!=1 or calls[0].keywords:raise RuntimeError('ROOT381 one exact original three-argument compile')
    call=calls[0]
    call.keywords=ast.parse("compile(None,None,None,flags=original.__code__.co_flags & _ROOT381_ANNOTATIONS,dont_inherit=True)",mode='eval').body.keywords
    projected=copy.deepcopy(tree)
    call.keywords=[]
    if ast.dump(tree,include_attributes=False)!=before:raise RuntimeError('ROOT381 complete compiler-keyword AST inverse')
    namespace=dict(vars(control380),_ROOT381_ANNOTATIONS=__future__.annotations.compiler_flag,__file__=__file__)
    exec(compile(ast.fix_missing_locations(projected),__file__+'::preserve-original-annotation-flags','exec',dont_inherit=True),namespace)
    control380.project_mc2=namespace['project_mc2']


def projected_probe():
    import probe380
    source=inspect.getsource(probe380.run)
    before="""            if target:
                modes[-1]['all72_prefixes_protected']=intact
                lane('retain-each-control',lambda:save(Path(output)/f'rank{engine.rank}-{mode}-control380.json',modes[-1]))
"""
    after="""            if target:
                modes[-1]['all72_prefixes_protected']=intact
            lane('retain-each-control',lambda:save(Path(output)/f'rank{engine.rank}-{mode}-control380.json',modes[-1])if target else None)
"""
    if source.count(before)!=1:raise RuntimeError('ROOT381 one target-only retention stage')
    projected=source.replace(before,after)
    if projected.replace(after,before)!=source:raise RuntimeError('ROOT381 complete retention-stage source inverse')
    namespace=dict(vars(probe380))
    exec(compile(projected,__file__+'::all-four-roles-enter-same-retention-stage','exec',dont_inherit=True),namespace)
    return namespace['run']


def run(engine,native,stage,output,fixture):
    stage('ROOT381/preserve-original-compiler-annotation-mode',lambda:install())
    return projected_probe()(engine,native,stage,output,fixture)
