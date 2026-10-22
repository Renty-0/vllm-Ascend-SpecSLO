"""Retain fresh173 math; select the own pinned helper origin and N24 lifecycle."""
import ast
from pathlib import Path
import component199 as original


def baseline(engine,native,stage,policy,output):
    source=Path(original.__file__).read_text()
    before='from extension199 import loaded_origins as assert_origins'
    after='from extension419 import loaded_origins as assert_origins'
    if source.count(before)!=1:raise RuntimeError('ROOT419 exact baseline origin anchor')
    projected=source.replace(before,after)
    if projected.replace(after,before)!=source:raise RuntimeError('ROOT419 baseline inverse')
    node=next(n for n in ast.parse(projected).body if isinstance(n,ast.FunctionDef)and n.name=='baseline')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        __file__+'::original199-baseline-with-own-origin','exec'),namespace)
    return namespace['baseline'](engine,native,stage,policy,output)


def run(engine,native,stage,policy,output):
    proof=baseline(engine,native,stage,policy,output)
    if proof is not engine._v173_qualified_component:raise RuntimeError('ROOT419 fresh baseline identity')
    from binding419 import install
    binding=stage('ROOT419/current-target-backend-local-activation',lambda:install(engine,native))
    from actor419 import run as bounded
    extra=bounded(engine,native,stage,output)
    observed=engine._parent4_target391_binding.get('actual_metadata_matches',0)
    if not engine.is_draft and observed<=0:raise RuntimeError('ROOT419 current metadata checks missing')
    return dict(original173=proof,current_target_binding391=binding,actual_metadata_matches391=observed,
        bounded=extra,production_permission=False,full463=False,formal_Goodput=False,whole_AR=False)
