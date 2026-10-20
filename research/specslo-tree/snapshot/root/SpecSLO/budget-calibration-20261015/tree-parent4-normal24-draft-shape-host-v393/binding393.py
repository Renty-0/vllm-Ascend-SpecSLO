"""Same finite current-M120 binding with the new local role/shape context."""
import ast
from pathlib import Path
import binding391 as original
install=original.install
_source=Path(original.__file__).read_text()
_before='import live199';_after='import live393 as live199'
if _source.count(_before)!=1:raise RuntimeError('ROOT393 exact finite context import')
_changed=_source.replace(_before,_after)
if _changed.replace(_after,_before)!=_source:raise RuntimeError('ROOT393 context inverse')
_node=next(n for n in ast.parse(_changed).body if isinstance(n,ast.FunctionDef)and n.name=='diagnostic')
_namespace=dict(vars(original))
exec(compile(ast.fix_missing_locations(ast.Module(body=[_node],type_ignores=[])),
    __file__+'::original391-binding-with-new-local-scope','exec'),_namespace)
diagnostic=_namespace['diagnostic']
