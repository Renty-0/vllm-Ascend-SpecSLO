"""Original120-row padding with the explicitly expanded HOST B1..8 domain."""
import ast,inspect,textwrap
_PAD=None
def prepared():
 global _PAD
 if _PAD is not None:return _PAD
 import padding255
 source=textwrap.dedent(inspect.getsource(padding255.pad));tree=ast.parse(source)
 changes=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.ImportFrom)and n.module in('mapper255','target_layout'):
   changes.append((n,n.module));n.module='plan269'
 if len(changes)!=2:raise RuntimeError('HOST269 exactly two declared domain imports')
 changed=ast.unparse(ast.fix_missing_locations(tree))+'\n'
 for n,old in changes:n.module=old
 if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):
  raise RuntimeError('HOST269 complete padding source inverse')
 ns=dict(vars(padding255));exec(compile(changed,__file__+'::original-padding-new-domain','exec'),ns)
 _PAD=ns['pad'];return _PAD
def pad(*args,**kwargs):return prepared()(*args,**kwargs)
