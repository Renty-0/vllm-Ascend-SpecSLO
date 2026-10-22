"""Only full parent receive routing changes; original numerical worker retained."""
from pathlib import Path
import ast,copy,sys
P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'tree-adaptive-retained-CANN-path-full-v461'))
from check462 import ready
import runtime460
def source():
    before=runtime460.source();after=before
    changes=(('from wait436 import receive_qualification','from wait462 import receive_qualification'),
             ("__import__('wait436').snapshot()","__import__('wait462').snapshot()"))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT462 exact original parent-only receive routing')
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT462 full original numerical worker source inverse')
    return after
def build_main():
    ready();old=runtime460.build_runtime()
    ns=dict(old['main'].__globals__,__name__='ROOT462.full-parent-qualification-wait-only')
    nodes=[copy.deepcopy(node)for node in ast.parse(source()).body if isinstance(node,ast.FunctionDef)and node.name=='main']
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::original-main-with-wait-margin','exec'),ns)
    if ns['worker'] is not runtime460.worker:raise RuntimeError('ROOT462 unchanged actual numerical worker')
    return ns['main']
if __name__=='__main__':build_main()()
