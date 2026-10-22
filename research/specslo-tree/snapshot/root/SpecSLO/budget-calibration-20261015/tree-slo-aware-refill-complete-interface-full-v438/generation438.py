"""Whole437 LIVE source with one current-SLO refill threshold expression."""
import ast,copy
from trace432 import generation as original


def project(source):
    before=ast.parse(source);tree=copy.deepcopy(before)
    funcs=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='admit_available']
    if len(funcs)!=1:raise RuntimeError('ROOT438 one current LIVE admission')
    fn=funcs[0]
    rows=[n for n in fn.body if isinstance(n,ast.Assign)and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_REFILL_LOW_WATER426']
    if len(rows)!=1 or ast.unparse(rows[0].value)!='32':raise RuntimeError('ROOT438 exact inherited lowwater32')
    old=rows[0].value;imp=ast.parse('from admission438 import low_water as _low_water438').body[0]
    fn.body.insert(0,imp);rows[0].value=ast.parse('_low_water438(request_params,active)',mode='eval').body
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    fn.body.remove(imp);rows[0].value=old
    if ast.dump(tree,include_attributes=False)!=ast.dump(before,include_attributes=False):raise RuntimeError('ROOT438 complete LIVE inverse after one scheduling expression')
    return result


def patch(source):return project(original(source))
