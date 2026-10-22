"""HOST-only projection of actual425 admission; no model or runtime installer."""
import ast
import copy
import hashlib
from functools import lru_cache
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent/'tree-parent4-canonical-tail-observer-binding-v425/HOST425.generated-LIVE.py'


@lru_cache(maxsize=1)
def sources():
    tree = ast.parse(SOURCE.read_text())
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available']
    if len(found) != 1:
        raise RuntimeError('HOST426 exact actual LIVE admission body required')
    before = copy.deepcopy(found[0])
    after = copy.deepcopy(before)
    guard = ast.parse('if active:\n return closed').body[0]
    if ast.dump(after.body[1], include_attributes=False) != ast.dump(guard, include_attributes=False):
        raise RuntimeError('HOST426 exact original empty-cohort guard')
    after.body[1] = ast.parse('if active and len(active)>_REFILL_LOW_WATER426:\n return closed').body[0]
    undo = [(after.body, 1, copy.deepcopy(before.body[1]))]
    for name, replacement in (
        ('_root256_reserved_pages', '64+sum((int(local_states[i].prompt_length)+local_states[i].max_tokens+5+127)//128 for i in active)'),
        ('vacant_capacity', 'min(initial_batch_size-len(active)-_v75_hooks.reserved_slots(), _REFILL_MAX426) if active else initial_batch_size-len(active)-_v75_hooks.reserved_slots()'),
    ):
        nodes = [n for n in ast.walk(after) if isinstance(n, ast.Assign)
                 and len(n.targets)==1 and ast.unparse(n.targets[0])==name]
        if len(nodes)!=1:
            raise RuntimeError('HOST426 exact original assignment '+name)
        node=nodes[0];undo.append((node,'value',copy.deepcopy(node.value)))
        node.value=ast.parse(replacement,mode='eval').body
    homes = [i for i,n in enumerate(after.body) if isinstance(n,ast.Assign)
             and len(n.targets)==1 and ast.unparse(n.targets[0])=='homes']
    if len(homes)!=1:
        raise RuntimeError('HOST426 minimum refill before home/cache/prefill/publication changes')
    minimum=ast.parse('if active and len(admitted)<_REFILL_MIN426:\n return closed').body[0]
    after.body.insert(homes[0],minimum)
    projected=ast.unparse(ast.fix_missing_locations(after))+'\n'
    after.body.remove(minimum)
    for owner,key,old in reversed(undo):
        if isinstance(key,int):owner[key]=old
        else:setattr(owner,key,old)
    if ast.dump(after,include_attributes=False)!=ast.dump(before,include_attributes=False):
        raise RuntimeError('HOST426 complete original admission inverse')
    return ast.unparse(before)+'\n',projected


def construct(namespace, *, low_water=32, minimum_refill=8, maximum_refill=32):
    if not (type(low_water) is int and 0<low_water<64
            and type(minimum_refill) is int and type(maximum_refill) is int
            and 1<=minimum_refill<=maximum_refill<=32):
        raise ValueError('HOST426 explicit capacity64/qualified future max32 proposal')
    original,projected=sources()
    old=dict(namespace)
    new=dict(namespace,_REFILL_LOW_WATER426=low_water,
             _REFILL_MIN426=minimum_refill,_REFILL_MAX426=maximum_refill)
    exec(compile(original,__file__+'::original425-admission','exec',dont_inherit=True),old)
    exec(compile(projected,__file__+'::HOST-only-nonempty-proposal','exec',dont_inherit=True),new)
    return old['admit_available'],new['admit_available']


def descriptor():
    before,after=sources()
    return dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        original_function_sha256=hashlib.sha256(before.encode()).hexdigest(),
        proposed_function_sha256=hashlib.sha256(after.encode()).hexdigest(),entire_AST_inverse=True,
        original_arrival_clock_and_prefill_clock_and_incumbent_charge_unchanged=True,
        original_first_token_append_and_publication_unchanged=True,
        additional_incumbent_full_output_page_reservations=True,
        global_active_limit=64,page_pool=544,serving_installed=False,
        NN_permission=False,Raw=None,Goodput=None)
