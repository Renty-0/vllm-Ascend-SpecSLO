"""Candidate optimization: cache only immutable physical index tensors.

Dynamic token/position/slot, page-table and complete mask operations retain the
original transform body. No dynamic tensor or neural result is cached. This
module performs no device work at import and is not production-qualified.
"""
from collections import OrderedDict
from pathlib import Path
import ast,hashlib

ORIGINAL=Path('/root/SpecSLO/budget-calibration-20261015/tree-static-q5-projection-host-v243/projection243.py')
ORIGINAL_SHA='f6f596f704d55cf672e8b05c265d6de707ec8a0485daf41c4040de9de817390e'


class IndexCache:
    def __init__(self,maximum=256):
        if type(maximum)is not int or maximum<1:raise ValueError('positive bounded immutable-index cache')
        self.maximum=maximum;self.entries=OrderedDict();self.hits=0;self.misses=0;self.evictions=0

    def get(self,projection,device,old_extent,mask_capacity):
        import torch
        order=projection['physical_to_original'];inverse=projection['original_to_physical']
        sources=[x for x in projection['physical_group_sources']if x is not None]
        moves=projection['real_moves'];phantoms=projection['phantom_rows']
        key=(str(device),tuple(order),tuple(inverse),tuple(sources),old_extent,mask_capacity,
             tuple((m['old_group'],m['old_query'],m['new_row'])for m in moves),tuple(phantoms))
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key]
        value=dict(index=torch.tensor(order,dtype=torch.long,device=device),
            cpu_index=torch.tensor(order,dtype=torch.long,device='cpu'),
            inverse=torch.tensor(inverse,dtype=torch.long,device=device),
            selected=torch.tensor(sources,dtype=torch.long,device=device),
            source=torch.tensor([m['old_group']*old_extent+m['old_query']for m in moves],dtype=torch.long,device=device),
            destination=torch.tensor([(m['new_row']//5)*mask_capacity+m['new_row']%5 for m in moves],dtype=torch.long,device=device),
            phantom=torch.tensor([(r//5)*mask_capacity+r%5 for r in phantoms],dtype=torch.long,device=device))
        self.entries[key]=value;self.misses+=1
        if len(self.entries)>self.maximum:self.entries.popitem(last=False);self.evictions+=1
        return value


CACHE=IndexCache()


def prepared(expected_sha):
    source=ORIGINAL.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=expected_sha:raise RuntimeError('exact original physical projection source')
    tree=ast.parse(source)
    plan_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='plan')
    body=ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='transform'))
    replacements=[
        ('    device=inputs.device',
         "    device=inputs.device\n    geometry=CACHE.get(p,device,int(original_mask.shape[2]),mask_capacity)"),
        ("index=torch.tensor(p['physical_to_original'],dtype=torch.long,device=device)","index=geometry['index']"),
        ("cpu_index=torch.tensor(p['physical_to_original'],dtype=torch.long,device='cpu')","cpu_index=geometry['cpu_index']"),
        ("inverse=torch.tensor(p['original_to_physical'],dtype=torch.long,device=device)","inverse=geometry['inverse']"),
        ("selected=torch.tensor(real_sources,dtype=torch.long,device=device)","selected=geometry['selected']"),
        ("source=torch.tensor([m['old_group']*old_extent+m['old_query']for m in p['real_moves']],dtype=torch.long,device=device)","source=geometry['source']"),
        ("destination=torch.tensor([(m['new_row']//5)*mask_capacity+m['new_row']%5 for m in p['real_moves']],dtype=torch.long,device=device)","destination=geometry['destination']"),
        ("phantom=torch.tensor([(r//5)*mask_capacity+r%5 for r in p['phantom_rows']],dtype=torch.long,device=device)","phantom=geometry['phantom']"),
    ]
    changed=body
    for before,after in replacements:
        if changed.count(before)!=1:raise RuntimeError('exact immutable index allocation anchor '+before)
        changed=changed.replace(before,after)
    restored=changed
    for before,after in reversed(replacements):
        if restored.count(after)!=1:raise RuntimeError('exact inverse cache anchor')
        restored=restored.replace(after,before)
    if ast.dump(ast.parse(restored),include_attributes=False)!=ast.dump(ast.parse(body),include_attributes=False):
        raise RuntimeError('entire original projection AST inverse')
    namespace=dict(CACHE=CACHE)
    exec(compile(ast.unparse(plan_node)+'\n\n'+changed,str(Path(__file__).resolve())+'::original-transform-with-immutable-indices','exec'),namespace)
    return namespace['transform']


transform=prepared(ORIGINAL_SHA)
