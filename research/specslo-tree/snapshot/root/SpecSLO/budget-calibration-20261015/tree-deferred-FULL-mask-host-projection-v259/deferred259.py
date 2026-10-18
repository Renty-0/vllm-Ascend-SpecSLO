"""Defer FULL mask upload until the exact final physical mask is constructed.

This module only prepares functions. It does not execute devices or models.
All dynamic masks remain fresh. Original HOST validations, branch geometry,
slot mapping and physical projection semantics are retained by full inverses.
"""
import ast,copy,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent.parent
PACKET=B/'tree-packed-FULL-host-vector-metadata-v258/packet258.py'
PROJECTION=B/'tree-static-q5-M120-node-math-v255/projection255.py'
BRANCH=B/'tree-static-q5-M120-node-math-v255/branch_tail255.py'

def changed(source,replacements,label):
    result=source
    for before,after in replacements:
        if result.count(before)!=1:raise RuntimeError(label+' exact unique anchor '+before)
        result=result.replace(before,after)
    inverse=result
    for before,after in reversed(replacements):
        if inverse.count(after)!=1:raise RuntimeError(label+' exact unique inverse '+after)
        inverse=inverse.replace(after,before)
    if ast.dump(ast.parse(inverse),include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):
        raise RuntimeError(label+' full original AST inverse')
    return result

def host_factory(host):
    import packet258
    if Path(packet258.__file__).resolve()!=PACKET:raise RuntimeError('exact frozen packet258 module')
    function,proof=packet258.prepared(host)
    source=PACKET.read_text();node=copy.deepcopy(next(n for n in ast.walk(ast.parse(source))if isinstance(n,ast.FunctionDef)and n.name=='pack'))
    body=ast.unparse(node)
    body=changed(body,[("host.make_tree_fia_mask(masks).to(device=device, dtype=torch.bool)","host.make_tree_fia_mask(masks)")],'HOST CPU mask')
    namespace=dict(host=host)
    exec(compile(body,__file__+'::original-packet-CPU-mask-only','exec'),namespace)
    function.__globals__['_packet258_full']=namespace['pack']
    return function,dict(proof,raw_mask_upload_deferred=True,all_original_validation_preserved=True,
        packet_source_sha256=hashlib.sha256(source.encode()).hexdigest())

def branch_prepare(state):
    import branch_tail255
    if Path(branch_tail255.__file__).resolve()!=BRANCH:raise RuntimeError('exact original branch module')
    source=BRANCH.read_text();cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)and n.name=='State')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='prepare')
    body=changed(ast.unparse(node),[("desc['mask'].to(device=engine.device)","desc['mask']")],'branch CPU mask')
    namespace=dict(vars(branch_tail255));exec(compile(body,__file__+'::original-branch-CPU-mask-only','exec'),namespace)
    import types
    return types.MethodType(namespace['prepare'],state)

def cpu_indices(projection,geometry,old_extent,mask_capacity):
    import torch
    # Geometry already has an immutable bounded cache and a key covering these
    # exact row mappings and capacities. Never cache any current dynamic mask.
    if '_mask259_CPU'not in geometry:
        geometry['_mask259_CPU']=dict(
            source=torch.tensor([m['old_group']*old_extent+m['old_query']for m in projection['real_moves']],dtype=torch.long,device='cpu'),
            destination=torch.tensor([(m['new_row']//5)*mask_capacity+m['new_row']%5 for m in projection['real_moves']],dtype=torch.long,device='cpu'),
            phantom=torch.tensor([(r//5)*mask_capacity+r%5 for r in projection['phantom_rows']],dtype=torch.long,device='cpu'))
    return geometry['_mask259_CPU']

def projected():
    import projection255
    if Path(projection255.__file__).resolve()!=PROJECTION:raise RuntimeError('exact original physical projector')
    source=PROJECTION.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='transform')
    body=ast.unparse(node)
    replacements=[
        ("geometry = CACHE.get(p, device, int(mask_in.shape[2]), mask_capacity)",
         "geometry = CACHE.get(p, device, int(mask_in.shape[2]), mask_capacity)\n    if mask_in.device.type != 'cpu':\n        raise RuntimeError('ROOT259 deferred raw mask must be CPU')\n    _cpu = cpu_indices(p, geometry, int(mask_in.shape[2]), mask_capacity)"),
        ("mask = torch.ones((24, 1, mask_capacity, 4096), dtype=torch.bool, device=device)",
         "mask = torch.ones((24, 1, mask_capacity, 4096), dtype=torch.bool, device='cpu')"),
        ("geometry['destination']","_cpu['destination']"),
        ("geometry['source']","_cpu['source']"),
        ("geometry['phantom']","_cpu['phantom']"),
        ("tree_attention_mask=mask,","tree_attention_mask=mask.to(device=device),")]
    body=changed(body,replacements,'physical CPU mask')
    namespace=dict(vars(projection255),cpu_indices=cpu_indices)
    exec(compile(body,__file__+'::original-projection-CPU-mask-only','exec'),namespace)
    return namespace['transform']
