"""Reuse original CPU Native fixture setup without any model call."""
import ast,copy,dataclasses,importlib.util,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
SOURCE=B/'tree-static-q5-M120-full463-own-AR-v256/check_full256_HOST.py'

def build():
    spec=importlib.util.spec_from_file_location('_ROOT259_original_CPU_fixture',SOURCE)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    node=copy.deepcopy(next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='main'))
    cut=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='forward'for t in n.targets))
    node.name='_ROOT259_fixture';node.body=node.body[:cut]+[ast.Return(value=ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]))]
    ns=dict(vars(fixture));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE)+'::original-fixture-prefix','exec'),ns)
    sys.path.insert(0,str(B/'tree-packed-FULL-host-vector-metadata-v258'))
    return ns[node.name]()

def equal(a,b):
    import torch
    for av,bv in zip(a[:2]+a[3:4],b[:2]+b[3:4],strict=True):
        assert av.device==bv.device and av.dtype==bv.dtype and av.stride()==bv.stride()and torch.equal(av.cpu(),bv.cpu())
    for field in dataclasses.fields(a[2]):
        av,bv=getattr(a[2],field.name),getattr(b[2],field.name)
        if isinstance(av,torch.Tensor):
            assert isinstance(bv,torch.Tensor)and av.device==bv.device and av.dtype==bv.dtype and av.stride()==bv.stride()and torch.equal(av.cpu(),bv.cpu()),field.name
        else:assert av==bv,field.name
    assert a[2].block_tables is a[2].request_block_tables and b[2].block_tables is b[2].request_block_tables
    assert a[4]==b[4]
