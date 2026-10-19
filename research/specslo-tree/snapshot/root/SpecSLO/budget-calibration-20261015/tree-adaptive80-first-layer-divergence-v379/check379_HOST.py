"""Pure HOST shape alignment, reversible runtime and actual audit import path."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(P))


def main():
    from trace379 import aligned_shape,Trace
    checks=[((120,5120),(80,5120),0),((2880,128),(1920,128),0),
            ((1,24,120,128),(1,24,80,128),2),((5120,),(5120,),None)]
    for a,b,expected in checks:assert aligned_shape(a,b)==expected
    for a,b in [((120,32),(80,64)),((80,32),(120,32)),((120,),(80,1))]:
        try:aligned_shape(a,b)
        except ValueError:pass
        else:raise RuntimeError('ROOT379 invalid tensor alignment accepted')
    class Handle:
        def __init__(self):self.removed=False
        def remove(self):self.removed=True
    class Module:
        def register_forward_pre_hook(self,f):return Handle()
        def register_forward_hook(self,f):return Handle()
    class Norm(Module):
        calls=0
        def forward_mc2(self,*args,**kwargs):
            Norm.calls+=1
            return args
    class Layer(Module):
        def __init__(self):self.input_layernorm=Norm()
        def named_modules(self):return [('',self),('input_layernorm',self.input_layernorm)]
    old=Norm.forward_mc2
    layer=Layer();engine=SimpleNamespace(model=SimpleNamespace(embed_tokens=Module(),layers=[layer,Layer()],norm=Norm()))
    trace=Trace(engine);handles=list(trace.handles)
    value=layer.input_layernorm.forward_mc2('first','second',static_route=SimpleNamespace(layer_index=0,projection_kind='attention'))
    assert value==('first','second')and Norm.calls==1
    trace.close();assert Norm.forward_mc2 is old and all(h.removed for h in handles)
    import guard379,runtime379
    ns=guard379.execution_namespace();source=runtime379.worker_source()
    compile(source,'ROOT379.actual-retained-worker','exec')
    assert ns['ACTION']=='adaptive80-first-layer-stage-observation'
    assert ns['SCOPE']['finite_cases']==1 and ns['SCOPE']['independent_prefixes']==144
    assert source.count('numeric_cases=1')==2 and source.count('from probe379 import run')==1
    try:guard379.actual_audit(Path('/nonexistent-HOST379-observation-path'))
    except FileNotFoundError:pass
    else:raise RuntimeError('ROOT379 audit probe expected missing data, after successful actual imports')
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST379_ALIGN_HOOK_LIFECYCLE_WORKER_AND_ACTUAL_GUARD_IMPORTS',
        NPU_used=False,alignment_cases=7,actual_mc2_impl_forwarded_once=True,
        class_method_and_hooks_restored=True,actual_audit_import_path_tested_without_extra_CORE_injection=True,
        original_owned_lifecycle_preserved=True,model_qualified=False,Raw=None,Goodput=None)
    (P/'HOST379_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
