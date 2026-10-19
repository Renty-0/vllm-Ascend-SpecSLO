"""HOST-only temporary control, AST inverse and actual lifecycle checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType

P=Path(__file__).parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-adaptive80-first-layer-divergence-v379')]


def mc2_sample(x,weight,*,qualified=True):
    if not qualified:raise RuntimeError('original qualification check retained')
    local_projection=F.linear(x,weight)
    return local_projection


def main():
    global F
    from control380 import Control,project_mc2
    F=SimpleNamespace(linear=lambda x,w:('old',x,w))
    projected=project_mc2(mc2_sample,lambda x,w:('new',x,w))
    assert mc2_sample(1,2)==('old',1,2)and projected(1,2)==('new',1,2)
    try:projected(1,2,qualified=False)
    except RuntimeError:pass
    else:raise AssertionError('qualification branch must remain')
    class Column:
        token_pad_multiple=1
    class Head:
        token_pad_multiple=1
    modules=[(f'layers.{i}.self_attn.qkv_proj',Column())for i in range(64)]
    modules +=[(f'layers.{i}.mlp.gate_up_proj',Column())for i in range(64)]+[('lm_head',Head())]
    engine=SimpleNamespace(model=SimpleNamespace(named_modules=lambda:modules))
    name='vllm_ascend.spec_decode.pearl';originals={}
    for part in ['vllm_ascend','vllm_ascend.spec_decode',name]:
        originals[part]=sys.modules.get(part);sys.modules[part]=ModuleType(part)
    sys.modules[name].native_model=SimpleNamespace(NativeColumnLinear=Column,NativeLMHead=Head)
    try:
        for mode,count in [('none',0),('qkv120',64),('columns120',129)]:
            with Control(engine,mode)as state:
                assert sum(m.token_pad_multiple==120 for _,m in modules)==count
            assert all(m.token_pad_multiple==1 for _,m in modules)
            assert state.counts['original_column_padding_restored']and state.counts['original_mc2_dispatch_restored']
        try:
            with Control(engine,'columns120'):raise ValueError('exercise original cleanup on exception')
        except ValueError:pass
        assert all(m.token_pad_multiple==1 for _,m in modules)
    finally:
        for key,value in originals.items():
            if value is None:del sys.modules[key]
            else:sys.modules[key]=value
    from trace380 import Trace
    class Handle:
        def remove(self):pass
    class Module:
        def register_forward_pre_hook(self,f):return Handle()
        def register_forward_hook(self,f):return Handle()
    class Norm(Module):
        def forward_mc2(self,*args,**kwargs):return args
    class Attention(Module):
        def _write_to_cache(self,*args):return args
        def _fused_infer_attention(self,*args):return args
    class Layer(Module):
        def __init__(self):self.self_attn=Attention();self.input_layernorm=Norm()
        def named_modules(self):return [('',self),('self_attn',self.self_attn)]
    layer=Layer();engine=SimpleNamespace(model=SimpleNamespace(embed_tokens=Module(),layers=[layer,Layer()],norm=Norm()))
    original_norm=Norm.forward_mc2
    tracer=Trace(engine)
    assert layer.self_attn._write_to_cache('slot','key','value')==('slot','key','value')
    assert layer.self_attn._fused_infer_attention('q','metadata')==('q','metadata')
    tracer.close()
    assert Norm.forward_mc2 is original_norm and '_write_to_cache'not in layer.self_attn.__dict__
    assert '_fused_infer_attention'not in layer.self_attn.__dict__
    import guard380,runtime380
    ns=guard380.execution_namespace();source=runtime380.worker_source();compile(source,'actual380worker','exec')
    assert ns['ACTION']=='M80-original-GEMM120-three-eager-controls'
    assert ns['SCOPE']['finite_cases']==3 and ns['SCOPE']['independent_prefixes']==288
    assert source.count('numeric_cases=3')==2 and source.count('from probe380 import run')==1
    try:guard380.actual_audit('/nonexistent-HOST380-path')
    except FileNotFoundError:pass
    else:raise AssertionError('actual audit imported, expected no data')
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST380_GEMM_CONTROL_RESTORATION_AST_INVERSE_AND_ACTUAL_LIFECYCLE',
        NPU_used=False,original_qualification_branch_retained=True,exception_cleanup_passed=True,
        actual_guard_audit_imports_checked=True,models_qualified=False,Raw=None,Goodput=None)
    (P/'HOST380_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
