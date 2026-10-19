"""Execute original decoder/MLP/norm/binder code with CPU transport doubles."""
import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import sys
sys.path.insert(0,"/root/SpecSLO/budget-calibration-20261015/tree-M160-actual165-MC2-bridge-host-v330")
from bridge334 import install

P=Path(__file__).resolve().parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
SOURCE=N/'native_model.py'
MC2=N/'mc2.py'

def extract(path, name, parent=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body
    if parent: nodes=next(x for x in nodes if isinstance(x,ast.ClassDef) and x.name==parent).body
    return next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name==name)

records=[]
seal=object()
def ticket(qualification,x,weight,residual,gamma,**kwargs):
    return NS(qualification=qualification, x=x, weight=weight, residual=residual, gamma=gamma)

def operator(x, weight, residual, gamma, **kwargs):
    rows=x.shape[0]
    if rows!=165: raise AssertionError('native MC2 call must really receive165')
    if torch.count_nonzero(x[-5:]) or torch.count_nonzero(residual[-5:]):
        raise AssertionError('padding must be zero')
    if kwargs['prequalification'].x is not x: raise AssertionError('original binding ownership')
    records.append(dict(M=rows,K=x.shape[1],weight=id(weight),gamma=id(gamma),flush=kwargs['flush_chain']))
    normalized=x[:,:1].expand(-1,5120).contiguous()
    if kwargs.get('chain_state')is not None:
        records[-1]['actual_chain_state']=True
        return normalized,residual.clone(),kwargs['chain_state'].clone()
    return normalized,residual.clone()

ns=dict(__name__=__name__, torch=torch, math=math, prod=math.prod, dataclass=dataclass, MC2Qualification=lambda qualified,reason:NS(qualified=qualified,reason=reason),
        _MC2_STATIC_ROUTE_SEAL=seal, _dtype_name=lambda x:str(x.dtype),
        _bind_mc2_dispatch_ticket=ticket,
        matmul_allreduce_add_rmsnorm_or_fallback=operator)
node=extract(MC2,'bind_mc2_static_route')
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(MC2),'exec'),ns)
binder=ns['bind_mc2_static_route']
ns['bind_mc2_static_route']=binder

def method(name,parent):
    fn=extract(SOURCE,name,parent)
    space=dict(ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),str(SOURCE),'exec'),space)
    return space[name]

# The actual original chain class, including order/commit/final-drain checks.
chain_node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef)and n.name=='_MC2IntraLayerChain')
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),chain_node],type_ignores=[])),str(SOURCE),'exec'),ns)
Chain=ns['_MC2IntraLayerChain']

class Norm:
    forward_mc2=method('forward_mc2','NativeRMSNorm')
    def __init__(self): self.weight=torch.ones(5120,dtype=torch.bfloat16);self.eps=1e-6
    def __call__(self,x,residual=None): return x if residual is None else (x,residual)

class Row:
    def __init__(self,k): self.weight=torch.empty((5120,k),dtype=torch.bfloat16);self.bias=None
    def __call__(self,x): return x[:,:1].expand(-1,5120).contiguous()

class Attention:
    def __init__(self): self.o_proj=Row(3072)
    def __call__(self,positions,x,metadata,return_pre_projection=False):
        out=x[:,:1].expand(-1,3072).contiguous()
        return out if return_pre_projection else self.o_proj(out)

class MLP:
    forward_down_mc2=method('forward_down_mc2','NativeQwen2MLP')
    def __init__(self): self.down_proj=Row(8576)
    def _activate(self,x): return x[:,:1].expand(-1,8576).contiguous()
    def _maybe_capture_down(self,*a,**kw): pass

class Layer:
    forward=method('forward','NativeQwen2DecoderLayer')
    def __call__(self,*args,**kwargs):return self.forward(*args,**kwargs)

profile=NS(metadata=dict(operator='actual_qualified_165_test_double'))
context=NS(size=3,rank=0,group=None)
layer=Layer();layer.enable_mc2=True;layer.enable_down_mc2=True;layer.mc2_routes_frozen=True
layer.mc2_profile=profile;layer.context=context;layer.layer_index=0;layer.mc2_input_capture=None
layer.input_layernorm=Norm();layer.post_attention_layernorm=Norm();layer.self_attn=Attention();layer.mlp=MLP()
final=Norm()

def route(weight,gamma):
    return NS(seal=seal,profile_identity=id(profile),operator_kind=profile.metadata['operator'],
              weight_identity=id(weight),gamma_identity=id(gamma),weight_shape=tuple(weight.shape),
              weight_stride=weight.stride(),gamma_shape=tuple(gamma.shape),gamma_stride=gamma.stride(),
              device='cpu',activation_layout='contiguous',weight_layout='contiguous',
              residual_layout='contiguous',gamma_layout='contiguous',dtype=str(torch.bfloat16),
              input_size=weight.shape[1],output_size=5120,
              decisions={160:NS(qualified=False),165:NS(qualified=True)},disabled_reason='',
              enabled=True,tp_rank_size=3,tp_rank_id=0,epsilon=1e-6,is_trans_b=True,group_tp='HOST-only',layer_index=0,projection_kind='attention'if weight.shape[1]==3072 else'down')
layer.mc2_attention_route=route(layer.self_attn.o_proj.weight,layer.post_attention_layernorm.weight)
layer.mc2_down_route=route(layer.mlp.down_proj.weight,final.weight)
class Model:
    forward=method('forward','NativeQwen2ForCausalLM')
model=Model()
model.layers=[layer];model.norm=final;model.track_cache_finiteness=False
model.modules=lambda:[layer.input_layernorm,layer.post_attention_layernorm,final]
model.embed_tokens=lambda ids:ids.to(torch.bfloat16)[:,None].expand(-1,5120).contiguous()
model._mc2_chained_row_counts=frozenset({165});model._mc2_deferred_row_counts=frozenset()
model._mc2_chain_zero=torch.zeros((64,4),dtype=torch.int64)
original_model_code=Model.forward.__code__
original_layer_code=Layer.forward.__code__; original_down_code=MLP.forward_down_mc2.__code__
bridge=install(model,NS(bind_mc2_static_route=binder,NativeRMSNorm=Norm))
x=torch.arange(160,dtype=torch.bfloat16)[:,None].expand(-1,5120).contiguous()
res=(x+2).contiguous();positions=torch.arange(160)
metadata=NS(tree_attention=True,actual_seq_lengths_q=tuple(range(5,161,5)))
with torch.inference_mode():
    model_result=model.forward(torch.arange(160),positions,metadata)
    assert torch.equal(model_result,x)
    chain=Chain(model._mc2_chain_zero,1,defer_read_done=False)
    result=layer.forward(positions,x,res,metadata,input_is_normalized=True,
                         return_normalization_state=True,next_layernorm=final,mc2_chain=chain)
    chain.finish()
assert Model.forward.__code__ is original_model_code
assert result[2] is True and torch.equal(result[0],x) and torch.equal(result[1],res)
assert len(records)==4 and {r['K']for r in records}=={3072,8576}
assert all(r['M']==165 and r['flush']is True and r['actual_chain_state']is True for r in records)
assert chain.next_step==2 and chain._pending_key is None
assert Layer.forward.__code__ is original_layer_code and MLP.forward_down_mc2.__code__ is original_down_code
assert binder is ns['bind_mc2_static_route'] and bridge.pending=={}
# Outside tree scope, exact original160 route remains unqualified; no hidden165 admission.
outside=bridge.bind(layer.mc2_attention_route,profile,
    x[:,:3072].contiguous(),layer.self_attn.o_proj.weight,res,layer.post_attention_layernorm.weight)
assert outside.qualification.qualified is False
# Actual original binder rejects changed parameter identity even for a correctly padded input.
bridge.active_layer=0
try:
    bridge.bind(layer.mc2_attention_route,profile,x[:,:3072].contiguous(),
                layer.self_attn.o_proj.weight.clone(),res,layer.post_attention_layernorm.weight)
except RuntimeError as error:
    assert 'ownership' in str(error)
else: raise AssertionError('changed weight must be rejected')
finally: bridge.active_layer=None
assert bridge.pending=={}
report=dict(status='HOST_pass',original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    original_binder_source_sha256=hashlib.sha256(MC2.read_bytes()).hexdigest(),
    original_decoder_MLP_norm_binder_code_executed_with_CPU_operator_double=True,
    physical_attention_queries=160,real_MC2_mock_received165=True,actual_original_model_forward_and_chain_class_executed=True,original165_profile_admission_retained=True,every_projection_flushed=True,deferred_read_done_enabled=False,actual_chain_finish_two_operations=True,
    both_attention_and_down_covered=True,all160_rows_and_both_outputs_exact=True,
    actual_padding_suffix_zero=True,unchanged_profile160_refuses=True,
    changed_weight_original_identity_check_rejects=True,original_class_code_unmodified=True,
    no_models_loaded=True,no_NPU_run=True,model_qualified=False,Goodput=None,
    bridge=bridge.summary(),operator_calls=records)
(P/'HOST334_CHAINED_FLUSH_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('bridge','operator_calls')}))
