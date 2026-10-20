"""Exercise independent metadata with actual CPU Native plans and packer.

The CPU packer's output is represented with the production FIA metadata ABI;
no model weight, forward, device allocation or numerical qualification occurs.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace

P=Path(__file__).parent;B=P.parent
for folder in ('tree-bounded-parent4-shared4-eager-diagnostic-v199',
        'tree-static-q5-M120-node-math-v255','tree-static-q5-immutable-index-cache-host-v247',
        'tree-best257-uniform-KV-buckets-full-v376'):
    sys.path.insert(0,str(B/folder))
import base199
import torch
from vllm_ascend.spec_decode.pearl.native_model import NativeQwen2ForCausalLM
from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan
import metadata390


def main():
    assert not torch.npu.is_initialized()
    tables=[[-1]*32 for _ in range(527)]
    for i in range(24):tables[i][:21]=list(range(21*i,21*i+21))
    # Max real48 pages is not assumed; this is a HOST metadata carrier. Only
    # actually queried pages need fit the original544 page pool.
    tables[526][0]=526
    attention=SimpleNamespace(uses_paged_attention=True,block_size=128,
        key_cache=torch.zeros((544,128,1),dtype=torch.bfloat16),
        value_cache=torch.zeros((544,128,1),dtype=torch.bfloat16))
    model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.empty(1)),
        max_model_len=4096,layers=[SimpleNamespace(self_attn=attention)])
    packer=MethodType(NativeQwen2ForCausalLM.make_tree_attention_metadata,model)
    def abi(*args,**kwargs):
        x,pos,meta=packer(*args,**kwargs)
        assert x.device.type=='cpu'and not meta.use_fused_infer_attention
        return x,pos,replace(meta,use_fused_infer_attention=True,
            block_tables=meta.request_block_tables,attention_mask=None)
    model.make_tree_attention_metadata=abi
    engine=SimpleNamespace(model=model,draft_vocab_size=151936,device=torch.device('cpu'),
        cache_allocation=SimpleNamespace(block_tables=tables),
        _branch_tail_state=SimpleNamespace(uniform_KV_bucket376_enabled=True))
    observations=[]
    for n in (1,16,24):
        for prefix in (128,511,1500,2555):
            plans=[]
            for i in range(24):
                full=cached_cpu_tree_speculation_plan(2,4,prefix if i<n else 0,4096,candidate_budget=8)
                plans.append(pack_selected_tree_plan(full,range(4)))
            roots=list(range(100,124));rows=[[300+i+j for j in range(4)]for i in range(24)]
            frame=(plans,roots,rows,list(range(24)),n)
            x,pos,meta=metadata390.physical(engine,frame)
            expected=((prefix+5+511)//512)*512
            assert x.numel()==120 and tuple(meta.sequence_lens)==(expected,)*24
            assert meta.actual_seq_lengths_q==tuple(range(5,121,5))
            assert tuple(meta.tree_attention_mask.shape)==(24,1,10,4096)
            assert torch.equal(x,torch.tensor([v for i in range(24)for v in [roots[i],*rows[i]]]))
            assert pos[:5].tolist()==list(range(prefix,prefix+5))
            assert meta.slot_mapping[:5].tolist()==[tables[0][k//128]*128+k%128 for k in range(prefix,prefix+5)]
            changed,_,_=metadata390.physical(engine,frame,changed=True)
            assert changed[::5].tolist()==[r+1 for r in roots]
            assert torch.equal(x.reshape(24,5)[:,1:],changed.reshape(24,5)[:,1:])
            if n<24:assert not meta.tree_attention_mask[n,0,:5,0].any()
            observations.append(dict(N=n,prefix=prefix,extent=expected,all120_input_tokens=True,
                original_real_KV_slots=True,all24_Q5_groups=True,changed_roots_only=True))
    invalid=(frame[0],frame[1],frame[2],frame[3],25)
    try:metadata390.physical(engine,invalid)
    except RuntimeError as error:assert 'finite uniform B4'in str(error)
    else:raise AssertionError('out of scope N25')
    assert not torch.npu.is_initialized()
    result=dict(status='PASS_HOST390_ACTUAL_NATIVE_PACKER_AND_STATIC_PROJECTION',
        NPU_used=False,model_forward_called=False,model_weights_loaded=False,
        production_FIA_ABI_on_CPU_carriers=True,observations=observations,
        numerical_qualification=False,production_permission=False,Raw=None,Goodput=None)
    (P/'HOST390_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cases=len(observations),NPU_used=False)))


if __name__=='__main__':main()
