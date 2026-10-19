"""Synthetic FIA group invariance diagnostic; no model or service metrics."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import json
from pathlib import Path
import sys
import torch
import torch_npu

P=Path(__file__).resolve().parent
torch.npu.set_device(1)
torch.npu.config.allow_internal_format=True
device=torch.device('npu:1')
generator=torch.Generator(device='cpu').manual_seed(324)
shape=(544,128,3,128)
key=torch.randn(shape,dtype=torch.bfloat16,generator=generator,device='cpu').to(device)
value=torch.randn(shape,dtype=torch.bfloat16,generator=generator,device='cpu').to(device)
records=[]
for groups in (24,32,33):
    query_host=torch.randn((5*groups,24,128),dtype=torch.bfloat16,generator=generator,device='cpu')
    root=query_host[0].clone()
    query_host[::5].copy_(root)
    query=query_host.to(device)
    table=torch.tensor([list(range(20))+[543]*12]*groups,dtype=torch.int32,device=device)
    for capacity in (10,11):
        mask_host=torch.ones((groups,1,capacity,4096),dtype=torch.bool,device='cpu')
        mask_host[:,:,1:5,0]=False
        mask=mask_host.to(device)
        for precise in (1,2):
            for prefix in (0,97,127,128,1505,2555):
                mask_host[:,:,0,:]=True
                mask_host[:,:,0,:prefix+1]=False
                mask.copy_(mask_host)
                baseline=None
                for variant in (0,1,0):
                    payload=query_host.clone()
                    if variant:
                        payload[::5].copy_(root+torch.tensor(.125,dtype=torch.bfloat16,device='cpu'))
                    query.copy_(payload)
                    out=torch.empty_like(query)
                    lse=torch.empty(1,dtype=query.dtype,device=device)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,key=key.view(544,128,-1),value=value.view(544,128,-1),
                        atten_mask=mask,block_table=table,input_layout='TND',block_size=128,
                        actual_seq_lengths=list(range(5,5*groups+1,5)),actual_seq_lengths_kv=[2560]*groups,
                        num_key_value_heads=3,num_heads=24,scale=128**-.5,sparse_mode=1,inner_precise=precise,
                        out=[out,lse])
                    torch.npu.synchronize()
                    result=out.cpu()
                    assert bool(torch.isfinite(result).all())
                    roots=result[::5]
                    diff=(roots.float()-roots[0].float()).abs().flatten(1).amax(1)
                    equal=(roots.view(torch.int16)==roots[0].view(torch.int16)).flatten(1).all(1)
                    restored=None
                    if variant==0:
                        if baseline is None:
                            baseline=result.clone()
                        else:
                            restored=torch.equal(result.view(torch.int16),baseline.view(torch.int16))
                            assert restored,'same-input whole output did not restore'
                    records.append(dict(groups=groups,M=5*groups,mask_capacity=capacity,inner_precise=precise,prefix=prefix,variant=variant,
                        same_Q_K_V_mask_and_page_table_for_all_root_queries=True,
                        all_roots_raw16_equal=bool(equal.all()),different_root_groups=(~equal).nonzero().flatten().tolist(),
                        root_max_abs_by_group=diff.tolist(),whole_output_restored=restored))
                    print(json.dumps({k:v for k,v in records[-1].items()if k not in ('root_max_abs_by_group','different_root_groups')}),flush=True)
output=P/'FIA324_GROUP_INVARIANCE_RESULT.json'
with output.open('x')as f:
    json.dump(dict(schema='ROOT324.synthetic-FIA-group-invariance.v1',rows=records,models_loaded=False,model_qualified=False,performance=False,Goodput=None),f,indent=2)
print(json.dumps(dict(status='complete',rows=len(records),noninvariant_rows=sum(not x['all_roots_raw16_equal']for x in records),models_loaded=False)),flush=True)
