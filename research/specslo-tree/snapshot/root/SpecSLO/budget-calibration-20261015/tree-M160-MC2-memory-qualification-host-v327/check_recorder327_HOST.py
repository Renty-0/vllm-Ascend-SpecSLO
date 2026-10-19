"""CPU capture ownership regression; no model, NPU, files of tensor values."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import json
from pathlib import Path
import sys
import torch
from recorder327 import MemoryInputs

torch.set_num_threads(1)
p=MemoryInputs()
p.maybe_capture(None,None,None,None,projection_kind='attention',layer_index=0)
assert not p._payloads
try:
    with p.capture(dict(N=31,budgets=[4]*31)):
        raise AssertionError('invalid case admitted')
except ValueError:
    pass
originals={}
with torch.inference_mode(),p.capture(dict(N=32,budgets=[4]*32)):
    for layer in (0,31,63):
        for kind,k in (('attention',3072),('down',8576)):
            x=torch.full((160,k),.5,dtype=torch.bfloat16)
            w=torch.empty((5120,k),dtype=torch.bfloat16)
            w[0,0]=layer+1
            residual=torch.full((160,5120),.25,dtype=torch.bfloat16)
            gamma=torch.ones(5120,dtype=torch.bfloat16)
            p.maybe_capture(x,w,residual,gamma,projection_kind=kind,layer_index=layer)
            value=p._payloads[layer,kind]
            assert value['activation'].data_ptr()!=x.data_ptr()and value['residual'].data_ptr()!=residual.data_ptr()
            assert value['weight'].data_ptr()==w.data_ptr()and value['gamma'].data_ptr()==gamma.data_ptr()
            x.zero_();residual.zero_()
            assert bool((value['activation']==.5).all())and bool((value['residual']==.25).all())
            assert w[0,0].item()==layer+1 and bool((gamma==1).all())
            originals[layer,kind]=(w,gamma)
    try:
        p.maybe_capture(x,w,residual,gamma,projection_kind=kind,layer_index=layer)
    except RuntimeError:
        pass
    else:
        raise AssertionError('duplicate model observation accepted')
assert p.sealed and len(p.payloads())==6 and not p.active
try:
    p.payloads()[0,'attention']={}
except TypeError:
    pass
else:
    raise AssertionError('unsealed payload map')
value=p._payloads[0,'attention'];saved=value['original_weight']
value['original_weight']=torch.empty_like(saved)
try:
    p.validate_readonly_references()
except RuntimeError:
    pass
else:
    raise AssertionError('changed model parameter binding accepted')
value['original_weight']=saved;p.validate_readonly_references()
assert 'torch_npu'not in sys.modules
summary=p.summary()
result=dict(passed=True,entries=6,actual_CPU_tensor_storage_checks=True,activation_and_residual_cloned=True,
    model_weight_and_gamma_storage_shared_readonly=True,original_model_weight_sentinels_and_gamma_unchanged=True,
    duplicate_and_wrong_case_and_changed_weight_binding_rejected=True,
    extra_cloned_bytes=sum(r['extra_cloned_bytes']for r in summary['entries']),
    production_capture_installed=False,real_model_weights_loaded=False,NPU_used=False,Goodput=None)
output=Path(__file__).with_name('HOST327_MEMORY_RECORDER_RESULT.json')
with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
