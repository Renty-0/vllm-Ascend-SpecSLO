"""Actual strict Native profile normalization and frozen config, no NPU model."""
from pathlib import Path
import dataclasses
import hashlib
import json
import os
import sys

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
B=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(B/'tree-M160-real32-inmemory-MC2-qualification-v454'))
from host_env454 import setup
setup()
P=Path(__file__).resolve().parent
F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
REPO=F/'source/vllm-ascend-hust';VENDOR=F/'runtime/mc2-vendor'
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(VENDOR)+':'+str(REPO/'vllm_ascend/_cann_ops_custom/vendors/custom_transformer')
os.environ['LD_LIBRARY_PATH']=':'.join([str(VENDOR/'op_api/lib'),str(REPO/'vllm_ascend'),str(REPO/'vllm_ascend/lib64'),str(REPO/'build/temp.linux-aarch64-cpython-311'),os.environ.get('LD_LIBRARY_PATH','')])
os.environ['HCCL_DETERMINISTIC']='true';os.environ['HCCL_OP_EXPANSION_MODE']='AIV'
import torch
import torch_npu
if torch_npu.npu.is_initialized():raise RuntimeError('ROOT455 HOST check started with an NPU context')
def forbid_device(*args,**kwargs):raise RuntimeError('ROOT455 HOST profile normalization may not initialize an NPU')
torch_npu.npu._lazy_init=forbid_device
torch_npu.npu.init=forbid_device
from vllm_ascend.spec_decode.pearl.mc2 import normalize_mc2_profile
path=P/'mc2-profile-with-qualified160.json'
profile=normalize_mc2_profile(path)
keys=sorted(key for key in profile.entries if key[0]==160)
assert len(keys)==2 and {key[1]for key in keys}=={3072,8576}and all(key[2]==5120 for key in keys)
assert len(profile.entries)==50 and profile.minimum_samples==31 and profile.minimum_speedup==1.02
from startup_config_v132 import make_config as old_config
from config455 import make_config
before=old_config();after=make_config()
changes=[field.name for field in dataclasses.fields(before)if getattr(before,field.name)!=getattr(after,field.name)]
assert changes==['mc2_profile']and after.max_num_seqs==64 and after.enable_mc2 is True
assert not torch_npu.npu.is_initialized()
out=dict(status='PASS_HOST455_ACTUAL_NATIVE_NORMALIZER_AND_PROFILE_ONLY_CONFIG_CHANGE',
    profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),normalized_shapes=len(profile.entries),
    exact160_shapes=[list(v)for v in keys],changed_config_fields=changes,max_num_seqs=64,
    original_Native_normalizer_used=True,new_profile_used_by_model=False,
    NPU_initialized=False,actual_model_constructors=0,production_permission=False,Raw=None,Goodput=None)
with(P/'HOST455_NATIVE_PROFILE_RESULT.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out))
