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
P=B/'tree-M80-own-geometry-whole-model-v457'
F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
REPO=F/'source/vllm-ascend-hust';VENDOR=F/'runtime/mc2-vendor'
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(VENDOR)+':'+str(REPO/'vllm_ascend/_cann_ops_custom/vendors/custom_transformer')
os.environ['LD_LIBRARY_PATH']=':'.join([str(VENDOR/'op_api/lib'),str(REPO/'vllm_ascend'),str(REPO/'vllm_ascend/lib64'),str(REPO/'build/temp.linux-aarch64-cpython-311'),os.environ.get('LD_LIBRARY_PATH','')])
os.environ['HCCL_DETERMINISTIC']='true';os.environ['HCCL_OP_EXPANSION_MODE']='AIV'
import torch
import torch_npu
if torch_npu.npu.is_initialized():raise RuntimeError('ROOT458 HOST check started with an NPU context')
def forbid_device(*args,**kwargs):raise RuntimeError('ROOT458 HOST profile normalization may not initialize an NPU')
torch_npu.npu._lazy_init=forbid_device
torch_npu.npu.init=forbid_device
from vllm_ascend.spec_decode.pearl.mc2 import normalize_mc2_profile, _mc2_profile_sha256
path=F/'runtime/mc2-profile.json'
profile=normalize_mc2_profile(path)
keys=sorted(key for key in profile.entries if key[0]==80)
assert len(keys)==2 and {key[1]for key in keys}=={3072,8576}
assert len(profile.entries)==48 and profile.minimum_samples==31 and profile.minimum_speedup==1.02
from startup_config_v132 import make_config
config=make_config()
assert config.max_num_seqs==64 and config.enable_mc2 is True and config.mc2_profile==str(path)
file_sha=hashlib.sha256(path.read_bytes()).hexdigest()
normalized_sha=_mc2_profile_sha256(profile)
assert file_sha=='ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c'
assert normalized_sha=='67feca4c924ce95b252812d2db606a65a43942c472c429a71d12e5270442deb7'
canonical_document=profile._document()
assert hashlib.sha256(json.dumps(canonical_document,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()==normalized_sha
assert not torch_npu.npu.is_initialized()
out=dict(status='PASS_HOST458_ACTUAL_NATIVE_ORIGINAL_PROFILE_FILE_AND_CANONICAL_DIGEST',
 profile_path=str(path),original_config_profile_path=config.mc2_profile,file_sha256=file_sha,
 normalized_profile_sha256=normalized_sha,normalized_shapes=len(profile.entries),exact80_shapes=[list(v)for v in keys],
 original_Native_normalizer_used=True,original_Native_digest_function_used=True,
 actual457_reported_profile_sha256_matches=True,NPU_initialized=False,actual_model_constructors=0,
 normalized_document=canonical_document,production_permission=False,Raw=None,Goodput=None)
destination=B/'budget-diagnostics/ROOT458_ACTUAL_NATIVE_ORIGINAL_PROFILE_DIGEST_HOST.json'
with destination.open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps({k:v for k,v in out.items()if k!='normalized_document'}))
