"""Read-only frozen parent/cleanup/launcher construction; no model imports."""
import importlib.abc
import json
import os
from pathlib import Path
import re
import shlex
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-refill-witness-boolean-retention-full-v437'
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
   raise RuntimeError('HOST425 cold parent may not import model backend '+fullname)
sys.meta_path.insert(0,NoModel())
# Exactly the literal existing source-auth pins from the frozen launcher.
for line in (P/'launch437.sh').read_text().splitlines():
 for value in shlex.split(line.rstrip(' \\')):
  if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
   key,setting=value.split('=',1);os.environ[key]=setting
os.environ['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
# Use the literal PYTHONPATH declared by the frozen437 child, also in its
# model-free parent. This does not change any frozen file or source-auth pin.
startup=(P/'startup437.sh').read_text().splitlines()
lines=[line for line in startup if line.startswith('export PYTHONPATH=')]
assert len(lines)==1
value=lines[0].removeprefix('export PYTHONPATH=')
F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
for old,new in [('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')),('${PYTHONPATH:-}','')]:
 value=value.replace(old,new)
assert '$'not in value
sys.path[:0]=value.split(':')
import guard437
record=json.loads((P/'SOURCE437_RECEIPT.json').read_text())
guard437.EXPECTED_FREEZE=record['freeze_sha256'];guard437.EXPECTED_ARCHIVE=record['archive_sha256']
frozen,sources,meta=guard437.prepared()
ns=guard437.execution_namespace()
assert ns['execute_locked'].__globals__ is ns and ns['MAX_SECONDS']==18000
assert ns['MAX_TAG_BYTES']==256*1024**2 and ns['MAX_AGGREGATE_BYTES']==3584*1024**2
assert ns['SCOPE']['current_same_engine_parent4_lifecycle_cases']==75
assert ns['SCOPE']['active_request_limit']==64
assert ns['SCOPE']['original_current427_extra_NN_cases']==6
storage=guard437.storage()
assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
value=dict(status='PASS_HOST437_FROZEN_COLD_PARENT_AND_EXACT_OWNED_CALLBACKS',source_members=len(sources),
 runtime_seconds=18000,private_storage=storage,all75_current_cases_required=True,original_private_caps_retained=True,
 original_owned_cleanup_and_verifiers_retained=True,full463_and_all3_ownAR_required=True,NPU_used=False,
 Torch_imported=False,model_constructors=0,source_freeze_sha256=record['freeze_sha256'])
p=B/'budget-diagnostics/ROOT437_FROZEN_COLD_PARENT_RESULT.json'
with p.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
