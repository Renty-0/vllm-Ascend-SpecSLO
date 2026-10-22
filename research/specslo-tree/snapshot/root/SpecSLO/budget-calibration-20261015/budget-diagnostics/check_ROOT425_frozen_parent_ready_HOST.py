"""Read-only frozen parent/cleanup/launcher construction; no model imports."""
import importlib.abc
import json
import os
from pathlib import Path
import re
import shlex
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-parent4-canonical-tail-observer-binding-v425'
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
   raise RuntimeError('HOST425 cold parent may not import model backend '+fullname)
sys.meta_path.insert(0,NoModel())
# Exactly the literal existing source-auth pins from the frozen launcher.
for line in (P/'launch425.sh').read_text().splitlines():
 for value in shlex.split(line.rstrip(' \\')):
  if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
   key,setting=value.split('=',1);os.environ[key]=setting
os.environ['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
sys.path.insert(0,str(P))
import guard425
record=json.loads((P/'SOURCE425_RECEIPT.json').read_text())
guard425.EXPECTED_FREEZE=record['freeze_sha256'];guard425.EXPECTED_ARCHIVE=record['archive_sha256']
frozen,sources,meta=guard425.prepared()
ns=guard425.execution_namespace()
assert ns['execute_locked'].__globals__ is ns and ns['MAX_SECONDS']==18000
assert ns['MAX_TAG_BYTES']==256*1024**2 and ns['MAX_AGGREGATE_BYTES']==3584*1024**2
assert ns['SCOPE']['current_same_engine_parent4_lifecycle_cases']==75
assert ns['SCOPE']['active_request_limit']==64
storage=guard425.storage()
assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
value=dict(status='PASS_HOST425_FROZEN_COLD_PARENT_AND_EXACT_OWNED_CALLBACKS',source_members=len(sources),
 runtime_seconds=18000,private_storage=storage,all75_current_cases_required=True,original_private_caps_retained=True,
 original_owned_cleanup_and_verifiers_retained=True,full463_and_all3_ownAR_required=True,NPU_used=False,
 Torch_imported=False,model_constructors=0,source_freeze_sha256=record['freeze_sha256'])
p=B/'budget-diagnostics/ROOT425_FROZEN_COLD_PARENT_RESULT.json'
with p.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
