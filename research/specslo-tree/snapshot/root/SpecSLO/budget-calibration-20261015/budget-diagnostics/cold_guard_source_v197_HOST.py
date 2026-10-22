"""Cold source/guard only: never invoke model/process/device helpers."""
import importlib.abc,importlib.util,json,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-empty-active-cohort-causal80-active32-full-v197';G=B/'budget-diagnostics/v197-root-execution'
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('197 cold HOST no backend '+name)
sys.meta_path.insert(0,Firewall());start=time.monotonic()
spec=importlib.util.spec_from_file_location('_V197_cold_guard',G/'guard197.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=False);inputs=guard.guard_inputs()
try:guard.prepared()
except FileNotFoundError as error:
 assert str(P/'FROZEN_SOURCE197.json')in str(error)
else:raise AssertionError('source-only missing ROOT freeze must not launch')
assert not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules)
row=dict(schema='V197.cold-actual-source-state.v1',status='passed',candidate=str(P),source_state=state,guard_inputs=inputs,elapsed_seconds=time.monotonic()-start,ROOT_lock_opened=False,entry_or_child_launched=False,device_or_process_query=False,Torch_imported=False,NPU_used=False,model_constructed=False,parent_archive_verified=False,parent_archive_obligation='ROOT prepare must verify P173/P148/P160 parent archives in place before launch',guard_missing_ROOT_freeze_rejected=True)
path=B/'budget-diagnostics/V197_COLD_ACTUAL_GUARD_SOURCE_STATE_HOST_RESULT.json';assert not path.exists();path.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');print(path);print(guard.sha(path),row['elapsed_seconds'])
