"""Actual source/archive guard call; no freeze/prepare/device entry."""
import hashlib,importlib.abc,importlib.util,json,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-slo-admission1-vacancy4-cap4-active32-full-v191'
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V191 coldHOST backend forbidden '+name)
sys.meta_path.insert(0,Firewall());start=time.perf_counter();q=D/'v191-root-execution/guard191.py'
spec=importlib.util.spec_from_file_location('_V191_actual_cold_guard',q);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);inputs=guard.guard_inputs();assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
row=dict(schema='V191.actual-cold-guard-HOST.v1',status='passed',candidate=str(P),source_state=state,guard_inputs=inputs,elapsed_seconds=time.perf_counter()-start,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),Torch_imported=False,NPU_used=False,model_constructed=False,freeze_created=False,old190_4HOST_and185_suites_rerun=False,current190_actual_read=False)
out=D/'V191_COLD_ACTUAL_GUARD_SOURCE_STATE_HOST_RESULT.json';assert not out.exists();out.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),seconds=row['elapsed_seconds'])))
