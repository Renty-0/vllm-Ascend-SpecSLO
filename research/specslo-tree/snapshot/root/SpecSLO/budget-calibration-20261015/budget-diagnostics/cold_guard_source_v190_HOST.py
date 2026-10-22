"""Cold actual source identity/archive guard; never a model/device invocation."""
import hashlib,importlib.abc,importlib.util,json,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-vacancy-cohort-prefill-cap4-active32-full-v190';G=B/'budget-diagnostics/v190-root-execution'
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V190 cold HOST forbids '+name)
sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P));start=time.perf_counter()
spec=importlib.util.spec_from_file_location('_V190_cold_actual_guard',G/'guard190.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
s=guard.source_state(archive=True);g=guard.guard_inputs();assert not any(k.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend')for k in sys.modules)
row=dict(status='passed',candidate=str(P),elapsed_seconds=time.perf_counter()-start,source_state=s,guard_inputs=g,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),Torch_imported=False,NPU_used=False,model_constructed=False,freeze_created=False,old_suites_rerun=False)
p=B/'budget-diagnostics/V190_COLD_ACTUAL_GUARD_SOURCE_STATE_HOST_RESULT.json';assert not p.exists();p.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),elapsed_seconds=row['elapsed_seconds'])))
