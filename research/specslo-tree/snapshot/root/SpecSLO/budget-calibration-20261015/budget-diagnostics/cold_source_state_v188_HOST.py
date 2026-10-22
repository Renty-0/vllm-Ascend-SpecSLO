"""Actual guard/source closure check only; no prepare/freeze/device action."""
import importlib.abc,importlib.util,json,sys,time,hashlib
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-parent3-union3-padding-observer-devicefix-full-v188'
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V188 cold HOST no backend '+name)
sys.meta_path.insert(0,Firewall());start=time.perf_counter()
q=D/'v188-root-execution/guard188.py';spec=importlib.util.spec_from_file_location('_V188_actual_cold_guard',q);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
state=module.source_state(archive=True);guards=module.guard_inputs()
assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
row=dict(schema='V188.actual-cold-guard-source-state-HOST.v1',status='passed',candidate=str(P),source_state=state,guard_inputs=guards,elapsed_seconds=time.perf_counter()-start,Torch_imported=False,NPU_used=False,model_constructed=False,freeze_created=False,old36_oracle_carriers_rerun=False,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out=D/'V188_COLD_ACTUAL_GUARD_SOURCE_STATE_HOST_RESULT.json';assert not out.exists();out.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),seconds=row['elapsed_seconds'])))
