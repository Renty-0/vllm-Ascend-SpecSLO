"""Actual sanitized startup environment imports; no model or kernel call."""
import importlib
import json
from pathlib import Path
modules=('runtime317','probes317','oracle317','oracle311','oracle255','reference_schedule317','bank317','baseline317',
         'static_state317','raw_controls317','benchmark317','audit_static_calls317','metadata285',
         'plan285','write313','kernels292','neural_oracle231','dual_bank_v146','capture_key296','device296')
resolved={}
for name in modules:
    module=importlib.import_module(name)
    resolved[name]=module.__file__
from oracle317 import cases,CASE_COUNT
assert CASE_COUNT==len(cases())==546
p=Path(__file__).resolve().parent/'HOST317_ACTUAL_STARTUP_IMPORT_RESULT.json'
with p.open('x')as f:json.dump(dict(passed=True,modules=resolved,actual_startup_environment=True,cases=546,model_constructed=False,kernels_launched=False),f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,imported_modules=len(resolved),cases=546,models_loaded=False,kernels_launched=False)))
