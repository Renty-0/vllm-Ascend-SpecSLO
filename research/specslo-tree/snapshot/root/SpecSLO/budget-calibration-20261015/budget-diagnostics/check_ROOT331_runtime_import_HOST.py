import json
import pickle
from pathlib import Path
import runtime331
import runtime329
from check331 import ready
ready("fea0e3c69858b0e8b4b35f82332a103ed256f9a0733980541b5c6785bb3df569")
assert runtime331.previous is runtime329
assert runtime331.previous.original.worker is runtime331.worker331
assert runtime331.base_worker is runtime329.worker329
assert pickle.loads(pickle.dumps(runtime331.worker331)) is runtime331.worker331
cfg=runtime329.configuration.make_config()
assert cfg.spec_rhythm_verification_budget==128
report=dict(status="HOST_pass",worker331_actual_runtime326_registration=True,worker329_capacity_wrapper_preserved=True,actual_constructor_C128=True,worker_pickling_roundtrip=True,no_model_constructed=True,no_NPU_model_run=True,no_performance_claim=True)
Path("/root/SpecSLO/budget-calibration-20261015/tree-FIA160-projection165-MC2-target-math-cost-v331/HOST331_RUNTIME_IMPORT_RESULT.json").write_text(json.dumps(report,indent=2)+"\n")
print("HOST331_RESULT "+json.dumps(report))
