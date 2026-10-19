import runtime326
import oracle326,metadata326
from pathlib import Path
import json
f=oracle326.specialized('reference')
assert f.__globals__['_checked_reference']is metadata326.checked
run=oracle326.specialized('run')
assert run.__globals__['candidate']is oracle326.candidate
assert run.__globals__['reference']is oracle326.reference
proof=dict(passed=True,cases=oracle326.CASE_COUNT,actual_reference_checker=metadata326.__file__,original_oracle=oracle326.original.__file__,models_loaded=False)
assert proof['cases']==532
Path('/root/SpecSLO/budget-calibration-20261015/tree-M160-S32-N32-target-math-cost-v326/HOST326_COMPILED_REFERENCE_BINDING.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
