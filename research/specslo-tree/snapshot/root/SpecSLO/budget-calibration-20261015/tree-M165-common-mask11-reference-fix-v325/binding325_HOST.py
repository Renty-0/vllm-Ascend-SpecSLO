import runtime325
import oracle325,metadata323
f=oracle325.specialized('reference')
assert f.__globals__['_checked_reference'] is metadata323.checked
run=oracle325.specialized('run')
assert run.__globals__['candidate']is oracle325.candidate
assert run.__globals__['reference']is oracle325.reference
import json
from pathlib import Path
proof=dict(passed=True,reference_compiled_checked_origin=metadata323.__file__,actual_compiled_reference_checker_correct=True,model_cases=oracle325.CASE_COUNT,original_oracle_source=oracle325.original.__file__,models_loaded=False)
assert proof['model_cases']==546
Path('/root/SpecSLO/budget-calibration-20261015/tree-M165-common-mask11-reference-fix-v325/HOST325_ACTUAL_COMPILED_REFERENCE_BINDING.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
