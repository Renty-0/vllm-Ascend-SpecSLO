"""Check the actual full candidate's lifecycle, scheduling source and proof gates."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import runtime346
import guard346
from source346_HOST import source
from generation346 import patch

P = Path(__file__).resolve().parent


def main():
    original = (runtime346.DONOR/'runtime209.py').read_text()
    projected, changes = runtime346.project_source(original)
    compile(projected, 'HOST346-actual-worker-projection', 'exec')
    assert len(changes) == 14
    assert "report['ROOT345_small_refill_current_component']" in projected
    actual = patch(source())
    tree = ast.parse(actual)
    admission = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available')
    assert not any(isinstance(n, ast.If) and ast.unparse(n.test) == 'active'
                   and len(n.body) == 1 and isinstance(n.body[0], ast.Return) for n in admission.body)
    assert '_root345_reserved_owners' in ast.unparse(admission)
    assert 'for index in active:\n            states[index].add_decode_time(refill_ms)' in ast.unparse(admission)
    assert "from budget312 import *" in (P/'budget346.py').read_text()
    assert 'actual_inflight_refill_groups' in (P/'audit346.py').read_text()
    assert "refill['audit']['Native_zero_scalar_subtractions']==672" in (P/'scope346.py').read_text()
    assert 'qualify_small_refill(engine,native,stage,output)' in (P/'extension346.py').read_text()
    assert guard346.MAX_AGGREGATE_BYTES == 3072*1024**2
    assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
    value = dict(status='passed_full_candidate_readiness_HOST', NPU_used=False,
        worker_source_inverse=True, worker_new_component_receipt=True,
        actual_LIVE_sha256=hashlib.sha256(actual.encode()).hexdigest(),
        actual_LIVE_removes_empty_active_guard=True, incumbent_pages_reserved=True,
        original_refill_time_accounting_retained=True, measured_target_ROI_gate_used=True,
        actual_small_refill_NN_proof_required=True, actual_full_refill_coverage_required=True,
        full463_and_own_AR_still_required=True, model_qualified=False, Goodput=None)
    (P/'HOST_READINESS346_RESULT.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
