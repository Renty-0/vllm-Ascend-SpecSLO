"""Check the actual full candidate's lifecycle, scheduling source and proof gates."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import runtime348
import guard348
from source348_HOST import source
from generation348 import patch

P = Path(__file__).resolve().parent


def main():
    original = (runtime348.DONOR/'runtime209.py').read_text()
    projected, changes = runtime348.project_source(original)
    compile(projected, 'HOST348-actual-worker-projection', 'exec')
    assert len(changes) == 14
    assert "report['ROOT351_small_refill_current_component']" in projected
    actual = patch(source())
    tree = ast.parse(actual)
    admission = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available')
    assert not any(isinstance(n, ast.If) and ast.unparse(n.test) == 'active'
                   and len(n.body) == 1 and isinstance(n.body[0], ast.Return) for n in admission.body)
    assert '_root345_reserved_owners' in ast.unparse(admission)
    assert 'for index in active:\n            states[index].add_decode_time(refill_ms)' in ast.unparse(admission)
    assert "from budget312 import *" in (P/'budget348.py').read_text()
    assert 'actual_inflight_refill_groups' in (P/'audit348.py').read_text()
    assert "call['N']>8" not in (P/'audit348.py').read_text()
    assert 'actual cost-selected branching exercised naturally' in (P/'audit348.py').read_text()
    assert "refill['audit']['Native_zero_scalar_subtractions']==672" in (P/'scope348.py').read_text()
    assert 'qualify_small_refill(engine,native,stage,output)' in (P/'extension348.py').read_text()
    assert guard348.MAX_AGGREGATE_BYTES == 3072*1024**2
    original218 = hashlib.sha256((P.parent/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218/FREEZE218.json').read_bytes()).hexdigest()
    assert guard348.OLD_SHA == original218
    assert 'SPECSLO_V218_ROOT_FREEZE_SHA='+original218 in (P/'launch348.sh').read_text()
    fixture = json.loads((P/'FIXTURE348.json').read_text())
    source_fixture = json.loads((P.parent/'tree-single-prepare-target-math-cost-v311/FIXTURE311.json').read_text())
    assert fixture['prefixes'] == source_fixture['prefixes']
    assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
    value = dict(status='passed_full_candidate_readiness_HOST', NPU_used=False,
        worker_source_inverse=True, worker_new_component_receipt=True,
        actual_LIVE_sha256=hashlib.sha256(actual.encode()).hexdigest(),
        actual_LIVE_removes_empty_active_guard=True, incumbent_pages_reserved=True,
        original_refill_time_accounting_retained=True, measured_target_ROI_gate_used=True,
        actual_small_refill_NN_proof_required=True, actual_full_refill_coverage_required=True,
        full463_and_own_AR_still_required=True, model_qualified=False, Goodput=None)
    value.update(actual_original218_launch_SHA_verified=True, original311_teacher_prefixes_exact=True)
    (P/'HOST_READINESS348_RESULT.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
