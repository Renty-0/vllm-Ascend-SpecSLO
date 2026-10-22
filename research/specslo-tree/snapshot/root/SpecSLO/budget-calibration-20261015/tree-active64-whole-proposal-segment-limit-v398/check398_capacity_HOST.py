"""Validate the real admission/trace adapters and64-owner cohort boundary."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
for name in ('tree-active-window32-16-local-v160',
             'tree-causal80-small-cohort-role-aware-audit-full-v209',
             'tree-static-q5-M120-SLO-first-full463-own-AR-v257'):
    sys.path.append(str(B / name))


def rejects(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('invalid active/cohort ceiling accepted')


def main():
    import window398 as window
    import window257 as previous
    assert window.DEFAULT_WINDOWS == (64,)
    assert window.parse_windows('64') == (64,)
    for bad in (48, 32, 65, 0, True, 64.0):
        rejects(lambda: window.window(bad))
    source = 'def _generate_spec_rhythm_tree_decode(*, initial_batch_size, sentinel=None):\n    """Original caller argument."""\n    return initial_batch_size, sentinel\n'
    projected = window.adapt_source(source, 64)
    namespace = {}
    exec(compile(projected, 'ROOT398.actual-original-admission-adapter', 'exec'), namespace)
    marker = object()
    for incoming, expected in [(1, 1), (48, 48), (64, 64), (96, 64)]:
        assert namespace['_generate_spec_rhythm_tree_decode'](
            initial_batch_size=incoming, sentinel=marker) == (expected, marker)
    tree = ast.parse(projected)
    assert ast.unparse(tree.body[0].body[1]) == 'initial_batch_size = min(initial_batch_size, 64)'
    del tree.body[0].body[1]
    assert ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    assert previous.window(48) == 48
    rejects(lambda: previous.window(64))
    trace = window.trace_source(64)
    compile(trace, 'ROOT398.actual-production-trace-adapter', 'exec')
    text = ast.unparse(trace)
    assert '0 < peak <= 64' in text and 'admission_window160=64' in text
    assert 'actual_window_covered160=peak == 64' in text
    # Preserve the old Native/S35 scope flag; actual active64 coverage is
    # independently reported by observed peak, never assigned as qualification.
    assert 'actual_capacity64_scope_completed=False' in text
    from empty398 import audit_trace
    from empty257 import audit_trace as previous_audit
    def cohort(size):
        return dict(scope_count=size, standalone_prefill_calls=[dict(sequence_ids=list(range(size)))],
                    stream_events=[dict(request_index=i, finished=False) for i in range(size)]
                    + [dict(request_index=i, finished=True) for i in range(size)])
    positive = audit_trace(cohort(64))
    assert positive['cohort_count'] == 1 and len(positive['cohorts'][0]['owners']) == 64
    rejects(lambda: audit_trace(cohort(65)))
    rejects(lambda: previous_audit(cohort(64)))
    bad = cohort(64)
    bad['stream_events'].pop()
    rejects(lambda: audit_trace(bad))
    original_reference = (B / 'tree-static-q5-M120-SLO-first-full463-own-AR-v257/reference257.py').read_text()
    new_reference = (P / 'reference398.py').read_text()
    assert new_reference == original_reference.replace(
        'all(0<len(group)<=48 for group in groups)', 'all(0<len(group)<=64 for group in groups)')
    for name in ('target_step', 'frame'):
        old = next(n for n in ast.parse(original_reference).body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in ast.parse(new_reference).body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    import runtime398
    original_runtime = (B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text()
    actual, changes = runtime398.project_source(original_runtime)
    compile(actual, 'ROOT398.actual-full-worker', 'exec')
    assert "windows==(64,)" in actual and "default='64'" in actual
    assert 'from trace398 import run_windows' in actual and 'from audit398 import audit' in actual
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST398_REAL_ACTIVE64_ADAPTER_COHORT_REFERENCE_BOUNDARIES',
                  NPU_used=False, effective_active_ceiling=64, previous_active_ceiling=48,
                  configured_max_num_seqs=64, target_max_requests=24, target_candidates=96,
                  target_model_math_unchanged=True, independent_reference_math_unchanged=True,
                  actual_observed_peak_required_in_device_result=True,
                  Native_qualification_flags_assigned=False, production_qualified=False)
    (P / 'HOST398_CAPACITY_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
