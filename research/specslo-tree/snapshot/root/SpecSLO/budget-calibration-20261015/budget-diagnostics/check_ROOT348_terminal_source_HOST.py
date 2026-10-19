"""Exercise the actual read-only terminal loader and live-refill scope checks."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType

B = Path(__file__).resolve().parent.parent
D = B/'budget-diagnostics'
P = B/'tree-coalesced-live-refill-M120-singleton-capture-full-v348'


def main():
    reader = D/'audit_ROOT348_terminal_full_HOST.py'
    call = next(n for n in ast.walk(ast.parse(reader.read_text()))
                if isinstance(n, ast.Call) and ast.unparse(n.func) == 'original.definitions')
    changes = ast.literal_eval(call.args[2])
    spec = importlib.util.spec_from_file_location('_HOST346_original_terminal_reader', D/'audit_v209_terminal_HOST.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    sys.meta_path.insert(0, original.Firewall())
    sys.path[:0] = [str(P), str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),
                   str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')]
    original.expected_current_geometry = lambda value: None
    original.definitions(original.PARENT_AUDITOR,
                         {'admission_and_full_statistics','common_full_statistics'}, changes)
    # Isolate only the newly projected cohort/accounting wrapper. The unchanged
    # complete metrics body still runs against real data in the terminal reader.
    fn = original.admission_and_full_statistics
    namespace = dict(fn.__globals__, common_full_statistics=lambda traces, reports: {})
    projected = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    trace = dict(scope_count=6, requests=[dict(max_tokens=256) for _ in range(6)],
        tokenized_inputs=[[0]*128 for _ in range(6)], absolute_arrivals=[90.]*6,
        standalone_prefill_calls=[dict(sequence_ids=[0,1]), dict(sequence_ids=[2,3,4,5])],
        stream_events=[dict(request_index=i, finished=False, first_token_wall_seconds160=100.+i)
                       for i in range(6)]+[dict(request_index=i, finished=True) for i in range(6)],
        sealed_before=True, sealed_after=True, cache_released=True,
        configured_capacity=64, admission_window160=48, active_peak=6,
        actual_capacity64_scope_completed=False,
        decode_counters=dict(spec_rhythm_slo_aware_admission_enabled=1,
            spec_rhythm_target_slo_first_enabled=1, spec_rhythm_prefill_coalesce_enabled=0,
            spec_rhythm_prefill_coalesce_deferred_polls=0))
    config = dict(spec_rhythm_prefill_coalesce_min_requests=1,
                  spec_rhythm_prefill_coalesce_max_wait_ms=0, spec_rhythm_verification_budget=96)
    reports = [dict(configuration_before_constructor=dict(config),
                    configuration_after_constructor=dict(config)) for _ in range(4)]
    traces = [copy.deepcopy(trace) for _ in range(4)]
    value = projected(traces, reports)
    assert value['no_prefill_pause_memberships_for_incumbents'] is False
    assert value['empty_active_cohort_replay']['actual_inflight_refill_groups'] == 1
    assert value['empty_active_cohort_replay']['cohorts'][1]['incumbent_owners'] == [0,1]
    bad = copy.deepcopy(traces)
    bad[1]['tokenized_inputs'][0] = [0]*(128*545)
    try: projected(bad, reports)
    except RuntimeError: pass
    else: raise AssertionError('overcommitted live refill accepted')
    assert not original.Firewall.attempts
    assert not any(n.split('.')[0] in original.BACKENDS for n in sys.modules)
    result = dict(status='passed_terminal_source_projection_and_scope_HOST', NPU_used=False,
        backend_import_attempts=[], actual_original_arithmetic_loader_executed=True,
        changes=len(changes), actual_live_refill_and_incumbent_page_fields_required=True,
        actual_projected_wrapper_executed=True, fake_clock_fixture_only=True,
        original_constructor_and_scheduling_counters_checked=True,
        incumbent_prefill_pause_reported_truthfully=True, unsafe_page_inventory_rejected=True,
        original_complete_metrics_and_own_AR_bodies_retained=True,
        no_complete_model_run_or_performance_qualification=True)
    (D/'HOST_ROOT348_TERMINAL_SOURCE_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
