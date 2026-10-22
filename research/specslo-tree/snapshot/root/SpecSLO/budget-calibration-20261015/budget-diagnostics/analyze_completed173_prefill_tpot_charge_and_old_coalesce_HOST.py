"""Bounded, read-only arithmetic/source audit. Never imports Torch or a model.

This is an analysis of an already completed V173 trace and two old result files.
It neither executes a candidate nor changes its timing or admission policy.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import statistics
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust')
P148 = B / 'tree-high-normal-cap4-native-families-v148'
P173 = B / 'tree-causal80-real16-bankfix-full-v173'
OLD = Path('/root/data/nano-pearl-benchmark-results/20260917-specslo-further-optimization')
RAW = P173 / 'actual/causal80-bankfix-v173-full463-v1-overlay177/windows/window32/full463/rank1-full463-actual-trace.json.gz'
MANIFEST = Path('/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl')
OUTPUT = D / 'V173_STANDALONE_PREFILL_TPOT_CHARGE_AND_OLD_COALESCE4_PROVENANCE_HOST.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    assert sys.version_info[:2] == (3, 11)
    assert not OUTPUT.exists(), 'Preserve any prior output instead of replacing it'
    sources = [
        RAW, MANIFEST,
        P148 / 'runtime_tree_decode.py', P148 / 'poisson_coverage_probe.py',
        P148 / 'prefill_exact.py', P148 / 'no_mixed_prefill_v143.py',
        P173 / 'trace173.py', P173 / 'extension173.py',
        NATIVE / 'vllm_ascend/spec_decode/pearl/spec_rhythm.py',
        NATIVE / 'vllm_ascend/spec_decode/pearl/admission.py',
        NATIVE / 'examples/benchmark_nano_pearl_speculative.py',
        NATIVE / 'tools/specslo_experiments/run_online_coalesce1_candidate.sh',
        Path('/root/SpecSLO/SPECSLO_FINAL_EXPERIMENT_HANDOFF_zh.md'),
        OLD / 'online_coalesce1_tp3_vs_tp4_audit_20260918_zh.md',
        OLD / 'slo-aware-admission-author463-v1/result.json',
        OLD / 'slo-aware-admission-author463-v1/manifest.sha256',
        OLD / 'slo-aware-admission-author463-v2-linear/result.json',
        D / 'SLO_AWARE_ADMISSION_COALESCING_HISTORY_AND_CURRENT_FEASIBILITY_HOST.json',
        Path(__file__).resolve(),
    ]
    pins = {str(p): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size} for p in sources}
    trace = json.loads(gzip.decompress(RAW.read_bytes()))
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    assert trace['manifest_sha256'] == sha(MANIFEST.read_bytes())
    assert len(rows) == len(trace['generate_results']) == 463
    assert all(x['ignore_eos'] is True and x['max_tokens'] == 256 for x in trace['generate_results'])
    assert all(len(x['completion_token_ids']) == 256 for x in trace['generate_results'])

    # Original admit_available emits all new roots after one helper charge. The
    # order of first callbacks and helper sequence_ids identifies that boundary.
    # Stream finished events remove an owner before the next admission pause.
    calls = trace['standalone_prefill_calls']
    assert len(calls) == trace['decode_counters']['spec_rhythm_prefill_batches'] == 427
    assert all(set(c) == {'sequence_ids', 'cached_tokens', 'prompt_lengths',
                         'precomputed_target_tokens', 'draft_prefill_completed',
                         'prompt_sha256', 'uncached_queries'} for c in calls)
    assert all(c['precomputed_target_tokens'] is False and c['draft_prefill_completed'] is False for c in calls)
    active, seen = set(), set()
    next_call, pending_roots = 0, []
    membership_rows = []
    per_request = Counter({i: 0 for i in range(463)})
    first_wall = {}
    for event in trace['stream_events']:
        owner = event['request_index']
        assert type(owner) is int and 0 <= owner < 463
        if owner not in seen:
            if not pending_roots:
                assert next_call < len(calls)
                pending_roots = list(calls[next_call]['sequence_ids'])
                assert pending_roots and pending_roots[0] == owner
                assert not (set(pending_roots) & seen)
                charged = sorted(active)
                membership_rows.append({'pause': next_call,
                                        'new_owners': list(pending_roots),
                                        'charged_old_owners': charged})
                per_request.update(charged)
                next_call += 1
            assert pending_roots.pop(0) == owner
            assert len(event['token_ids']) == 1 and event['finished'] is False
            first_wall[owner] = event['first_token_wall_seconds160']
            seen.add(owner)
            active.add(owner)
        else:
            assert not pending_roots, 'No incumbent token is emitted inside the root-delivery loop'
        if event['finished']:
            assert owner in active
            active.remove(owner)
    assert next_call == 427 and not pending_roots and not active and seen == set(range(463))
    charged_count_histogram = Counter(len(x['charged_old_owners']) for x in membership_rows)
    request_count_histogram = Counter(per_request.values())
    assert sum(per_request.values()) == sum(len(x['charged_old_owners']) for x in membership_rows) == 12744
    assert charged_count_histogram[31] == 369 and charged_count_histogram[30] == 28
    zero_pause = [i for i in range(463) if not per_request[i]]
    assert zero_pause == [461, 462]
    assert not trace['logical_events'] and all(not x['decode_timeline'] for x in trace['generate_results'])
    counters = trace['decode_counters']
    refill_ms = counters['spec_rhythm_online_refill_ms']
    assert all(x['spec_rhythm']['spec_rhythm_online_refill_ms'] == refill_ms for x in trace['generate_results'])
    assert trace['phase_seconds']['target'] == trace['generate_results'][0]['decode_phase_seconds']['target']
    result_by_id = {x['request_id']: x for x in trace['generate_results']}
    ttft = [1000 * (first_wall[i] - result_by_id[rows[i]['request_id']]['arrival_ts']) for i in range(463)]
    assert min(ttft) >= 0

    historical = []
    for name in ('slo-aware-admission-author463-v1', 'slo-aware-admission-author463-v2-linear'):
        path = OLD / name / 'result.json'
        old = json.loads(path.read_text())
        run = old['results'][0]
        metrics = run['slo']['request_metrics']
        by_id = {x['request_id']: x for x in metrics}
        assert set(by_id) == {x['request_id'] for x in rows}
        assert old['request_manifest_sha256'] == trace['manifest_sha256']
        origins = [by_id[x['request_id']]['arrival_ts'] - x['arrival_offset_sec'] for x in rows]
        # Large absolute float timestamp addition rounds sub-microsecond digits.
        origin = origins[0]
        roundtrip_error = max(abs(by_id[x['request_id']]['arrival_ts'] - (origin + x['arrival_offset_sec'])) for x in rows)
        assert roundtrip_error <= 2.5e-7
        display_errors = [by_id[x['request_id']]['arrival_offset_sec'] - x['arrival_offset_sec'] for x in rows]
        env = old['runtime_environment']
        historical.append({
            'result_path': str(path), 'run_identity': name,
            'source_commit_or_per_run_source_hash': None,
            'source_identity_limit': 'This old run directory retains result.json and manifest.sha256 only. Its result records configuration, not a source revision/archive. Current retained benchmark source explains the schema but is not claimed to be that exact historical source snapshot.',
            'draft_model': old['draft_model'], 'target_model': old['target_model'],
            'TP': [old['draft_tensor_parallel_size'], old['target_tensor_parallel_size']],
            'batch_size': run['batch_size'], 'gamma': old['gamma'],
            'draft_mode': old['draft_mode'],
            'linear_full_window': old['spec_rhythm_linear_full_window'],
            'bonus_token': old['spec_rhythm_linear_bonus_token'],
            'SLO_AWARE_ADMISSION': env.get('VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION'),
            'TARGET_SLO_FIRST': env.get('VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST'),
            'coalesce_min': old['spec_rhythm_prefill_coalesce_min_requests'],
            'coalesce_max_wait_ms': old['spec_rhythm_prefill_coalesce_max_wait_ms'],
            'mixed_target_prefill_requested': env.get('VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL'),
            'enable_mc2': old['enable_mc2'], 'mc2_profile': old['mc2_profile'],
            'respect_eos': old['respect_eos'], 'output_tokens': run['output_tokens'],
            'paper_TPOT_definition': run['slo']['paper']['tpot_definition'],
            'paper_GP_definition': run['slo']['paper']['goodput_definition'],
            'paper_GP': run['slo']['paper']['goodput_tokens_per_e2e_second'],
            'paper_goodput_tokens': run['slo']['paper']['goodput_tokens'],
            'paper_class_attained': {k: v['attained_requests'] for k, v in run['slo']['paper']['by_class'].items()},
            'raw': run['e2e_throughput_tokens_per_second'],
            'outerwall_seconds': run['e2e_elapsed_seconds'],
            'inference_seconds': run['inference_elapsed_seconds'],
            'recorded_e2e_scope': run['e2e_timing_scope'],
            'warmup_excluded': old['warmup_excluded_from_measurement'],
            'all_463_arrivals_equal_original_offsets_plus_constant_with_float_tolerance': True,
            'absolute_arrival_origin_from_first_row': origin,
            'absolute_arrival_roundtrip_max_error_seconds': roundtrip_error,
            'reported_offset_minus_manifest_offset_minmax_seconds': [min(display_errors), max(display_errors)],
            'separate_TTFT_or_per_pause_charge_not_retained': True,
        })

    report = {
        'schema': 'completed173.prefill-charge-and-old-coalesce4.readonly.v1',
        'status': 'passed_bounded_HOST_source_and_arithmetic_review',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'execution': {'Torch_imported': 'torch' in sys.modules,
                      'model_or_NPU_executed': False, 'candidate_sources_changed': False,
                      'current180_raw_read': False},
        'inputs': pins,
        'actual173': {
            'original_manifest_sha256': trace['manifest_sha256'], 'requests': 463,
            'active_window': 32, 'helper_batches': 427,
            'singleton_pair_multi': [counters['spec_rhythm_prefill_singleton_batches'], counters['spec_rhythm_prefill_pair_batches'], counters['spec_rhythm_prefill_multi_batches']],
            'standalone_helper_leader_HOST_wall_seconds': refill_ms / 1000,
            'target_phase_leader_HOST_seconds': trace['phase_seconds']['target'],
            'whole_generate_wall_seconds': trace['elapsed_seconds'],
            'target_phase_is_not_device_kernel_duration': True,
            'no_addition_as_device_or_independent_critical_path_time': True,
            'which_requests_are_charged': 'At each helper call, precisely the pre-existing active set. Every such request receives that full leader-measured helper elapsed_ms once. Newly admitted requests are appended after the charge and start their decode accumulator after their own root prefill.',
            'charged_request_pause_memberships': 12744,
            'old_active_count_per_pause_histogram': dict(sorted(charged_count_histogram.items())),
            'pause_count_per_request_histogram': dict(sorted(request_count_histogram.items())),
            'pause_count_per_request_min_median_max': [min(per_request.values()), statistics.median(per_request.values()), max(per_request.values())],
            'zero_pause_request_indices': zero_pause,
            'membership_reconstruction_sha256': sha(json.dumps(membership_rows, separators=(',', ':'), sort_keys=True).encode()),
            'membership_reconstruction_order_checks': '427 helper sequence_id groups match every first-root callback, no incumbent callback interleaves any group, every owner finishes and active is empty. All 463 output lengths are 256, no immediate-root EOS completion.',
            'per_pause_duration_retained': False,
            'per_request_exact_refill_milliseconds_reconstructible': False,
            'why_not': 'The trace stores one aggregate online_refill_ms and prefill metadata but no per-pause elapsed or explicit charge ledger. Stream callback gaps include other work and cannot identify individual helper duration. Counts times global mean would be an estimate, not exact charging.',
            'TTFT_source_available': 'first_token_wall_seconds160 - same request result.arrival_ts, all 463 nonnegative; this covers new-request pending/admission/own-prefill to first callback. No TTFT SLO threshold is invented.',
            'TTFT_min_median_max_ms': [min(ttft), statistics.median(ttft), max(ttft)],
        },
        'source_clock_boundaries': [
            {'source': str(P148 / 'runtime_tree_decode.py'), 'lines': [203, 287],
             'finding': 'Leader arrival_clock/readiness and cache activation happen before refill_started. Helper duration is sampled before prefill_clock broadcast; only that duration is added to old active. Root delivery and active append happen afterward.'},
            {'source': str(P148 / 'runtime_tree_decode.py'), 'lines': [399, 416, 984, 1002],
             'finding': 'Live admission precedes cycle_started; offline admission follows the cycle elapsed snapshot and finished-owner removal. Helper duration is explicitly charged outside the cycle timer, without double counting that same helper in cycle_ms. Fullwall includes additional untimed admission/accounting overhead.'},
            {'source': str(P148 / 'runtime_tree_decode.py'), 'lines': [1035, 1041],
             'finding': 'Native_paper numerator is finished_decode_elapsed minus decode_start_elapsed. Observed TPOT divides by output_tokens-1; paper TPOT divides by output_tokens. Pending/admission and own-root prefill are excluded from this numerator and belong in TTFT/fullwall.'},
            {'source': str(NATIVE / 'vllm_ascend/spec_decode/pearl/spec_rhythm.py'), 'lines': [106, 114, 171, 172],
             'finding': 'add_decode_time adds nonnegative charge to accumulator; scheduler effective_elapsed also includes arrival debt, a separate planning clock that does not alter reported paper numerator.'},
            {'source': str(P148 / 'runtime_tree_decode.py'), 'lines': [496, 511, 609, 627],
             'finding': 'target phase uses HOST perf_counter around early submit through original target completion/join. It contains submission, CPU preparation overlap and residual waiting; it is not a device profiler measurement.'},
        ],
        'old_coalesce4_completed_results': historical,
        'correction_to_prior_history_note': {
            'old_note_preserved': str(D / 'SLO_AWARE_ADMISSION_COALESCING_HISTORY_AND_CURRENT_FEASIBILITY_HOST.json'),
            'finding': 'An old report offset starting at 0 does not establish a changed client arrival trace. In both old runs all 463 absolute arrivals reproduce original manifest offsets plus one constant to timestamp float precision. The retained benchmark summary subtracts minimum arrival only when displaying arrival_offset_sec (1202-1223); materialize_arrivals uses origin + manifest offset (731-757).',
            'remaining_limit': 'Old result has no separate saved arrival-origin event tied to outer perf_counter or per-run source snapshot. Its declared E2E scope includes arrival origin/enqueue/IPC/prefill/decode; do not claim complete byte-identical historical/current instrumentation.',
        },
        'why_current_contract_is_coalesce1': {
            'handoff_exact_evidence': {'source': '/root/SpecSLO/SPECSLO_FINAL_EXPERIMENT_HANDOFF_zh.md', 'lines': [462, 513],
                                       'meaning': 'The historical handoff explicitly fixes baseline and candidate to coalesce1, excluding coalesce4 batching/TTFT changes as the source of final reproduction gains.'},
            'older_audit_exact_evidence': {'source': str(OLD / 'online_coalesce1_tp3_vs_tp4_audit_20260918_zh.md'), 'lines': [10, 13, 92, 93],
                                         'meaning': 'The September18 audit says this round uses coalesce1 as required; it separately rejects strict SLO-aware admission after lower GP and KV residency failure. It does not identify who imposed the coalesce requirement.'},
            'user_hard_requirement_established': False,
            'user_requirement_limit': 'The available user messages for the ongoing dynamic-B/tree task do not explicitly fix coalesce1. The documents establish a historical reproduction/fair-comparison contract, not proof of a permanent user prohibition. No underlying original user instruction was found in this bounded source/result review. Current ROOT instructions still keep all present candidates at coalesce1/wait0.',
        },
        'bounded_candidate_specific_admission_feasibility': {
            'mechanically_possible_without_changing_client_trace': True,
            'original_semantics': 'The existing admission helper can defer an already-arrived cohort while incumbents continue, then genuinely prefill a larger ready group. It never admits future arrivals; empty service and capacity/tail constraints force release. coalesce1 already naturally made 30 pairs and 3 multis in completed173.',
            'possible_reason_to_test': 'Fewer standalone helper pauses and amortized per-batch overhead could protect incumbents. Each larger pause can also cost more and charge many active requests. Waiting increases the newly queued requests TTFT and changes residency; no speed or GP conclusion follows from counts.',
            'comparison_contract': 'Only a separately named server-policy variant may alter candidate admission. Preserve exact 463 original arrivals, IDs, prompts/SLO/output limits, all outputs, absolute first-token timestamps, original TPOT charging and whole generate/arrival wall. Report per-class pending wait, TTFT distribution, TPOT and overall raw/GP; disclose admission-policy mismatch to any coalesce1 baseline. A matched server-policy control is needed for a batching-specific causal comparison.',
            'Native_execution_limit': 'Current exact-prefill wrapper admits only its supported single-owner graph case; multiple-owner cohorts delegate to the original helper. Thus coalescing can change actual Native shapes/graph path. Any future opt-in needs current-engine numerical/protocol/finite-inventory warm/zero/seal proof for reached calls, not copied flags or old linear560 GP.',
            'no_implementation_or_new_epoch': True,
            'no_GP_extrapolation': True,
        },
    }
    assert report['execution']['Torch_imported'] is False
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'sha256': sha(OUTPUT.read_bytes()),
                      'bytes': OUTPUT.stat().st_size,
                      'helper_batches': 427, 'charged_memberships': 12744,
                      'Torch_imported': False}, sort_keys=True))


if __name__ == '__main__':
    main()
