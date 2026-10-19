"""Test rejection of corrupt diagnostic records without executing a launcher."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from geometry336 import LAYOUTS, cases, cost_arms
import guard336

P = Path(__file__).resolve().parent


def reports():
    values = []
    arms = cost_arms()
    names = [a['name'] for a in arms]
    for rank in range(4):
        records, calls = [], []
        for case in cases():
            comparisons = []
            for future in (0, 17):
                comparisons.append(dict(future=future, exact_greedy=True,
                    hidden_bitwise_equal=True, all128_KV_bitwise_equal=True,
                    graph_eager_KV_bitwise_equal=True, candidate_KV_copied=False,
                    candidate_tokens=list(range(case['query_count'])),
                    reference_tokens=list(range(case['query_count'])),
                    frames=[dict(depth=d, real_nodes=case['N'], physical_queries=case['M'],
                        future=future, candidate_KV_copied=False, reference_eager_only=True)
                        for d in range(case['width'])]))
            records.append(dict(case=case, comparisons=comparisons) if rank else None)
            calls.append(dict(layout=case['layout'], M=case['M'], groups=case['groups'],
                width=case['width'], mask_shape=[case['groups'], 1, 10, 4096],
                actual_seq_lengths_q=list(range(case['width'], case['M']+1, case['width'])),
                sequence_lens=[2560]*case['groups'], used_aclgraph=True, fallback_reason=None))
        pairs = [dict(order=names[i % 4:]+names[:i % 4],
            milliseconds={name:float(j+1) if rank else None for j, name in enumerate(names)}) for i in range(20)]
        metrics = {name:0 for name in ('aclgraph_captures', 'aclgraph_capture_attempts',
            'aclgraph_runtime_validation_replays', 'aclgraph_failed_captures',
            'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks')}
        before = dict(mc2_dispatch_fallback=0, mc2_dispatch_exception=0,
            mc2_dispatch_native_epilogue_chained_success=1, mc2_dispatch_native_epilogue_chain_flush=1)
        after = dict(before, mc2_dispatch_native_epilogue_chained_success=2, mc2_dispatch_native_epilogue_chain_flush=2)
        values.append(dict(rank=rank, case_count=36, cases=records,
            performance=False, adopted=False, real_draft_generation=False,
            whole_workload_AR_qualified=False, model_dtype='torch.bfloat16',
            bank=dict(independent_neural_prefills=96, all_triplet_roots_equal=True,
                all_triplet_prefix_KV_equal=True) if rank else None,
            all_prefixes_exact_after_cost=True if rank else None,
            activation=dict(original_model_forward_unchanged=True, Native_flags_assigned=False,
                deferred_read_done=False), MC2_routes=dict(routes=[{}]*128),
            original_engine_metrics_before=before, original_engine_metrics_after=after,
            cold_native_calls=calls if rank else [],
            paired_target_cost=dict(rounds=20, no_new_graph_work=True, arms=list(arms),
                graph_counters_before=metrics, graph_counters_after=metrics, pairs=pairs)))
    return values


def main():
    base = reports()
    completed = dict(replies=[['calibrated', r, dict(numeric_cases=36, performance=False)] for r in range(4)])
    rejected = 0
    with tempfile.TemporaryDirectory(prefix='specslo336-HOST-') as directory:
        folder = Path(directory)
        (folder/'completed.json').write_text(json.dumps(completed))
        (folder/'worker-exit.json').write_text(json.dumps(dict(closed=True, all_exited=True)))
        for rank, row in enumerate(base):
            (folder/f'rank{rank}-numeric.json').write_text(json.dumps(row))
        actual, _, _ = guard336.actual_audit(folder)
        assert actual['cases_each_target'] == 36 and len(actual['costs']) == 3
        for kind in ('greedy', 'hidden', 'prefix', 'frames', 'M', 'MC2_fallback', 'cost_order', 'exit'):
            broken = copy.deepcopy(base[1])
            exit_record = dict(closed=True, all_exited=kind != 'exit')
            if kind == 'greedy':
                broken['cases'][0]['comparisons'][0]['reference_tokens'][0] = -1
            elif kind == 'hidden':
                broken['cases'][0]['comparisons'][0]['hidden_bitwise_equal'] = False
            elif kind == 'prefix':
                broken['all_prefixes_exact_after_cost'] = False
            elif kind == 'frames':
                broken['cases'][0]['comparisons'][0]['frames'].pop()
            elif kind == 'M':
                broken['cold_native_calls'][0]['M'] += 1
            elif kind == 'MC2_fallback':
                broken['original_engine_metrics_after']['mc2_dispatch_fallback'] = 1
            elif kind == 'cost_order':
                broken['paired_target_cost']['pairs'][0]['order'].reverse()
            (folder/'rank1-numeric.json').write_text(json.dumps(broken))
            (folder/'worker-exit.json').write_text(json.dumps(exit_record))
            try:
                guard336.actual_audit(folder)
            except RuntimeError:
                rejected += 1
            else:
                raise AssertionError('corrupt record accepted: '+kind)
    namespace = guard336.execution_namespace()
    assert namespace['SCOPE']['models_loaded'] is True and namespace['SCOPE']['full463'] is False
    assert namespace['MAX_SECONDS'] == 1800
    assert 'ROOT.V336.linear-target-shape-supervisor.v1' in namespace['execute_locked'].__code__.co_consts
    assert not any(n.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') for n in sys.modules)
    result = dict(status='passed_HOST_auditor', corrupt_records_rejected=rejected,
        fixture_is_not_model_evidence=True, original_owned_guard_compiled=True,
        Torch_imported=False, NPU_used=False, launcher_executed=False)
    with (P/'HOST336_GUARD_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
