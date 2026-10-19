"""Synthetic timing/call evidence accepted only with exact mode geometry."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent
B = P.parent


def main():
    output = P / 'HOST361_MODE_AUDIT_RESULT.json'
    assert not output.exists() and 'torch' not in sys.modules
    fake_torch = types.ModuleType('torch')
    fake_torch.npu = types.SimpleNamespace(synchronize=lambda: None)
    fake_plan = types.ModuleType('plan269')
    fake_plan.make = lambda budget, prefix, anchor: dict(budget=budget, prefix=prefix, anchor=anchor)
    spec = importlib.util.spec_from_file_location('_ROOT361_original_resource_domain',
                                                 B / 'tree-alternate-chain-target-host-v269/plan269.py')
    original_plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_plan)
    fake_plan.envelope = original_plan.envelope
    fake_plan.plan_family = original_plan.plan_family
    sys.modules.update(torch=fake_torch, plan269=fake_plan)
    sys.path.insert(0, str(B / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'))
    from check_cost361_HOST import Engine
    from benchmark361 import run
    from audit361 import mode_panel
    prefixes = [[7] * 127 for _ in range(24)]
    engine = Engine()
    panel = run(engine, None, lambda label, function: function(), dict(prefixes=prefixes, roots=list(range(24))))
    calls = []
    for n, budget, enabled in engine.calls:
        branch = budget > 4
        copied = n * 132 if branch else 0
        grid = next(v for v in (128, 256, 512, 1024, 1792) if v >= copied) if copied else 0
        extent = 512 if enabled else 2560
        calls.append(dict(N=n, real_queries=n * (budget + 1), M=120, attention_segments=24,
            actual_seq_lengths_q=list(range(5, 121, 5)), sequence_lens=[extent] * 24,
            mask_shape=[24, 1, 11 if branch else 10, 4096], copy_enabled=branch,
            copy_grid=grid, primary_side_groups=n * (2 if branch else 1),
            used_aclgraph=True, execution_mode='replay', fallback_reason=None,
            actual_capture_key='key' + (f'|ROOT296:copy-grid:{grid}' if branch else ''),
            uniform_KV_bucket344=dict(enabled=enabled, extent=extent, original_max_context=132)))
    row = dict(name='HOST_synthetic_only', paired_KV_modes361=panel, actual_KV_mode_calls361=calls,
               all72_prefixes_after_KV_modes361=True)
    values = mode_panel(row, prefixes, 1)
    assert len(values) == 5 and all(value['bank'] == 'HOST_synthetic_only' for value in values)
    rejected = []
    for fault in ('wrong_full_extent', 'wrong_mode', 'wrong_copy_key', 'missing_call', 'invalid_elapsed', 'unprotected_prefix'):
        bad = copy.deepcopy(row)
        if fault == 'wrong_full_extent':
            bad['actual_KV_mode_calls361'][0]['sequence_lens'] = [512] * 24
        elif fault == 'wrong_mode':
            bad['actual_KV_mode_calls361'][0]['uniform_KV_bucket344']['enabled'] = True
        elif fault == 'wrong_copy_key':
            next(call for call in bad['actual_KV_mode_calls361'] if call['copy_enabled'])['actual_capture_key'] = 'key'
        elif fault == 'missing_call':
            bad['actual_KV_mode_calls361'].pop()
        elif fault == 'invalid_elapsed':
            bad['paired_KV_modes361']['pairs'][0]['samples'][0]['milliseconds']['full2560'] = float('nan')
        else:
            bad['all72_prefixes_after_KV_modes361'] = False
        try:
            mode_panel(bad, prefixes, 1)
        except RuntimeError:
            rejected.append(fault)
        else:
            raise AssertionError('invalid retained mode evidence accepted: ' + fault)
    result = dict(status='PASS_HOST_mode_auditor_positive_and_six_corruption_controls',
                  synthetic_calls=210, synthetic_summaries=5, rejected_controls=rejected,
                  real_model_or_timing_evidence=False, fake_Torch_only=True,
                  NPU_used=False, full594_qualified=False, Goodput=None)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
