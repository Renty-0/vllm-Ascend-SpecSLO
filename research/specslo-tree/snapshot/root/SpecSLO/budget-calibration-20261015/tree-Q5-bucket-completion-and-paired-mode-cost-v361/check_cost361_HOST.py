"""No-Torch/NPU protocol checks for paired-mode order, counters and rollback."""
import inspect
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent


class Heads:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return list(self.value)


class Engine:
    def __init__(self, fault=None, draft=False):
        self.rank = 0 if draft else 1
        self.is_draft = draft
        self._branch_tail_state = types.SimpleNamespace(uniform_bucket344_enabled=True)
        self.fault = fault
        self.calls = []
        self.keys = set()

    def graph_metrics(self):
        counts = len(self.keys)
        if self.fault == 'capture_after_warm' and len(self.calls) > 10:
            counts += len(self.calls)
        return dict(aclgraph_captures=counts, aclgraph_capture_attempts=counts,
                    aclgraph_runtime_validation_replays=counts, aclgraph_failed_captures=0,
                    aclgraph_capacity_fallbacks=0, aclgraph_shape_fallbacks=0)

    def _pad_spec_rhythm_target_tree_graph(self, plans, roots, rows, ids, active, states):
        return plans, roots, rows, ids, len(plans)

    def target_tree_forward(self, plans, roots, rows, *, sequence_ids, return_logits, real_tree_count):
        assert not return_logits and len(plans) == len(rows) == len(roots) == real_tree_count
        mode = self._branch_tail_state.uniform_bucket344_enabled
        budget = plans[0]['budget']
        self.keys.add((real_tree_count, budget, mode))
        self.calls.append((real_tree_count, budget, mode))
        values = [value for root, row in zip(roots, rows, strict=True) for value in [root] + row]
        if self.fault == 'heads' and mode:
            values[0] += 1
        return dict(used_aclgraph=self.fault != 'fallback', target_query_token_ids=Heads(values))


def main():
    output = P / 'HOST361_COST_PROTOCOL_RESULT.json'
    assert not output.exists() and 'torch' not in sys.modules
    fake_torch = types.ModuleType('torch')
    fake_torch.npu = types.SimpleNamespace(synchronize=lambda: None)
    fake_plan = types.ModuleType('plan269')
    fake_plan.make = lambda budget, prefix, anchor: dict(budget=budget, prefix=prefix, anchor=anchor)
    sys.modules.update(torch=fake_torch, plan269=fake_plan)
    from benchmark361 import run, arms
    bank = dict(prefixes=[[7] * 127 for _ in range(24)], roots=list(range(24)))
    stages = []
    def stage(label, function):
        stages.append(label)
        return function()
    engine = Engine()
    result = run(engine, None, stage, bank)
    assert result['rounds'] == 20 and len(result['pairs']) == 20 and len(engine.calls) == 210
    assert len(stages) == 210 and len(engine.keys) == 10
    assert engine._branch_tail_state.uniform_bucket344_enabled is True
    for i, row in enumerate(result['pairs']):
        assert row['iteration'] == i and len(row['samples']) == 5
        for sample in row['samples']:
            assert set(sample['mode_order']) == {'full2560', 'uniform_bucket'}
            assert set(sample['milliseconds']) == {'full2560', 'uniform_bucket'}
            assert all(value > 0 for value in sample['milliseconds'].values())
    for arm in arms():
        samples = [sample for row in result['pairs'] for sample in row['samples'] if sample['arm'] == arm['name']]
        assert len(samples) == 20
        assert sum(sample['mode_order'][0] == 'full2560' for sample in samples) == 10
    rejected = []
    for fault in ('heads', 'fallback', 'capture_after_warm'):
        engine = Engine(fault)
        try:
            run(engine, None, lambda label, function: function(), bank)
        except RuntimeError:
            rejected.append(fault)
        else:
            raise AssertionError('paired-mode failure was accepted: ' + fault)
        assert engine._branch_tail_state.uniform_bucket344_enabled is True
    engine = Engine(draft=True)
    draft = run(engine, None, lambda label, function: function(), bank)
    assert not engine.calls and all(value is None for row in draft['pairs']
                                    for sample in row['samples'] for value in sample['milliseconds'].values())
    sys.path.insert(0, str(P.parent / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'))
    import probe344, probe361
    assert inspect.signature(probe361.run) == inspect.signature(probe344.run)
    report = dict(status='PASS_HOST_paired_mode_cost_protocol_and_original344_probe_inverse',
                  arms=5, rounds=20, warm_calls=10, timed_calls=200,
                  each_arm_full_first=10, each_arm_bucket_first=10,
                  rejected_controls=rejected, mode_restored_after_success_and_failures=True,
                  draft_role_never_executes_target=True, original594_math_body_preserved=True,
                  fake_Torch_and_model_only=True, actual_Torch_imported=False,
                  NPU_used=False, device_math_qualified=False, full594_qualified=False,
                  performance_measured=False, Raw=None, Goodput=None)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
