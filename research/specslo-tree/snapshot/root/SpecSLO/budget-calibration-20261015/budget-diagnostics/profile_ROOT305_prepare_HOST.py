"""Profile the actual prepare callback on the CPU Native protocol fixture.

No model, NPU, or serving timing is measured. This locates Python/CPU work
remaining after305, without altering the active frozen299 experiment.
"""
import cProfile
import datetime
import hashlib
import json
import pstats
import sys
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'


def main():
    sys.path[:0] = [str(B / name) for name in (
        'tree-concrete-device-span-dual-cache-target-cost-v296',
        'tree-deferred-FULL-mask-host-projection-v259',
        'tree-single-validated-prefix-descriptors-host-v304',
        'tree-single-validated-prefix-prepare-host-v305',
    )]
    import check296
    from fixture259 import build
    import branch_tail255
    from static_state305 import bind_checked_prepare
    from padding269 import pad
    from plan269 import make

    env = build()
    engine, native = env['engine'], env['native']
    bind_checked_prepare(engine._branch_tail_state, branch_tail255)
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = pad
    records = []
    for n, forks in ((12, 0), (12, 1), (12, 12), (20, 4), (24, 0)):
        budgets = [8] * forks + [4] * (n - forks)
        plans = [make(budget, 127, 0) for budget in budgets]
        roots = list(range(41, 41 + n))
        rows = [[103 + 17 * i + 29 * j for j in range(budget)]
                for i, budget in enumerate(budgets)]

        def execute():
            pp, rr, dd, ss, real = pad(engine, plans, roots, rows, list(range(n)), [], [])
            result = native.NativePearlEngine.target_tree_forward(
                engine, pp, rr, dd, sequence_ids=ss,
                return_logits=False, real_tree_count=real)
            assert result['num_draft_tokens'] == budgets and result['query_count'] == 120

        execute()
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(30):
            execute()
        profiler.disable()
        stats = pstats.Stats(profiler)
        functions = []
        for (filename, line, name), (primitive, calls, total, cumulative, _) in stats.stats.items():
            functions.append(dict(filename=filename, line=line, function=name,
                                  primitive_calls=primitive, calls=calls,
                                  total_seconds=total, cumulative_seconds=cumulative))
        records.append(dict(N=n, branch_owners=forks, rounds=30,
                            total_profile_seconds=stats.total_tt,
                            functions=sorted(functions,
                                             key=lambda row: row['cumulative_seconds'],
                                             reverse=True)[:35]))
    assert 'torch_npu' not in sys.modules
    result = dict(
        schema='ROOT305.HOST-prepare-profile.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='CPU Native protocol fixture and arithmetic model double only',
        records=records, NPU_used=False, model_executed=False,
        serving_latency_measured=False, Goodput=None)
    output = D / 'ROOT305_CPU_PREPARE_PROFILE.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output),
                          cases=[dict(N=row['N'], forks=row['branch_owners'],
                                      profiled_CPU_ms_per_call=row['total_profile_seconds'] * 1000 / row['rounds'])
                                 for row in records], NPU_used=False)))


if __name__ == '__main__':
    main()
