"""Fresh paired FULL2560 versus uniform-bucket target costs, after node math.

No real draft generation, serving workload, gain curve or Goodput is measured.
The original344 budget-cost profile is retained separately by probe361.
"""
import time


def arms():
    return tuple(dict(name=f'N{n}-B{budget}', N=n, budget=budget,
                      budgets=[budget] * n)
                 for n, budget in ((1, 4), (1, 8), (12, 4), (12, 8), (24, 4)))


def run(engine, native, stage, bank, *, rounds=20):
    import torch
    from plan269 import make
    target = not engine.is_draft
    state = engine._branch_tail_state
    old_mode = state.uniform_bucket344_enabled if target else None
    modes = ('full2560', 'uniform_bucket')
    expected = {}
    cases = arms()
    inputs = {}
    for arm in cases:
        n = arm['N']
        budgets = arm['budgets']
        plans = [make(budget, len(bank['prefixes'][i]), 0) for i, budget in enumerate(budgets)]
        rows = [[103 + 17 * i + 29 * j for j in range(budget)] for i, budget in enumerate(budgets)]
        inputs[arm['name']] = plans, rows, bank['roots'][:n] if target else [0] * n

    def call(arm, mode):
        plans, rows, roots = inputs[arm['name']]
        state.uniform_bucket344_enabled = mode == 'uniform_bucket'
        padded = engine._pad_spec_rhythm_target_tree_graph(plans, roots, rows, list(range(arm['N'])), [], [])
        pp, rr, dd, ss, real = padded
        result = engine.target_tree_forward(pp, rr, dd, sequence_ids=ss, return_logits=False, real_tree_count=real)
        if result['used_aclgraph'] is not True:
            raise RuntimeError('ROOT361 paired KV modes require actual original ACLGraph')
        return result

    try:
        for arm in cases:
            heads = {}
            for mode in modes:
                def warm():
                    if not target:
                        return None
                    result = call(arm, mode)
                    torch.npu.synchronize()
                    return result['target_query_token_ids'].cpu().tolist()
                heads[mode] = stage('KV-modes361/' + arm['name'] + '/' + mode + '/warm', warm)
            if heads[modes[0]] != heads[modes[1]]:
                raise RuntimeError('ROOT361 FULL2560 and bucket greedy outputs differ')
            expected[arm['name']] = heads[modes[0]]
        before = engine.graph_metrics()
        rows = []
        for iteration in range(rounds):
            ordered = cases[iteration % len(cases):] + cases[:iteration % len(cases)]
            samples = []
            for index, arm in enumerate(ordered):
                order = modes if (iteration + index) % 2 == 0 else modes[::-1]
                values = {}
                for mode in order:
                    def measure():
                        if not target:
                            return None
                        torch.npu.synchronize()
                        started = time.perf_counter()
                        result = call(arm, mode)
                        torch.npu.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        if result['target_query_token_ids'].cpu().tolist() != expected[arm['name']]:
                            raise RuntimeError('ROOT361 repeated paired-mode output changed')
                        return elapsed
                    values[mode] = stage(f'KV-modes361/pair{iteration}/' + arm['name'] + '/' + mode, measure)
                samples.append(dict(arm=arm['name'], mode_order=list(order), milliseconds=values))
            rows.append(dict(iteration=iteration, arm_order=[a['name'] for a in ordered], samples=samples))
        after = engine.graph_metrics()
        keys = ('aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays',
                'aclgraph_failed_captures', 'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks')
        if any(before[key] != after[key] for key in keys):
            raise RuntimeError('ROOT361 paired-mode timed region captured, validated or fell back')
        return dict(rank=engine.rank, target_math_on_rank=target, arms=list(cases), modes=list(modes),
                    rounds=rounds, pairs=rows, graph_counters_before=before, graph_counters_after=after,
                    no_new_capture_validation_or_fallback=True, outputs_equal_between_modes=True,
                    scope='same-instance synchronous full target pad/metadata/model/sync; D2H head check and consensus excluded',
                    original_FIA_and_reference_backend_preserved=True, real_draft_generation=False,
                    complete_serving=False, Raw=None, Goodput=None)
    finally:
        if target:
            state.uniform_bucket344_enabled = old_mode
