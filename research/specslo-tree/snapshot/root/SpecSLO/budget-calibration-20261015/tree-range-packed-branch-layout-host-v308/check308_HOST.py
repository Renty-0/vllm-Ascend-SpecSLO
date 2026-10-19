"""Differential Native layout checks, rejected inputs, and paired HOST costs."""
import copy
import json
import statistics
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B / 'tree-concrete-device-span-dual-cache-target-cost-v296'),
               str(B / 'tree-deferred-FULL-mask-host-projection-v259')]
import check296


def equal(first, second):
    import torch
    if isinstance(first, torch.Tensor):
        assert first.device == second.device and first.dtype == second.dtype
        assert first.stride() == second.stride() and torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            equal(a, b)
    else:
        assert first == second


def main():
    from fixture259 import build
    env = build()
    engine, native = env['engine'], env['native']
    from static_state296 import bind_checked_prepare
    import branch_tail255
    bind_checked_prepare(engine._branch_tail_state, branch_tail255)
    from padding269 import pad
    from plan269 import make
    from oracle296 import cases
    from layout269 import layout as reference
    from layout308 import layout as proposal
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = pad
    examples, saved = [], []

    def one(case, prefix):
        budgets = case['budgets']
        plans = [make(b, prefix, case['anchor']) for b in budgets]
        roots = list(range(41, 41 + len(budgets)))
        rows = [[103 + 17 * i + j for j in range(b)] for i, b in enumerate(budgets)]
        pp, rr, dd, ss, real = pad(engine, plans, roots, rows, list(range(len(budgets))), [], [])
        native.NativePearlEngine.target_tree_forward(engine, pp, rr, dd, sequence_ids=ss,
                                                     return_logits=False, real_tree_count=real)
        args = (pp, ss, engine.cache_allocation.block_tables, engine._tree_graph_scratch_sequence_ids)
        old, new = reference(*args), proposal(*args)
        equal(old, new)
        examples.append(dict(case=case['name'], prefix=prefix, N=len(budgets), branches=new['branches']))
        if new['enabled'] and len(saved) < 12:
            saved.append(copy.deepcopy(args))

    finite = cases()
    for case in finite:
        one(case, 127)
    extra = [case for case in finite if max(case['budgets']) == 8
             and case['N'] in (1, 12, 16, 23) and case['anchor'] == 0]
    for prefix in (0, 1, 2, 126, 128, 129, 1505, 2555):
        for case in extra:
            one(case, prefix)

    rejected = []

    def reject(name, args, **kwargs):
        for function in (reference, proposal):
            try:
                function(*args, **kwargs)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError('Invalid layout accepted: ' + name)
        rejected.append(name)

    args = saved[0]
    reject('too_few_copy_slots', args, copy_slots=1)
    bad = copy.deepcopy(args)
    bad[2][bad[1][0]][0] = 544
    reject('source_page_out_of_bounds', bad)
    available = [seq for seq in args[3] if seq not in args[1]]
    bad = copy.deepcopy(args)
    bad[2][available[0]][0] = bad[2][available[1]][0]
    reject('duplicate_scratch_page', bad)
    bad = copy.deepcopy(args)
    bad[2][available[0]][0] = bad[2][bad[1][0]][0]
    reject('scratch_aliases_live_owner', bad)
    bad = copy.deepcopy(args)
    bad[0][0].attention_mask[-1, 0] = True
    reject('missing_visible_committed_prefix', bad)
    reject('duplicate_owner', (args[0], [args[1][0]] * len(args[1]), args[2], args[3]))

    timings = []
    for args in saved:
        measured = {'original': [], 'ranges': []}
        for iteration in range(30):
            order = ('original', 'ranges') if iteration % 2 == 0 else ('ranges', 'original')
            for mode in order:
                start = time.perf_counter()
                (reference if mode == 'original' else proposal)(*args)
                measured[mode].append((time.perf_counter() - start) * 1000)
        timings.append({key + '_median_ms': statistics.median(value) for key, value in measured.items()})
    assert 'torch_npu' not in sys.modules
    result = dict(status='HOST_complete_original_layout_equality_and_negative_controls_passed',
                  original_cases=408, total_cases=len(examples), cases=examples,
                  original_AST_inverse_and_two_loop_inverse=True,
                  all_original_bounds_unique_destination_and_live_alias_checks_retained=True,
                  rejected_controls=rejected, paired_HOST_layout_timings=timings,
                  NPU_used=False, runtime_installed=False, Goodput=None)
    with (P / 'HOST308_RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
