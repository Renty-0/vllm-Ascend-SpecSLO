"""CPU layout, exact copy relocation and old single-side equivalence checks."""
import hashlib
from collections import Counter
import json
from pathlib import Path
import random
import sys

P = Path(__file__).resolve().parent
B = P.parent


def fixture(budgets, prefix):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan, pack_selected_tree_plan
    plans = [pack_selected_tree_plan(cached_cpu_tree_speculation_plan(2, 4, prefix, 4096,
              candidate_budget=8), range(b)) for b in budgets]
    owners = list(range(len(plans)))
    remaining = 120 - sum(b + 1 for b in budgets)
    scratch = list(range(463, 527))
    for owner in scratch:
        if not remaining:
            break
        q = min(5, remaining)
        if q == 1:
            from padding231 import root_only
            dummy = root_only(4096)
        else:
            dummy = cached_cpu_tree_speculation_plan(1, q - 1, 0, 4096, candidate_budget=q - 1)
        plans.append(dummy)
        owners.append(owner)
        remaining -= q
    tables = [[-1] * 32 for _ in range(527)]
    cursor = 0
    for owner, budget in enumerate(budgets):
        pages = (prefix + budget + 1 + 127) // 128
        tables[owner][:pages] = list(range(cursor, cursor + pages))
        cursor += pages
    for owner in scratch:
        tables[owner][0] = cursor
        cursor += 1
    assert cursor <= 544 and remaining == 0
    return plans, owners, tables, scratch


def signatures(value):
    import torch
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), hashlib.sha256(value.numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {k: signatures(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [signatures(v) for v in value]
    return value


def check_relocation(inputs, output, real_count):
    import torch
    import projection255
    plans, owners, tables, _ = inputs
    assert output['enabled'] is True
    assert len(output['source']) == len(output['dest']) == 1792
    cache = torch.arange(544 * 128, dtype=torch.int64, device='cpu')
    copied = cache.clone()
    copied[torch.tensor(output['dest'])] = cache[torch.tensor(output['source'])]
    row_offsets = []
    offset = 0
    for plan in plans:
        row_offsets.append(offset)
        offset += plan.candidate_budget + 1
    for group in output['groups']:
        owner = group['owner']
        if owner >= real_count:
            continue
        plan = plans[owner]
        for original_row in group['original_rows']:
            local = original_row - row_offsets[owner]
            visible = (~plan.attention_mask[local]).nonzero().flatten().tolist()
            expected = [tables[owner][p // 128] * 128 + p % 128 for p in visible]
            allowed = (~group['masks'][group['original_rows'].index(original_row)]).nonzero().flatten().tolist()
            actual = [int(copied[group['tables'][p // 128] * 128 + p % 128]) for p in allowed]
            assert actual == expected
        assert all(tables[owner][p // 128] * 128 + p % 128 not in output['dest']
                   for p in range(plan.prefix_len))
    real_q = sum(p.candidate_budget + 1 for p in plans[:real_count])
    physical = projection255.plan(output['actual_seq_lengths_q'], real_q)
    assert physical['supported_primary_side_groups'] <= 24
    assert sorted(physical['physical_to_original']) == list(range(120))


def main():
    import torch
    import torch_npu
    def blocked(*args, **kwargs):
        raise RuntimeError('HOST403 may not initialize an NPU')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    torch.set_num_threads(1)
    import branch_tail255
    import layout403
    rng = random.Random(0)
    prefixes = [0, 1, 127, 128, 129, 511, 512, 1505, 2555]
    legacy = multi = 0
    budgets_seen = Counter()
    original_layout = branch_tail255.layout
    for trial in range(180):
        n = rng.randrange(1, 13)
        budgets = [rng.choice([1, 2, 3, 4, 5]) for _ in range(n)]
        budgets[0] = 5
        args = fixture(budgets, prefixes[trial % len(prefixes)])
        assert signatures(branch_tail255.layout(*args)) == signatures(layout403.layout(*args))
        legacy += 1
    for trial in range(270):
        n = rng.randrange(1, 13)
        budgets = [4] * n
        order = list(range(n))
        rng.shuffle(order)
        leaves_left = min(13, 24 - n)
        for owner in order:
            count = rng.randrange(min(4, leaves_left) + 1)
            budgets[owner] += count
            leaves_left -= count
        if max(budgets) <= 4:
            budgets[0] = 5
        args = fixture(budgets, prefixes[trial % len(prefixes)])
        result = layout403.layout(*args)
        check_relocation(args, result, n)
        budgets_seen.update(budgets)
        multi += 1
    for prefix in prefixes:
        for budgets in ([8], [8, 8], [8, 8, 8], [8, 8, 7, 6], [5] * 12):
            args = fixture(budgets, prefix)
            check_relocation(args, layout403.layout(*args), len(budgets))
            budgets_seen.update(budgets)
            multi += 1
    for prefix in [0, 1, 128, 129]:
        args = fixture([8] * 4, prefix)
        check_relocation(args, layout403.layout(*args), 4)
        budgets_seen.update([8] * 4)
        multi += 1
    # Group capacity alone allows N4/B8, but1792 copy slots cannot hold the
    # worst-case sixteen long tails. Keep that independent bound unchanged.
    try:
        layout403.layout(*fixture([8] * 4, 127))
    except ValueError as error:
        assert 'fixed branch-copy slot bound' in str(error)
    else:
        raise AssertionError('copy1792 capacity overrun accepted')
    try:
        layout403.layout(*fixture([8] * 5, 0))
    except ValueError as error:
        assert 'primary/side resources' in str(error)
    else:
        raise AssertionError('real group25 accepted into physicalS24')
    assert branch_tail255.layout is original_layout and all(budgets_seen[b] > 0 for b in [5, 6, 7, 8])
    output = dict(status='PASS_HOST403_M120_MULTISIDE_RELOCATION_AND_LEGACY_EQUIVALENCE',
                  exact_legacy_B5_cases=legacy, new_multi_side_relocation_cases=multi,
                  actual_CPU_tree_plans_and_masks_used=True,
                  copy_capacity_overrun_rejected=True,
                  real_group25_rejected=True, observed_real_budget_counts=dict(budgets_seen),
                  original_function_unchanged=True,
                  source=layout403.descriptor(), NPU_used=False, models_constructed=0,
                  production_runtime_installed=False, model_qualification=False,
                  required_followup=['actual typed caps/whole-ticket group and copy resource control',
                                     'maxGamma8 and physical KV reserve consistency',
                                     'reject unqualified eager parents6..8',
                                     'actual optimized draft compatibility and full-worker cost before claiming free gain',
                                     'fresh independent model and all128KV comparison',
                                     'complete global64 workload ownAR and real performance'],
                  Raw=None, Goodput=None)
    (P / 'HOST403_RESULT.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['status', 'exact_legacy_B5_cases', 'new_multi_side_relocation_cases']}))


if __name__ == '__main__':
    main()
