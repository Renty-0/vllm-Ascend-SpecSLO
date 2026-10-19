"""HOST temporal equivalence of full-copy and prefix/128-writer alias split."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent
B = P.parent
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')


def main():
    import torch
    torch.set_num_threads(1)
    output = P / 'HOST359_RESULT.json'
    assert not output.exists()
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    name = 'vllm_ascend.spec_decode.pearl.tree'
    spec = importlib.util.spec_from_file_location(name, NATIVE / 'tree.py')
    tree = importlib.util.module_from_spec(spec)
    sys.modules[name] = tree
    spec.loader.exec_module(tree)
    for directory in ('tree-depth3-branch6-protocol-host-v357',
                      'tree-alternate-chain-target-host-v269',
                      'tree-canonical-M80-FULL9-full463-own-AR-v231',
                      'tree-Q4S32-branch-projection-host-v358',
                      'tree-single-validated-prefix-descriptors-host-v304'):
        sys.path.append(str(B / directory))
    contracts = [path for directory in B.iterdir() if directory.is_dir()
                 for path in (directory / 'contract262.py',) if path.exists()]
    assert len(contracts) == 1
    sys.path.append(str(contracts[0].parent))
    from plan357 import make, resource_use
    from padding231 import root_only
    from layout358 import layout
    from projection358 import plan as project
    from descriptors359 import CheckedRows, separate
    counts = dict(frames=0, all_original_cache_row_comparisons=0,
                  moved_current_aliases=0, committed_prefix_copies=0, rejected_controls=0)
    saved = None
    cases = [(prefix, n, budget) for prefix in (0, 1, 126, 127, 128, 129, 1505, 2556)
             for n in (1, 4, 12) for budget in (4, 5, 6)]
    for prefix, n, budget in cases:
        budgets = [budget] * n
        resource_use(budgets)
        plans = [make(budget, prefix).plan for _ in budgets]
        ids = list(range(n))
        q = n * (budget + 1)
        remaining = 128 - q
        while remaining:
            width = min(4, remaining)
            ids.append(64 + len(ids) - n)
            plans.append(root_only(4096) if width == 1 else make(width - 1, 0).plan)
            remaining -= width
        tables = [[-1] * 32 for _ in range(160)]
        cursor = 0
        for owner, item in zip(ids, plans, strict=True):
            pages = (max(item.cache_positions.tolist()) + 128) // 128
            tables[owner][:pages] = list(range(cursor, cursor + pages))
            cursor += pages
        scratch = list(range(96, 160))
        for owner in scratch:
            tables[owner][0] = cursor
            cursor += 1
        assert cursor < 544
        desc = layout(plans, ids, tables, scratch)
        projection = project(desc['actual_seq_lengths_q'], q)
        slots = [tables[owner][pos // 128] * 128 + pos % 128
                 for owner, item in zip(ids, plans, strict=True) for pos in item.cache_positions.tolist()]
        assert len(set(slots)) == len(slots) == 128
        order = projection['physical_to_original']
        query_slots = [slots[i] for i in order]
        contract = CheckedRows(desc['source'], desc['dest'], rows=544 * 128)
        split = separate(contract, query_slots)
        base = torch.arange(544 * 128, dtype=torch.int64)
        values = -torch.arange(1, 129, dtype=torch.int64)
        original = base.clone()
        original[torch.tensor(slots)] = values
        original[torch.tensor(desc['dest'])] = original[torch.tensor(desc['source'])].clone()
        candidate = base.clone()
        for source, destination, length in split['spans']:
            candidate[destination:destination + length] = candidate[source:source + length].clone()
        candidate[torch.tensor(query_slots)] = values[torch.tensor(order)]
        for physical, destination in enumerate(split['query_alias_destination']):
            if destination >= 0:
                candidate[destination] = values[order[physical]]
        assert torch.equal(original, candidate)
        assert split['original_copies'] == split['prefix_copies'] + split['current_aliases']
        assert split['span_count'] <= 12
        counts['frames'] += 1
        counts['all_original_cache_row_comparisons'] += base.numel()
        counts['moved_current_aliases'] += split['current_aliases']
        counts['committed_prefix_copies'] += split['prefix_copies']
        saved = contract, query_slots

    def reject(contract, slots):
        try:
            separate(contract, slots)
        except ValueError:
            counts['rejected_controls'] += 1
        else:
            raise AssertionError('invalid canonical writer vector accepted')

    contract, slots = saved
    reject(contract, slots[:-1])
    reject(contract, [True] + slots[1:])
    reject(contract, [544 * 128] + slots[1:])
    reject(contract, [contract.destination[0]] + slots[1:])
    aliases = [source for source, dest in zip(contract.source, contract.destination, strict=True)
               if source != dest and source in slots]
    assert aliases
    duplicated = list(slots)
    duplicated[-1] = aliases[0]
    reject(contract, duplicated)
    assert 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_Q4S32_original_split128_snapshot_equivalence', counts=counts,
                  original_anti_alias_and_twelve_span_checks_preserved=True,
                  original_split_full_AST_inverse=True,
                  full_original_copy_after_writes_equals_prefix_before_writes_plus_aliases=True,
                  simulation_scope='integer address labels for one cache array; not neural or BF16 device validation',
                  NPU_used=False, serving_installed=False, neural_math_qualified=False,
                  full463_qualified=False, Raw=None, Goodput=None,
                  source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (P / 'descriptors359.py', contracts[0],
                                              B / 'tree-single-validated-prefix-descriptors-host-v304/descriptors304.py',
                                              B / 'tree-Q4S32-branch-projection-host-v358/layout358.py',
                                              B / 'tree-Q4S32-branch-projection-host-v358/projection358.py')})
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
