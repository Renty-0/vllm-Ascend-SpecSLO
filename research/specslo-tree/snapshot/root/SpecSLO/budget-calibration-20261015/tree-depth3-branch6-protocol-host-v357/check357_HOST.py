"""Use original plan packing and Native CPU verifier for every legal short path."""
import ast
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import types


P = Path(__file__).resolve().parent
B = P.parent
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')


def main():
    output = P / 'HOST357_RESULT.json'
    assert not output.exists()
    import torch
    torch.set_num_threads(1)
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    name = 'vllm_ascend.spec_decode.pearl.tree'
    spec = importlib.util.spec_from_file_location(name, NATIVE / 'tree.py')
    tree = importlib.util.module_from_spec(spec)
    sys.modules[name] = tree
    spec.loader.exec_module(tree)
    sys.path.append(str(B / 'tree-alternate-chain-target-host-v269'))
    from plan269 import make as old_make, plan_family as old_family
    from plan357 import make, PARENTS, RAW_ORDER, resource_use, spare_side_owners
    function = copy.deepcopy(next(node for node in ast.walk(ast.parse((NATIVE / 'native_engine.py').read_text()))
                                 if isinstance(node, ast.FunctionDef) and node.name == '_verify_tree_outputs_host'))
    function.decorator_list = []
    namespace = dict(torch=torch, TreeVerificationOutput=collections.namedtuple('OriginalVerdict', 'outputs accepted'))
    exec(compile('from __future__ import annotations\n' + ast.unparse(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))),
        str(NATIVE / 'native_engine.py') + '::original-CPU-verifier', 'exec'), namespace)
    verify = namespace['_verify_tree_outputs_host']
    mask_cases = verifier_cases = 0
    for prefix in (0, 1, 127, 128, 1505, 2556):
        for budget in range(1, 7):
            selection = make(budget, prefix)
            plan = selection.plan
            parents = list(PARENTS[:budget])
            assert plan.candidate_budget == budget and selection.raw_pool_indices == RAW_ORDER[:budget]
            assert plan.parent_indices.tolist() == parents
            assert plan.cache_positions.tolist() == list(range(prefix, prefix + budget + 1))
            expected_positions = [prefix] + [prefix + 1 + (i if i < 3 else i - 3) for i in range(budget)]
            assert plan.positions.tolist() == expected_positions and max(plan.positions) <= prefix + 3
            for query in range(budget + 1):
                visible = set(range(prefix + 1))
                node = query - 1
                while node >= 0:
                    visible.add(prefix + node + 1)
                    node = parents[node]
                actual = set(torch.nonzero(~plan.attention_mask[query], as_tuple=False).flatten().tolist())
                assert actual == visible
            if budget <= 3:
                previous = old_make(budget, prefix)
                assert torch.equal(plan.parent_indices, previous.parent_indices)
                assert torch.equal(plan.positions, previous.positions)
                assert torch.equal(plan.attention_mask, previous.attention_mask)
                assert plan.width == previous.width and plan.depth == previous.depth
            mask_cases += 1
            tokens = [11, 22, 33, 55, 66, 77][:budget]
            paths = [list(range(depth)) for depth in range(min(budget, 3) + 1)]
            paths += [list(range(3, 3 + depth)) for depth in range(1, max(0, budget - 3) + 1)]
            for path in paths:
                predictions = [99] * (budget + 1)
                if path and path[0] == 0:
                    for depth in range(len(path)):
                        predictions[depth] = tokens[depth]
                elif path:
                    predictions[0] = tokens[3]
                    for depth in range(len(path) - 1):
                        predictions[4 + depth] = tokens[4 + depth]
                result = verify([tokens], [plan], torch.tensor(predictions), 4, torch.device('cpu'))
                accepted = [v for v in result.accepted[0].tolist() if v >= 0]
                emitted = [v for v in result.outputs[0].tolist() if v >= 0]
                assert accepted == path and emitted == [tokens[i] for i in path] + [99]
                assert len(emitted) <= 4
                verifier_cases += 1
    try:
        old_family([make(4, 127).plan])
    except ValueError:
        pass
    else:
        raise AssertionError('new branch topology silently reused old P4 classifier')
    resource_cases = 0
    for n in range(1, 33):
        for side_count in range(spare_side_owners(n) + 1):
            budgets = [6] * side_count + [3] * (n - side_count)
            result = resource_use(budgets)
            assert result['logical_queries'] <= 128 and result['primary_and_side_groups'] <= 32
            assert result['selected'] <= 96
            resource_cases += 1
    for bad in (0, 7, True):
        try:
            make(bad, 127)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid B accepted')
    for budgets in ([6] * 13, [3] * 33, [6] * 12 + [3] * 9):
        try:
            resource_use(budgets)
        except ValueError:
            pass
        else:
            raise AssertionError('oversubscribed groups or candidates accepted')
    assert 'torch_npu' not in sys.modules
    value = dict(status='PASS_HOST_original_tree_packing_and_verifier_for_primary3_alternate3',
                 exact_mask_and_position_cases=mask_cases, actual_original_verdict_cases=verifier_cases,
                 budget_vectors=resource_cases, short_primary_plans_match_old_B3_semantics=True,
                 raw4_step_primary_bridge_retained=True, raw8_pool_not_relabelled_as6_generated_nodes=True,
                 old_P4_family_classifier_rejected_for_new_branch=True,
                 NPU_used=False, serving_installed=False, new_target_branch_projection_implemented=False,
                 new_eager_geometry_qualified=False, full463_qualified=False, Goodput=None,
                 source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (NATIVE / 'tree.py', NATIVE / 'native_engine.py')})
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
