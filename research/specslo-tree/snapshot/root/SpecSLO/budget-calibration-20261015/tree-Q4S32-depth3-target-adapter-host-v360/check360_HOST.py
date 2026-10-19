"""Run actual Native forward/allocator with a CPU input identity model double."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


P = Path(__file__).resolve().parent
B = P.parent
FIXTURE = B / 'tree-static-q5-M120-full463-own-AR-v256/check_full256_HOST.py'


def main():
    output = P / 'HOST360_RESULT.json'
    assert not output.exists()
    spec = importlib.util.spec_from_file_location('_ROOT360_original_fixture', FIXTURE)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    node = copy.deepcopy(next(n for n in ast.parse(FIXTURE.read_text()).body
                              if isinstance(n, ast.FunctionDef) and n.name == 'main'))
    cut = next(i for i, n in enumerate(node.body)
               if isinstance(n, ast.ImportFrom) and n.module == 'static_state256')
    node.name = '_ROOT360_fixture'
    node.body = node.body[:cut] + [ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()),
                                                          args=[], keywords=[]))]
    namespace = dict(vars(fixture))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(FIXTURE) + '::prefix-before-old-adapter-install', 'exec'), namespace)
    env = namespace[node.name]()
    engine, native = env['engine'], env['native']
    for directory in ('tree-depth3-branch6-protocol-host-v357',
                      'tree-alternate-chain-target-host-v269',
                      'tree-Q4S32-branch-projection-host-v358',
                      'tree-Q4S32-prefix-alias-split-host-v359',
                      'tree-single-validated-prefix-descriptors-host-v304',
                      'tree-fused-branch-KV-row-copy-v262'):
        sys.path.insert(0, str(B / directory))
    import torch
    from loader360 import source
    sources = {}
    for which in ('branch_tail', 'static_base', 'static_state', 'write'):
        projected, changes = source(which)
        compile(projected, which, 'exec')
        sources[which] = dict(changes=changes, sha256=hashlib.sha256(projected.encode()).hexdigest())
    from plan357 import make, spare_side_owners
    from padding360 import pad

    def identity_runner(xs, positions, metadata, vocabulary_size):
        assert len(xs) == len(positions) == len(metadata) == 1
        meta = metadata[0]
        assert xs[0].numel() == 128 and meta.actual_seq_lengths_q == tuple(range(4, 129, 4))
        assert meta.sequence_lens == (2560,) * 32
        assert tuple(meta.tree_attention_mask.shape) in ((32, 1, 10, 4096), (32, 1, 11, 4096))
        return (engine.model.compute_greedy_tokens(engine.model(xs[0], positions[0], meta), vocabulary_size),)

    engine.graph_runner.run_target_greedy = identity_runner
    import branch360
    from static_base360 import install
    from static_state360 import bind_checked_prepare
    activation = install(engine, native)
    state = engine._branch_tail_state
    bind_checked_prepare(state, branch360)
    forward = native.NativePearlEngine.target_tree_forward
    frames = branches = root_dummies = identity = 0

    def one(budgets, prefix=127):
        nonlocal frames, branches, root_dummies, identity
        plans = [make(budget, prefix).plan for budget in budgets]
        roots = list(range(41, 41 + len(plans)))
        rows = [[103 + 17 * owner + j for j in range(budget)] for owner, budget in enumerate(budgets)]
        padded = pad(engine, plans, roots, rows, list(range(len(plans))))
        pp, rr, dd, ss, n = padded
        result = forward(engine, pp, rr, dd, sequence_ids=ss, return_logits=False, real_tree_count=n)
        q = n + sum(budgets)
        assert result['query_count'] == 128 and result['logical_query_count'] == q
        assert result['graph_padding_query_count'] == 128 - q
        assert result['tree_count'] == n and result['num_draft_tokens'] == budgets
        assert result['target_query_token_ids'].tolist() == [token for root, row in zip(roots, rows, strict=True)
                                                           for token in [root] + row]
        assert result['cache_slot_mapping'].unique().numel() == q
        current = state.static_current
        meta = current['physical'][2]
        assert meta is engine.model.last_metadata
        if any(budget > 3 for budget in budgets):
            assert state.copy326_contract is not None and state.split326_last_HOST is not None
            assert state.alias326_destination.numel() == 128 and state.prefix326_spans.shape == (12, 3)
            assert meta.tree_attention_mask.shape[2] == 11
            zero = state.static_zero_page
            assert all(destination // 128 != zero for destination in state.copy326_contract.destination)
        else:
            assert state.split326_last_HOST is None and meta.tree_attention_mask.shape[2] == 10
        assert state.enabled is False
        frames += 1
        branches += sum(budget > 3 for budget in budgets)
        root_dummies += any(item.candidate_budget == 0 for item in pp)
        identity += current['physical'][4]['physical_to_original'] == list(range(128))

    for n in range(1, 33):
        for side in range(spare_side_owners(n) + 1):
            one([6] * side + [3] * (n - side))
    for prefix in (0, 1, 126, 128, 129, 1505, 2556):
        for budget in range(1, 7):
            one([budget], prefix)
    assert root_dummies > 0 and identity > 0 and branches > 0
    assert 'torch_npu' not in sys.modules
    assert activation['fixed_M'] == 128 and activation['fixed_q4_groups'] == 32
    assert activation['production_qualified'] is activation['whole_workload_AR_qualified'] is False
    result = dict(status='PASS_HOST_actual_Native_protocol_Q4S32_depth3_projection_and_split128',
                  frames=frames, alternate_owner_occurrences=branches,
                  frames_with_typed_root_padding=root_dummies, identity_projection_frames=identity,
                  original_Native_forward_and_allocator_executed=True,
                  all_real_tokens_budgets_and_KV_coordinates_preserved=True,
                  physical_metadata_is_actual_model_input=True, reserved_zero_not_branch_scratch=True,
                  complete_original_module_AST_inverses=sources,
                  model_double='CPU input/position identity; no attention, neural arithmetic or ACLGraph execution',
                  actual_device_copy_kernel_executed=False,
                  NPU_used=False, serving_installed=False, neural_math_qualified=False,
                  eager_promotion_qualified=False, full463_qualified=False, Raw=None, Goodput=None)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
