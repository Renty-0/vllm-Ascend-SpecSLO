"""Execute the actual original Native depth loop against a dependent CPU model.

CPU fixture submission objects simulate graph return metadata only. This is a
protocol regression test, not a graph, model, KV, or NPU qualification.
"""
import ast
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace, MethodType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tree-primary4-top2-chain-host-v364'))


def main():
    import torch
    torch.set_num_threads(1)
    from check364_HOST import ModelDouble
    from forward369 import SOURCE, TailState, project_forward
    from spine369 import execute
    result_path = HERE / 'HOST369_RESULT.json'
    assert not result_path.exists()

    def plan(width, depth, prefix, maximum, *, candidate_budget):
        count = width * depth
        return SimpleNamespace(width=width, depth=depth, prefix_len=prefix,
            candidate_budget=candidate_budget,
            parent_indices=torch.tensor([-1] + list(range(count - 1))),
            positions=torch.arange(count + 1) + prefix,
            cache_positions=torch.arange(count + 1) + prefix)

    source = SOURCE.read_text()
    node = copy.deepcopy(next(n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == 'draft_tree_forward'))
    class CPUProtocolRunner(SimpleNamespace):
        pass

    namespace = dict(torch=torch, time=time, cached_cpu_tree_speculation_plan=plan,
        NativeACLGraphRunner=CPUProtocolRunner,
        _trace_region=lambda *args: nullcontext(),
        envs=SimpleNamespace(VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH=True))
    exec(compile('from __future__ import annotations\n' + ast.unparse(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))), str(SOURCE), 'exec'), namespace)
    original = namespace['draft_tree_forward']

    class Engine:
        def __init__(self, vocabulary):
            self.is_draft = True
            self.gamma = 4
            self.draft_vocab_size = vocabulary
            self.device = torch.device('cpu')
            self.cache_allocation = SimpleNamespace(block_tables=[])
            self.cache_block_tables = torch.ones((32, 32), dtype=torch.int32)
            self._defer_tree_materialization = True
            self.model = ModelDouble(torch, vocabulary)
            self.config = SimpleNamespace(max_model_len=4096, enforce_eager=False)
            self._ensure_cache_capacity = lambda *args: None
            self.graph_runner = CPUProtocolRunner(enabled=True,
                last_target_execution=SimpleNamespace(used_aclgraph=True),
                last_draft_execution=SimpleNamespace(used_aclgraph=True))
            self.graph_runner.run_tree_logits = lambda inputs, positions, metadata, vocabulary: self.model.compute_logits(
                self.model(inputs, positions, metadata))
        def _pad_tree_draft_graph_requests(self, requests):
            return requests, 0
        def _pack_tree_draft_level(self, requests):
            inputs, positions = [], []
            for p, owner, indices, tokens in requests:
                indices = [indices] if isinstance(indices, int) else indices
                tokens = [tokens] if isinstance(tokens, int) else tokens
                inputs.extend(tokens)
                positions.extend(p.prefix_len + index + 1 for index in indices)
            inputs = torch.tensor(inputs, dtype=torch.long)
            positions = torch.tensor(positions, dtype=torch.long)
            return inputs, positions, SimpleNamespace(slot_mapping=torch.arange(len(inputs)))

    cases = probabilities = 0
    for n in (1, 4, 12, 24, 32):
        for vocabulary in (64, 151936):
            for with_catchup in (False, True):
                old, new = Engine(vocabulary), Engine(vocabulary)
                plans = [plan(2, 4, 127 + i % 3, 4096, candidate_budget=1 + i % 8) for i in range(n)]
                roots = [i % vocabulary for i in range(n)]
                catchup = [[(root - j) % vocabulary for j in range(i % 5, -1, -1)]
                           for i, root in enumerate(roots)] if with_catchup else None
                tail = TailState.__new__(TailState)
                tail.engine = new
                tail.calls = []
                def cpu_run(inputs, positions, metadata, vocabulary, *, valid_row_count, logical_row_count):
                    assert valid_row_count == inputs.numel() == logical_row_count == n
                    return execute(new.model, inputs, positions, metadata, vocabulary,
                                   logical_row_count=logical_row_count)
                tail.runner = SimpleNamespace(run=cpu_run)
                patched = project_forward(original, tail.finish)
                kwargs = dict(committed_catchup_token_ids=catchup)
                expected = original(old, plans, roots, list(range(n)), **kwargs)
                actual = patched(new, plans, roots, list(range(n)), **kwargs)
                for key in ('draft_token_ids', 'root_token_ids', 'eager_frontier_tokens',
                            'num_draft_tokens', 'query_count', 'graph_padding_query_count',
                            'materialization_deferred', 'cache_slot_mapping'):
                    assert actual[key] == expected[key], key
                assert torch.equal(actual['parent_indices'], expected['parent_indices'])
                assert torch.equal(actual['draft_confidence'].view(torch.int32), expected['draft_confidence'].view(torch.int32))
                for a, b in zip(actual['node_confidences'], expected['node_confidences'], strict=True):
                    assert torch.equal(a.view(torch.int32), b.view(torch.int32))
                    probabilities += a.numel()
                assert len(new.model.inputs) == len(old.model.inputs) == 4
                assert all(torch.equal(a, b) for a, b in zip(new.model.inputs, old.model.inputs, strict=True))
                assert expected['model_calls'] == expected['graph_calls'] == 4
                assert actual['model_calls'] == actual['graph_calls'] == 2
                assert len(tail.calls) == 1 and tail.calls[0]['actual_model_steps'] == 3
                cases += 1
    assert 'torch_npu' not in sys.modules
    value = dict(status='PASS_HOST_actual_original_depth_loop_first_catchup_plus_tail3',
        cases=cases, original_FP32_node_probabilities_compared=probabilities,
        original_first_call_and_all_four_dependent_model_inputs_exact=True,
        original_raw8_and_mean_confidence_bitwise=True,
        logical_model_steps=4, CPU_simulated_submissions_before=4, CPU_simulated_submissions_after=2,
        complete_original_function_AST_inverse=True,
        CPU_simulated_graph_submission_metadata=True, graph_capture_executed=False,
        NPU_used=False, model_qualified=False, serving_installed=False, Raw=None, Goodput=None,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in HERE.glob('*.py')})
    with result_path.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
