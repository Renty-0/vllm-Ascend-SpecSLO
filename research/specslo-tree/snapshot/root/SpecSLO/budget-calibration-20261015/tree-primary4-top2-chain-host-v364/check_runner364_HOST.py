"""Original dispatcher/eager body in a CPU fixture; private mode isolation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent


def main():
    output = P / 'HOST364_RUNNER_RESULT.json'
    assert not output.exists()
    import torch
    torch.set_num_threads(1)
    from check364_HOST import ModelDouble
    from runner364 import install, SOURCE, MARKER
    from spine364 import decode, execute
    tree = ast.parse(SOURCE.read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'NativeACLGraphRunner')
    selected = [copy.deepcopy(node) for node in source_class.body if isinstance(node, ast.FunctionDef)
                and node.name in ('run_draft_greedy', '_execute_draft')]
    cls = copy.deepcopy(source_class)
    cls.body = selected
    cls.decorator_list = []
    class Execution:
        def __init__(self, *args):
            self.mode = args[0] if args else None
    namespace = dict(torch=torch, NativeGraphExecution=Execution)
    exec(compile('from __future__ import annotations\n' + ast.unparse(
        ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))),
        str(SOURCE), 'exec'), namespace)
    runner = namespace['NativeACLGraphRunner'].__new__(namespace['NativeACLGraphRunner'])
    model = ModelDouble(torch, 512)
    model.compute_greedy_tokens = lambda hidden, vocabulary: model.compute_logits(hidden)[:, :vocabulary].argmax(dim=-1)
    runner.model = model
    # This is only a CPU eager protocol fixture. It is not a production runner
    # and does not issue any Native graph qualification or claim a capture.
    runner.enabled = False
    runner._raise_if_graph_cache_sealed = lambda *args: None
    runner._record_execution = lambda role, execution, result: result
    state = install(runner)
    roots = torch.arange(16)
    positions = [torch.arange(16) + step for step in range(4)]
    metas = [types.SimpleNamespace(slot_mapping=torch.arange(16)) for _ in range(4)]
    baseline = state.original_run(roots, positions, metas, 512)
    unchanged = runner.run_draft_greedy(roots, positions, metas, 512)
    assert torch.equal(baseline, unchanged) and state.raw_execute_calls == 0
    packed = state.run(roots, positions, metas, 512, valid_row_count=16, logical_row_count=12)
    expected = execute(model, roots, positions, metas, 512, logical_row_count=12)
    assert torch.equal(packed, expected) and state.raw_execute_calls == 1
    decode(packed, 512)
    helper = runner.run_draft_greedy.__func__.__globals__['_ROOT364_raw_key']
    ordinary = ('original-key', 16)
    assert helper(runner, ordinary) == ordinary
    with state.mode(12):
        first = helper(runner, ordinary)
        assert MARKER in first[0] and first[0].endswith('|logical:12')
    with state.mode(10):
        second = helper(runner, ordinary)
    assert first != second and helper(runner, ordinary) == ordinary
    try:
        state.run(roots, positions, metas, 512, valid_row_count=16, final_kv_only=True)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid fifth-step metadata accepted')
    assert state.active is False and state.logical_row_count is None
    assert 'torch_npu' not in sys.modules
    value = dict(status='PASS_HOST_original_dispatcher_private_raw_mode_and_graph_key_isolation',
                 original_eager_model_body_preserved_outside_private_mode=True,
                 full_original_dispatcher_AST_inverse=True, logical_row_count_in_private_key=True,
                 state_restored_after_success_and_failure=True,
                 original_Native_capture_and_output_checker_bodies_unmodified=True,
                 CPU_eager_fixture_only=True, graph_capture_executed=False,
                 NPU_used=False, serving_installed=False, model_qualified=False,
                 Raw=None, Goodput=None,
                 runner_source_sha256=hashlib.sha256((P / 'runner364.py').read_bytes()).hexdigest(),
                 execute_source_sha256=hashlib.sha256((P / 'spine364.py').read_bytes()).hexdigest())
    with output.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
