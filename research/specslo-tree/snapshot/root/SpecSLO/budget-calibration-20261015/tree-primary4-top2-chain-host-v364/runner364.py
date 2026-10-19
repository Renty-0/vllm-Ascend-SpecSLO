"""Private raw-spine graph family; preserve all Native capture/check bodies."""
import ast
import copy
from contextlib import contextmanager
import inspect
from pathlib import Path
from types import MethodType

from spine364 import execute


SOURCE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')
MARKER = '|ROOT364:raw-spine4-top2-float32-bits'


def project(function, helper):
    tree = ast.parse(SOURCE.read_text())
    node = copy.deepcopy(next(node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'run_draft_greedy'))
    if function.__code__.co_filename != str(SOURCE) or function.__name__ != node.name:
        raise RuntimeError('ROOT364 actual original Native draft dispatcher required')
    before = copy.deepcopy(node)
    assignments = [item for item in ast.walk(node) if isinstance(item, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'entry_key' for target in item.targets)]
    if len(assignments) != 1:
        raise RuntimeError('ROOT364 one original draft entry-key declaration required')
    assignment = assignments[0]
    value = copy.deepcopy(assignment.value)
    assignment.value = ast.Call(func=ast.Name(id='_ROOT364_raw_key', ctx=ast.Load()),
                                args=[ast.Name(id='self', ctx=ast.Load()), value], keywords=[])
    projected = copy.deepcopy(node)
    assignment.value = value
    if ast.dump(node, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('ROOT364 entire original dispatcher AST inverse required')
    namespace = dict(function.__globals__, _ROOT364_raw_key=helper)
    flags = function.__code__.co_flags & __import__('__future__').annotations.compiler_flag
    exec(compile(ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[])),
                 __file__ + '::original-Native-dispatcher-one-private-key', 'exec',
                 flags=flags, dont_inherit=True), namespace)
    result = namespace['run_draft_greedy']
    if inspect.signature(result) != inspect.signature(function):
        raise RuntimeError('ROOT364 original Native draft signature required')
    return result


class State:
    def __init__(self, runner):
        self.runner = runner
        self.active = False
        self.logical_row_count = None
        self.raw_execute_calls = 0
        self.dispatch_calls = 0
        self.original_run = runner.run_draft_greedy
        self.original_execute = runner._execute_draft

    @contextmanager
    def mode(self, logical_row_count):
        if self.active:
            raise RuntimeError('ROOT364 private graph mode cannot nest')
        self.active = True
        self.logical_row_count = logical_row_count
        try:
            yield
        finally:
            self.active = False
            self.logical_row_count = None

    def run(self, input_ids, positions, metadatas, vocabulary_size, *, valid_row_count,
            final_kv_only=False, logical_row_count=None):
        logical_row_count = input_ids.numel() if logical_row_count is None else logical_row_count
        with self.mode(logical_row_count):
            result = self.runner.run_draft_greedy(input_ids, positions, metadatas,
                vocabulary_size, valid_row_count=valid_row_count,
                final_kv_only=final_kv_only, return_confidence=True)
        self.dispatch_calls += 1
        return result


def install(runner):
    if hasattr(runner, '_ROOT364_raw_spine_state'):
        raise RuntimeError('ROOT364 one private installation per runner')
    state = State(runner)

    def key(this, value):
        if this is not runner:
            raise RuntimeError('ROOT364 same original draft runner required')
        return (value[0] + MARKER + f'|logical:{state.logical_row_count}', value[1]) if state.active else value

    patched = project(state.original_run.__func__, key)

    def executed(this, input_ids, positions, attention_metadatas, vocabulary_size, *,
                 final_kv_only=False, return_confidence=False):
        if this is not runner:
            raise RuntimeError('ROOT364 same model execution owner required')
        if not state.active:
            return state.original_execute(input_ids, positions, attention_metadatas, vocabulary_size,
                final_kv_only=final_kv_only, return_confidence=return_confidence)
        if not return_confidence:
            raise RuntimeError('ROOT364 explicit two-word lossless output contract required')
        state.raw_execute_calls += 1
        return execute(this.model, input_ids, positions, attention_metadatas, vocabulary_size,
                       final_kv_only=final_kv_only, logical_row_count=state.logical_row_count)

    runner.run_draft_greedy = MethodType(patched, runner)
    runner._execute_draft = MethodType(executed, runner)
    runner._ROOT364_raw_spine_state = state
    return state
