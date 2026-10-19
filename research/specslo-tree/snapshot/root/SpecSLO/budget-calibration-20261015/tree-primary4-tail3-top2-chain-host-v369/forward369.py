"""HOST candidate: retain the original first/catch-up step, graph the final3.

Only a projected function is produced. This module installs no serving method
and does not issue a device qualification. Original eager/frontier processing,
first-step catch-up, raw8 order, confidence aggregation and publication remain
in the original Native function body.
"""
import ast
import copy
import inspect
from pathlib import Path

from runner369 import install
from spine369 import decode


SOURCE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')


def project_forward(original, finish):
    tree = ast.parse(SOURCE.read_text())
    node = copy.deepcopy(next(n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == 'draft_tree_forward'))
    unwrapped = inspect.unwrap(original)
    if unwrapped.__name__ != node.name or unwrapped.__code__.co_filename != str(SOURCE):
        raise RuntimeError('ROOT369 actual original Native draft expansion required')
    before = copy.deepcopy(node)
    loops = [(i, n) for i, n in enumerate(node.body) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'depth_index']
    if len(loops) != 1:
        raise RuntimeError('ROOT369 one original outer depth loop required')
    index, loop = loops[0]
    old_iter = copy.deepcopy(loop.iter)
    loop.iter = ast.parse('range(1)', mode='eval').body
    extra = ast.parse('''
_ROOT369_tail = _ROOT369_finish(self, work_plans, request_ids, rows, node_confidences)
graph_calls += _ROOT369_tail['graph_calls']
model_calls += _ROOT369_tail['model_calls']
query_count += _ROOT369_tail['query_count']
graph_padding_query_count += _ROOT369_tail['graph_padding_query_count']
''').body
    node.body[index + 1:index + 1] = extra
    projected = copy.deepcopy(node)
    del node.body[index + 1:index + 1 + len(extra)]
    loop.iter = old_iter
    if ast.dump(node, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('ROOT369 entire original function AST inverse required')
    namespace = dict(unwrapped.__globals__, _ROOT369_finish=finish)
    flags = unwrapped.__code__.co_flags & __import__('__future__').annotations.compiler_flag
    exec(compile(ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[])),
        __file__ + '::original-first-step-then-three-step-tail', 'exec',
        flags=flags, dont_inherit=True), namespace)
    result = namespace['draft_tree_forward']
    if inspect.signature(result) != inspect.signature(original):
        raise RuntimeError('ROOT369 unchanged original signature')
    return result


class TailState:
    def __init__(self, engine):
        self.engine = engine
        self.runner = install(engine.graph_runner)
        self.calls = []

    def finish(self, engine, plans, owners, rows, confidences):
        if engine is not self.engine or not plans or len(plans) != len(owners):
            raise RuntimeError('ROOT369 one original draft model and aligned owners')
        if any(int(plan.width) != 2 or int(plan.depth) != 4 for plan in plans):
            raise RuntimeError('ROOT369 original width2 depth4 raw8 protocol')
        if any(row[0] < 0 or row[4] < 0 for row in rows):
            raise RuntimeError('ROOT369 original first top2 must already exist')
        inputs = None
        positions, metadatas = [], []
        padding_queries = 0
        physical_queries = 0
        for depth in (1, 2, 3):
            requests = [(plan, owner, depth - 1, row[0] if depth == 1 else 0)
                        for plan, owner, row in zip(plans, owners, rows, strict=True)]
            padded, padding = engine._pad_tree_draft_graph_requests(requests)
            current, pos, metadata = engine._pack_tree_draft_level(padded)
            if inputs is None:
                inputs = current
            if current.numel() != inputs.numel():
                raise RuntimeError('ROOT369 same original physical shape across final3')
            positions.append(pos)
            metadatas.append(metadata)
            padding_queries += padding
            physical_queries += current.numel()
        output = self.runner.run(inputs, positions, metadatas, engine.draft_vocab_size,
            valid_row_count=inputs.numel(), logical_row_count=len(owners))
        if engine.graph_runner.last_draft_execution.used_aclgraph is not True:
            raise RuntimeError('ROOT369 actual Native tail graph required')
        tokens, probabilities = decode(output[:len(owners)].detach().cpu(), engine.draft_vocab_size)
        for row, probability, new_tokens, new_probabilities in zip(
                rows, confidences, tokens.tolist(), probabilities.tolist(), strict=True):
            for slot, token, score in zip((1, 2, 3, 5, 6, 7), new_tokens, new_probabilities, strict=True):
                row[slot] = int(token)
                probability[slot] = float(score)
        value = dict(graph_calls=1, model_calls=1, query_count=physical_queries,
                     graph_padding_query_count=padding_queries, actual_model_steps=3,
                     logical_owners=len(owners), original_first_step_preserved=True,
                     final_leaf_KV_materialized=False)
        self.calls.append(value)
        return value


def build(original, engine):
    """Build for explicit component invocation; no global or instance install."""
    state = TailState(engine)
    projected = project_forward(original, state.finish)

    def call(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        plans = values['plans']
        temperatures = values['draft_temperatures'] or [0.0] * len(plans)
        eligible = (values['self'] is engine and engine.is_draft
            and engine.graph_runner.enabled and not values['profile_subphases']
            and not getattr(engine, '_profile_tree_subphases', False)
            and all(int(plan.width) == 2 and int(plan.depth) == 4 for plan in plans)
            and all(float(value) == 0 for value in temperatures))
        return projected(*args, **kwargs) if eligible else original(*args, **kwargs)

    return call, state
