"""Preserve RootSeed/FastV5; fuse only qualified all-eager frontier+chain calls."""
import ast
import copy
import hashlib
import inspect
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
LEGACY = BASE / 'tree-high-normal-cap4-native-families-v148'
FAST_SOURCE = LEGACY / 'fast_tree_draft_v5.py'
ROOT_SOURCE = LEGACY / 'root_seed.py'
SCALE = 1_000_000


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT373: ' + message)


def installed_fast(original):
    """Validate the actual production wrapper and its retained FastV5 closure."""
    need(original.__name__ == 'draft' and original.__code__.co_filename == str(ROOT_SOURCE),
         'actual installed RootSeed wrapper')
    closure = inspect.getclosurevars(original).nonlocals
    fast = closure.get('original')
    need(callable(fast) and fast.__name__ == 'draft' and fast.__code__.co_filename == str(FAST_SOURCE),
         'actual retained optimized FastV5 baseline, not bare Native')
    return fast


def project_fast(original, frontier, run, unpack):
    need(original.__name__ == 'draft' and original.__code__.co_filename == str(FAST_SOURCE),
         'exact installed FastV5 expansion')
    source = FAST_SOURCE.read_text()
    install = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    node = copy.deepcopy(next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'draft'))
    before = copy.deepcopy(node)
    frontier_blocks = [n for n in node.body if isinstance(n, ast.If)
                       and isinstance(n.test, ast.Name) and n.test.id == 'frontiers']
    need(len(frontier_blocks) == 1, 'one actual old frontier block')
    block = frontier_blocks[0]
    old_frontier = copy.deepcopy(block.body)
    block.body = ast.parse('''
_ROOT373_frontier_frame = _ROOT373_frontier(engine, frontiers)
frontier_total = _ROOT373_frontier_frame['physical']
frontier_padding = frontier_total - len(frontiers)
counts['eager_frontier_calls'] += 1
counts['eager_frontier_real_queries'] += len(frontiers)
counts['eager_frontier_padded_queries'] += frontier_total
''').body
    assignments = {n.targets[0].id: n for n in node.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    need('encoded' in assignments and 'values' in assignments, 'one original graph output and final readback')
    encoded = assignments['encoded']
    old_encoded = copy.deepcopy(encoded.value)
    encoded.value = ast.parse('_ROOT373_run(engine, _ROOT373_frontier_frame, padded_inputs, positions, metadatas, bucket, steps, len(ids))', mode='eval').body
    values = assignments['values']
    old_values = copy.deepcopy(values)
    value_index = node.body.index(values)
    extras = ast.parse('''
_ROOT373_all_values = encoded.cpu().tolist()
effective_roots, eager_frontier_tokens, values = _ROOT373_unpack(engine, frontiers, ids, _ROOT373_all_values, known)
''').body
    node.body[value_index:value_index + 1] = extras
    returns = [n for n in node.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
    need(len(returns) == 1, 'one original final output dictionary')
    returned = returns[0].value
    old_counts = []
    for i, key in enumerate(returned.keys):
        if isinstance(key, ast.Constant) and key.value in ('model_calls', 'graph_calls'):
            old_counts.append((i, copy.deepcopy(returned.values[i])))
            returned.values[i] = ast.parse('1 + repair_calls', mode='eval').body
    need(len(old_counts) == 2, 'truthful single fused graph/model submission counts')
    projected = copy.deepcopy(node)
    block.body = old_frontier
    encoded.value = old_encoded
    node.body[value_index:value_index + len(extras)] = [old_values]
    for i, value in old_counts:
        returned.values[i] = value
    need(ast.dump(node, include_attributes=False) == ast.dump(before, include_attributes=False),
         'entire original FastV5 AST inverse after declared frontier/readback/submission deltas')
    namespace = dict(original.__globals__, **inspect.getclosurevars(original).nonlocals)
    namespace.update(_ROOT373_frontier=frontier, _ROOT373_run=run, _ROOT373_unpack=unpack)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[])),
                 __file__ + '::original-FastV5-one-fused-frontier', 'exec'), namespace)
    function = namespace['draft']
    need(inspect.signature(function) == inspect.signature(original), 'original FastV5 signature')
    return function


class State:
    def __init__(self, engine, native):
        from runner373 import install
        self.engine = engine
        self.native = native
        self.graph = install(engine.graph_runner)
        self.calls = []

    def frontier(self, engine, frontiers):
        import torch
        need(engine is self.engine and frontiers, 'same model and actual frontier queries')
        ids = [seq for seq, _, _ in frontiers]
        positions = [pos for _, pos, _ in frontiers]
        tokens = [token for _, _, token in frontiers]
        engine._ensure_cache_capacity(ids, positions)
        inputs = torch.tensor(tokens, dtype=torch.long, device=engine.device)
        pos, metadata = engine._prepare_attention_metadata(ids, positions, False)
        physical = self.native._next_linear_draft_graph_bucket(len(ids), engine.config.max_num_seqs)
        inputs, pos, metadata = self.native.NativeACLGraphRunner._pad_inputs(inputs, pos, metadata, physical)
        return dict(inputs=inputs, positions=pos, metadata=metadata, physical=physical,
                    ids=ids, real=len(ids))

    def run(self, engine, frame, old_inputs, positions, metadatas, bucket, steps, logical):
        need(engine is self.engine and frame['real'] == logical and frame['physical'] == bucket
             and old_inputs.numel() == frame['inputs'].numel() == bucket,
             'all-eager identical original frontier/main physical bucket')
        need(steps == len(positions) == len(metadatas) and 1 <= steps <= 4, 'original actual chain steps')
        result = self.graph.run(frame['inputs'], [frame['positions']] + positions,
            [frame['metadata']] + metadatas, engine.draft_vocab_size,
            valid_row_count=logical, logical_row_count=logical)
        execution = engine.graph_runner.last_draft_execution
        need(execution.used_aclgraph is True, 'actual unmodified Native fused graph qualification')
        self.calls.append(dict(logical=logical, physical=bucket, chain_steps=steps,
            model_steps=steps + 1, graph_submissions=1, actual_physical_queries=(steps + 1) * bucket,
            frontier_CPU_readback_eliminated=True, original_weight_precision_unchanged=True))
        return result[:logical].clone()

    def unpack(self, engine, frontiers, ids, packed, known):
        from kv_ledger import record_overwrite
        need(engine is self.engine and [v[0] for v in frontiers] == list(ids) and len(packed) == len(ids),
             'original all-eager owner order')
        roots = [int(row[0][0]) // (SCALE + 1) for row in packed]
        need(all(0 <= token < engine.draft_vocab_size for token in roots), 'actual frontier root vocabulary')
        for seq, position, token in frontiers:
            record_overwrite(known[seq], position, [token])
        return roots, dict(zip(ids, roots, strict=True)), [row[1:] for row in packed]


def build(original, engine, native):
    fast = installed_fast(original)
    state = State(engine, native)
    projected = project_fast(fast, state.frontier, state.run, state.unpack)

    def proposed(current, plans, roots, ids, eager_parent_sources=None, **kwargs):
        eligible = (current is engine and engine.is_draft and 1 <= len(ids) <= 32
            and len(plans) == len(roots) == len(ids) and set(eager_parent_sources or {}) == set(ids)
            and all(int(p.width) == 2 and int(p.depth) == 4 for p in plans)
            and not any(kwargs.get('committed_catchup_token_ids') or ())
            and all(float(v) == 0 for v in (kwargs.get('draft_temperatures') or ()))
            and not int(getattr(engine.config, 'profile_host_decode_steps', 0)))
        if not eligible:
            return original(current, plans, roots, ids, eager_parent_sources=eager_parent_sources, **kwargs)
        return projected(current, plans, roots, ids, eager_parent_sources=eager_parent_sources, **kwargs)

    return proposed, state
