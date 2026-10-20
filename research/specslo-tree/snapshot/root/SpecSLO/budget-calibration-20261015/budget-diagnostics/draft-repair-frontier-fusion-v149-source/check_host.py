"""Small actual HOST functional checks, not a device/numerical qualifier.

Execute frozen Native dataclass, pad, hidden-return and head method AST bodies
with explicit list-backed tensor/model stubs. Native modules are not imported.
"""
import ast
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from repair_frontier_fusion import (
    AllocationWitness, METADATA_FIELDS, build_union, plan_frame, run_union,
)

ROOT = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
EXPECTED = {
    'native_graph.py': 'a29c1d18e1588af8c94bb1b598af2da19ffef7c4630e87844befe086a69f3bf8',
    'native_model.py': 'e0d9efe4572c7e87ad860b014ef146394ca44f53ac8d87d12edfed2b572422a1',
    'native_engine.py': '05d47ed6dc48e368010ca145a0b859aaffe596f246e0b28eb7b07d2d51a50cc9',
}


class Tensor:
    """Only list operations needed by original Native padding, never Torch."""
    def __init__(self, data, dtype='int64', device='host', *, parent=None, rows=None):
        self._data = copy.deepcopy(data)
        self.dtype, self.device = dtype, device
        self.parent, self.rows = parent, rows

    @property
    def data(self):
        if self.parent is None:
            return self._data
        return [self.parent.data[i] for i in self.rows]

    @property
    def shape(self):
        a = self.data
        return (len(a), len(a[0])) if a and isinstance(a[0], list) else (len(a),)

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Tensor([], self.dtype, self.device, parent=self,
                          rows=list(range(self.shape[0]))[key])
        return Tensor(self.data[key], self.dtype, self.device)

    def copy_(self, other):
        values = other.data
        if self.ndim == 2 and other.ndim == 1:
            values = [copy.deepcopy(values) for _ in range(self.shape[0])]
        assert self.shape == Tensor(values).shape
        if self.parent is None:
            self._data = copy.deepcopy(values)
        else:
            for i, value in zip(self.rows, values, strict=True):
                assert self.parent.parent is None, 'stub only needs one-level writes'
                self.parent._data[i] = copy.deepcopy(value)
        return self


class TensorOps:
    @staticmethod
    def full(shape, value, dtype='int64', device='host'):
        if type(shape) is int:
            shape = (shape,)
        data = [value] * shape[0] if len(shape) == 1 else [[value] * shape[1] for _ in range(shape[0])]
        return Tensor(data, dtype, device)

    @staticmethod
    def zeros(shape, dtype='int64', device='host'):
        return TensorOps.full(shape, 0, dtype, device)

    @staticmethod
    def ones(shape, dtype='int64', device='host'):
        return TensorOps.full(shape, 1, dtype, device)

    @staticmethod
    def cat(values, dim=0):
        assert dim == 0 and values and all((v.dtype, v.device) == (values[0].dtype, values[0].device) for v in values)
        return Tensor([row for value in values for row in value.data], values[0].dtype, values[0].device)


def node(file, cls, name):
    tree = ast.parse((ROOT / file).read_text())
    owner = tree if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def actual_function(file, cls, name, namespace):
    selected = copy.deepcopy(node(file, cls, name))
    selected.decorator_list = []
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / file) + ':HOST_AST', 'exec'), namespace)
    return namespace[name]


def load_originals():
    for name, sha in EXPECTED.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    tree = ast.parse((ROOT / 'native_model.py').read_text())
    metadata = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeAttentionMetadata'))
    assert tuple(n.target.id for n in metadata.body if isinstance(n, ast.AnnAssign)) == METADATA_FIELDS
    namespace = dict(dataclass=dataclass)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), metadata], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), 'actual-NativeAttentionMetadata:HOST_AST', 'exec'), namespace)
    pad = actual_function('native_graph.py', 'NativeACLGraphRunner', '_pad_inputs', dict(torch=TensorOps, replace=replace))
    return namespace['NativeAttentionMetadata'], pad


BUCKET_NAMESPACE = {}
actual_function('native_engine.py', None, '_linear_draft_graph_buckets', BUCKET_NAMESPACE)
bucket = actual_function('native_engine.py', None, '_next_linear_draft_graph_bucket', BUCKET_NAMESPACE)


def witness():
    # Distinct pages for96 logical owners, deliberately including IDs>64.
    return AllocationWitness(8675309, 128, tuple((2*i, 2*i+1) for i in range(96)))


def groups(E, R):
    return ([(64+i, 130+i, [700+i], 800+i) for i in range(R)],
            [(i, 129+i, 300+i) for i in range(E)])


def setup(E, R):
    repairs, frontiers = groups(E, R)
    w = witness()
    p = plan_frame(repairs, frontiers, next_bucket=bucket, witness=w)
    Meta, pad = load_originals()

    def prepare(ids, positions, fused):
        # Explicit HOST producer stub, not an import/execution of Native model.
        assert fused is False and len(ids) == len(set(ids))
        return Tensor(positions), Meta(
            slot_mapping=Tensor([w.block_tables[o][p//w.block_size]*w.block_size+p%w.block_size for o, p in zip(ids, positions)], 'int32'),
            context_lens=Tensor([p+1 for p in positions], 'int32'),
            block_tables=Tensor([list(w.block_tables[o]) for o in ids], 'int32'),
            actual_seq_lengths_q=tuple(range(1, len(ids)+1)),
            sequence_lens=tuple(p+1 for p in positions))
    packed = build_union(p, tensor=Tensor, prepare=prepare, pad=pad,
                         cat=TensorOps.cat, lease=lambda: w)
    return p, packed, prepare, pad


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('negative case was accepted')


def main():
    started = time.perf_counter()
    checks = []
    for E, R, expected in ((1, 1, (1, 2)), (32, 32, (32, 64)), (15, 1, (16, 20))):
        p, (tokens, positions, meta), _, _ = setup(E, R)
        assert (p.frontier_M, p.union_M) == expected
        assert tokens.data[:E] == [x[2] for x in p.frontiers]
        assert tokens.data[p.frontier_M:p.frontier_M+R] == [x[2][0] for x in p.repairs]
        real_slots = meta.slot_mapping.data[:E] + meta.slot_mapping.data[p.frontier_M:p.frontier_M+R]
        assert tuple(real_slots) == p.write_slots
        for i in list(range(E, p.frontier_M)) + list(range(p.frontier_M+R, p.union_M)):
            assert (tokens.data[i], positions.data[i], meta.context_lens.data[i], meta.slot_mapping.data[i]) == (0, 0, 1, -1)
            assert meta.block_tables.data[i] == meta.block_tables.data[0]
        assert meta.actual_seq_lengths_q == tuple(range(1, p.frontier_M+R+1))
        if E == 15:
            assert p.old_repair_M + p.frontier_M == 17 < p.union_M == 20
        checks.append(f'actual_original_pad/layout_E{E}_R{R}_F{p.frontier_M}_U{p.union_M}')

    repairs, frontiers = groups(1, 1)
    assert plan_frame([], frontiers, next_bucket=bucket, witness=None) is None
    assert plan_frame([(64, 130, [1, 2], 3)], frontiers, next_bucket=bucket, witness=None) is None
    rejected(lambda: plan_frame([(0, 130, [1], 3)], frontiers, next_bucket=bucket, witness=witness()))
    tables = list(witness().block_tables); tables[64] = tables[0]
    rejected(lambda: plan_frame(repairs, frontiers, next_bucket=bucket, witness=replace(witness(), block_tables=tuple(tables))))
    tables = list(witness().block_tables); tables[64] = (128, -1)
    rejected(lambda: plan_frame(repairs, frontiers, next_bucket=bucket, witness=replace(witness(), block_tables=tuple(tables))))
    checks.append('original_fallback_and_owner_page_unallocated_negatives')

    p, packed, prepare, pad = setup(15, 1)
    changed = replace(p.witness, allocation_id=p.witness.allocation_id+1)
    calls = iter((p.witness, changed))
    rejected(lambda: build_union(p, tensor=Tensor, prepare=prepare, pad=pad, cat=TensorOps.cat, lease=lambda: next(calls)))
    def bad_prepare(ids, positions, mode):
        pos, meta = prepare(ids, positions, mode)
        return pos, replace(meta, tree_attention=True)
    rejected(lambda: build_union(p, tensor=Tensor, prepare=bad_prepare, pad=pad, cat=TensorOps.cat, lease=lambda: p.witness))
    checks.append('stale_lease_and_non_PA_metadata_rejected')

    class Model:
        def __init__(self):
            self.head_rows = []
            self.lm_head = SimpleNamespace(greedy=self.head)
        def __call__(self, inputs, positions, metadata):
            return Tensor([[t, pos] for t, pos in zip(inputs.data, positions.data)])
        def head(self, hidden, vocabulary_size):
            self.head_rows.append((hidden.shape[0], vocabulary_size))
            return Tensor([row[0]+1 for row in hidden.data])
    Model.compute_greedy_tokens = actual_function('native_model.py', 'NativeQwen2ForCausalLM', 'compute_greedy_tokens', {})
    class Runner:
        def __init__(self, model): self.model, self.calls = model, []
        def _run_target(self, inputs, positions, metadata, vocabulary_size, *, output_kind):
            self.calls.append((tuple(x.shape[0] for x in inputs), vocabulary_size, output_kind))
            return self._execute_target(inputs, positions, metadata, vocabulary_size, output_kind=output_kind)
    Runner._execute_target = actual_function('native_graph.py', 'NativeACLGraphRunner', '_execute_target', {})
    Runner.run_tree_hidden = actual_function('native_graph.py', 'NativeACLGraphRunner', 'run_tree_hidden', {})
    model = Model(); runner = Runner(model)
    answer = run_union(p, packed, runner=runner, model=model, vocabulary_size=151936, lease=lambda: p.witness)
    assert runner.calls == [((20,), 0, 'hidden')]
    assert model.head_rows == [(16, 151936)] and answer.data == [x[2]+1 for x in p.frontiers]
    rejected(lambda: run_union(p, packed, runner=runner, model=model, vocabulary_size=151936, lease=lambda: changed))
    assert len(runner.calls) == 1
    leases = iter((p.witness, changed))
    rejected(lambda: run_union(p, packed, runner=runner, model=model, vocabulary_size=151936, lease=lambda: next(leases)))
    assert len(runner.calls) == 2  # rejected after model/head, before any publication
    checks.append('actual_original_hidden_return_and_original_head_M16_not_E15_or_U20')
    assert not any(n == 'torch' or n.startswith(('torch.', 'torch_npu', 'acl')) for n in sys.modules)
    result = dict(schema='HOST.repair-frontier-fusion.prototype.v1', status='pass', checks=checks,
                  actual_check_count=len(checks), elapsed_seconds=time.perf_counter()-started,
                  Native_source_sha256=EXPECTED, Native_AST_executed=['NativeAttentionMetadata', '_pad_inputs',
                      '_linear_draft_graph_buckets', '_next_linear_draft_graph_bucket',
                      'run_tree_hidden', '_execute_target', 'compute_greedy_tokens'],
                  explicit_stubs=['list Tensor/TensorOps', 'prepare metadata', 'model body/head', '_run_target dispatch'],
                  source_imported=False, Torch_imported=False, NPU_used=False,
                  production_installed=False, numerical_qualified=False, cost_qualified=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
