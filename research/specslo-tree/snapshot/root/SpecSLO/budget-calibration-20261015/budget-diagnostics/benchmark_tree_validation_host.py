"""Actual retained HOST benchmark of redundant FULL-attention ABI validation.

Extract the original function without importing Native or initializing NPU.
The proposed memo stores only HOST values and is cleared at each replay.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')


def original_validator():
    module = ast.parse(NATIVE.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_validate_tree_fia_mask')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NATIVE), 'exec'), namespace)
    return namespace[function.name], hashlib.sha256(ast.unparse(function).encode()).hexdigest()


class ReplayMemo:
    def __init__(self, validate):
        self.original = validate
        self.seen = set()
        self.calls = self.validations = self.hits = 0

    def reset(self):
        self.seen.clear()

    def __call__(self, query_count, mask, table, q, kv, block_size):
        self.calls += 1
        # Invalid/unrecognized inputs continue through the exact original
        # checker; no qualification or fallback counter is altered.
        if not isinstance(mask, torch.Tensor) or not isinstance(table, torch.Tensor):
            return self.original(query_count, mask, table, q, kv, block_size)
        key = (query_count, mask.ndim, mask.dtype, tuple(mask.shape),
               table.ndim, tuple(table.shape), tuple(q), tuple(kv), block_size)
        if key in self.seen:
            self.hits += 1
            return None
        self.original(query_count, mask, table, q, kv, block_size)
        self.validations += 1
        self.seen.add(key)


def fixture(trace, rows, segments):
    calls = [c for c in trace['normal_target_calls'] if c['family'] == 'full9'
             and c['physical_M'] == rows and c['physical_S'] == segments]
    assert calls
    call = calls[0]
    q, total = [], 0
    for length in call['physical_query_partition']:
        total += length
        q.append(total)
    assert len(q) == segments and total == rows
    kv = [context + length for context, length in zip(call['contexts'], call['physical_query_partition'])]
    kv += call['physical_query_partition'][call['N']:]
    assert len(kv) == segments
    return (rows, torch.empty((segments, 1, 9, 4096), dtype=torch.bool, device='cpu'),
            torch.empty((segments, 32), dtype=torch.int32, device='cpu'), q, kv, 128)


def main():
    import gzip
    trace_path = BASE / 'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1/rank1-full463-actual-trace.json.gz'
    trace = json.loads(gzip.decompress(trace_path.read_bytes()))
    validate, function_sha = original_validator()
    inputs = [fixture(trace, 96, 16), fixture(trace, 165, 38)]
    results = []
    for args in inputs:
        memo = ReplayMemo(validate)
        for _ in range(64):
            memo(*args)
        assert (memo.calls, memo.validations, memo.hits) == (64, 1, 63)
        negative_cases = [(*args[:3], args[3], [0] + args[4][1:], args[5]),
                          (args[0]-1, *args[1:]),
                          (args[0], args[1].to(dtype=torch.uint8), *args[2:]),
                          (args[0], args[1], None, *args[3:])]
        for bad in negative_cases:
            for checker in (validate, memo):
                try:
                    checker(*bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid metadata was admitted')
        validated_before = memo.validations
        memo.reset()
        memo(*args)
        assert memo.validations == validated_before + 1
        samples = {'original': [], 'replay_memo': []}
        cycles = 120
        for repeat in range(7):
            # Alternate order to avoid making the proposed path always warm.
            methods = ('original', 'replay_memo') if repeat % 2 else ('replay_memo', 'original')
            for method in methods:
                checker = validate if method == 'original' else memo
                start = time.perf_counter()
                for _ in range(cycles):
                    if method == 'replay_memo':
                        memo.reset()
                    for _ in range(64):
                        checker(*args)
                samples[method].append((time.perf_counter() - start) * 1000 / cycles)
        results.append(dict(M=args[0], S=args[1].shape[0], task_updates_per_replay=64,
                            replay_cycles_per_sample=cycles, paired_samples=7,
                            host_ms_per_64_validations=samples,
                            median_original_ms=statistics.median(samples['original']),
                            median_memo_ms=statistics.median(samples['replay_memo']),
                            median_paired_saved_ms=statistics.median(a-b for a,b in
                                zip(samples['original'], samples['replay_memo'], strict=True)),
                            negative_inputs_rejected_by_both=len(negative_cases)))
    output = dict(schema='ROOT.actual-HOST-FULL-validation-memo-benchmark.v1',
                  UTC=datetime.now(timezone.utc).isoformat(), executable=sys.executable,
                  python_version=sys.version, torch_version=torch.__version__,
                  actual_argv=sys.argv, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  Native_source=str(NATIVE), Native_source_sha256=hashlib.sha256(NATIVE.read_bytes()).hexdigest(),
                  extracted_function_AST_sha256=function_sha, actual_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  actual_CPU_tensor_metadata=True, original_length_values_from_actual_V96=True,
                  model_or_NPU_initialization=False, Native_imported=False,
                  cache_scope='cleared once at every replay; lengths copied into value-key each call',
                  original_Native_source_modified=False, device_performance_proved=False,
                  production_patch_installed=False, results=results)
    path = BASE / 'budget-diagnostics/TREE_FULL_VALIDATION_HOST_BENCHMARK.json'
    with path.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
