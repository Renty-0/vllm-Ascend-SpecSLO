"""Direct retained-Python HOST tests and small, unscored CPU microbenchmarks."""
import argparse
import ast
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import unittest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not 100 <= args.iterations <= 20000 or not 2 <= args.repeats <= 5:
        parser.error('bounded HOST benchmark requires iterations100..20000/repeats2..5')
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('output already exists; retain prior receipts')
    original_affinity = sorted(os.sched_getaffinity(0))
    pinned_cpu = original_affinity[0]
    os.sched_setaffinity(0, {pinned_cpu})
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import torch.utils._device as device_module
    import torch.overrides as overrides
    from cpu_default_device_cleanup_host import (
        capture_before_native_constructor, observe_cpu_device_context,
        restore_redundant_cpu_device_context, reinstall_redundant_cpu_device_context,
    )
    import test_cpu_default_device_cleanup_host as tests
    base = Path(__file__).resolve().parent
    paths = [Path(__file__), base/'cpu_default_device_cleanup_host.py',
             base/'test_cpu_default_device_cleanup_host.py', tests.TREE_SOURCE, tests.MODEL_SOURCE,
             Path(torch.__file__), Path(device_module.__file__), Path(overrides.__file__),
             base/'V103_ACTUAL_FIRST_WARMUP_HOST_PROFILE.json']
    before_hashes = {str(path): sha(path) for path in paths}
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    tests_seconds = time.perf_counter() - started
    if not result.wasSuccessful():
        raise RuntimeError('HOST tests failed:\n' + stream.getvalue())
    state = capture_before_native_constructor(torch)
    if not state.eligible:
        raise RuntimeError('benchmark requires exact empty CPU baseline: ' + state.reason)
    tree = tests.load_pinned_tree('_cleanup_benchmark_tree')
    vector = torch.tensor([0, 1, 2, 3, 4, 170], dtype=torch.int32, device='cpu')
    # Execute the actual small host validator, without importing Native/NPU.
    host_source = base.parent/'tree-s35-production-v101/host_target_metadata.py'
    before_hashes[str(host_source)] = sha(host_source)
    fn = next(node for node in ast.parse(host_source.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == '_cpu_values')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(host_source)+':actual-_cpu_values', 'exec'), namespace)
    cpu_values = namespace['_cpu_values']

    def plan_work():
        full = tree.cached_cpu_tree_speculation_plan(2, 4, 170, 4096, candidate_budget=5)
        packed = tree.pack_selected_tree_plan(full, [0, 1, 2, 3, 4])
        return int(packed.positions.numel())

    def values_work():
        return len(cpu_values(vector, 'HOST benchmark values'))

    def factories_work():
        return len(torch.tensor([1, 2, 3], dtype=torch.int32, device='cpu').tolist())

    modes = []
    def select_arm(name):
        if name == 'clean':
            if not observe_cpu_device_context(torch, state, redundant_cpu=False)['passed']:
                changed = restore_redundant_cpu_device_context(torch, state)
                if changed['status'] != 'cleaned':
                    raise RuntimeError('cannot select clean arm: ' + repr(changed))
                modes.append(changed)
        else:
            if not observe_cpu_device_context(torch, state, redundant_cpu=True)['passed']:
                changed = reinstall_redundant_cpu_device_context(torch, state)
                if changed['status'] != 'reinstalled':
                    raise RuntimeError('cannot select CPU mode arm: ' + repr(changed))
                modes.append(changed)

    findings = {}
    try:
        for label, work in [('actual_cached_tree_packing', plan_work),
                            ('actual_host_cpu_values', values_work),
                            ('explicit_cpu_tensor_and_tolist', factories_work)]:
            timings = {'clean': [], 'cpu_context': []}
            checksums = {}
            for repeat in range(args.repeats):
                for arm in (('clean', 'cpu_context') if repeat % 2 == 0 else ('cpu_context', 'clean')):
                    select_arm(arm)
                    for _ in range(20): work()
                    start = time.perf_counter()
                    checksum = 0
                    for _ in range(args.iterations): checksum += work()
                    timings[arm].append(time.perf_counter() - start)
                    checksums[arm] = checksum
            if checksums['clean'] != checksums['cpu_context']:
                raise AssertionError('benchmark outputs differ across device mode arms')
            profile_counts = {}
            for arm in ('clean', 'cpu_context'):
                select_arm(arm)
                profile = cProfile.Profile()
                profile.enable()
                for _ in range(100): work()
                profile.disable()
                profile_counts[arm] = sum(entry.callcount for entry in profile.getstats()
                    if hasattr(entry.code, 'co_filename')
                    and entry.code.co_filename == device_module.__file__
                    and entry.code.co_name == '__torch_function__')
            if profile_counts['clean'] != 0 or profile_counts['cpu_context'] <= 0:
                raise AssertionError('real DeviceContext callback control did not separate arms')
            clean = statistics.median(timings['clean'])
            redundant = statistics.median(timings['cpu_context'])
            findings[label] = {'iterations_per_sample': args.iterations, 'samples_seconds': timings,
                'median_seconds': {'clean': clean, 'cpu_context': redundant},
                'cpu_context_over_clean_ratio': redundant/clean,
                'outputs_equal': True, 'checksum': checksums['clean'],
                'DeviceContext_callbacks_per_100_work_calls': profile_counts}
    finally:
        select_arm('clean')
        sys.modules.pop('_cleanup_benchmark_tree', None)
    after_hashes = {path: sha(path) for path in before_hashes}
    if before_hashes != after_hashes:
        raise RuntimeError('source changed during HOST verification')
    if any(name == 'torch_npu' or name.startswith('torch_npu.') for name in sys.modules):
        raise RuntimeError('HOST task imported torch_npu')
    final = observe_cpu_device_context(torch, state, redundant_cpu=False)
    if not final['passed']:
        raise RuntimeError('final CPU baseline not restored')
    report = {'schema': 'strict-cpu-default-device-cleanup-HOST-v1', 'passed': True,
        'scope': 'CPU planning/state/dispatch microbenchmark only; no Native constructor or NPU/model execution',
        'argv': sys.argv, 'python_executable': sys.executable, 'python_version': platform.python_version(),
        'torch_version': torch.__version__, 'torch_file': torch.__file__,
        'affinity_before': original_affinity, 'affinity_during': sorted(os.sched_getaffinity(0)),
        'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
        'selected_environment': {key: os.environ.get(key) for key in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TORCH_DEVICE_BACKEND_AUTOLOAD',
             'PYTHONPATH', 'ASCEND_RT_VISIBLE_DEVICES')},
        'runtime_wrapper_used': False, 'NPU_started': False, 'formal_full_timing': False,
        'installed_into_frozen_source': False, 'NPU_or_performance_qualification': False,
        'source_hashes_before': before_hashes, 'source_hashes_after': after_hashes,
        'tests': {'run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
                  'seconds': tests_seconds, 'output': stream.getvalue()},
        'CPU_tree_plan_equivalence': tests.LAST_TREE_EQUIVALENCE,
        'benchmarks': findings, 'timer_excludes_mode_switch_and_observation': True,
        'mode_transitions': modes, 'final_state': final,
        'limits': ['CPU microbenchmark cannot predict end-to-end target or Goodput savings',
                   'existing user DeviceContext/non-CPU defaults/changed modes/dtype/APIs are preserved',
                   'get_default_device with CPU context may construct an empty CPU tensor',
                   'explicit finite CPU tree-plan domain, not all arbitrary TorchFunctionMode behavior']}
    with output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'passed': True, 'tests_run': result.testsRun, 'output': str(output),
        'sha256': sha(output), 'plan_records_per_arm': tests.LAST_TREE_EQUIVALENCE['before']['plan_records'],
        'ratios': {key: value['cpu_context_over_clean_ratio'] for key, value in findings.items()},
        'NPU_started': False}, ensure_ascii=False))


if __name__ == '__main__': main()
