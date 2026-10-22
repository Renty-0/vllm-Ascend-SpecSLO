"""Synthetic HOST-only build cost; never a model, full-footprint or GP result."""
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import statistics
import sys
import time


BASE = Path(__file__).resolve().parent.parent
CANDIDATE = BASE/'tree-live-prefix-page-protection-v85'
PARENT = BASE/'tree-live-poisson-policy-counter-v84'
EXPECTED = {
    CANDIDATE/'live_native_boundaries.py': '3656b1b4708c367557440adb699560c24af27d79977b0f39e21f07a7c4357d21',
    PARENT/'live_native_boundaries.py': '854da435a5cb6ad1707ddbfe090c6dd373d559ad8344c9b4452e3fa8a273a4ec',
}


class PureHostImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
            raise RuntimeError('Pure HOST micro forbids device/model package imports: '+fullname)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_original(prefix_lengths, current):
    protected = set()
    for owner, length in prefix_lengths.items():
        pages = current[owner]
        if length > len(pages)*128:
            raise ValueError('Original protected prefix capacity rejected')
        protected.update(pages[position//128]*128+position%128 for position in range(length))
    return protected


def main():
    if sys.version_info[:2] != (3, 11) or not sys.dont_write_bytecode:
        raise RuntimeError('Retained Python3.11 -B is required')
    if any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') for name in sys.modules):
        raise RuntimeError('Benchmark process already imported a model/device package')
    if any(sha(path) != wanted for path, wanted in EXPECTED.items()):
        raise RuntimeError('Pinned old/new protection implementation changed')
    sys.meta_path.insert(0, PureHostImports())
    sys.path.insert(0, str(CANDIDATE))
    from live_native_boundaries import _protected_page_limits
    rows = []
    for count, length in ((1, 443), (16, 443), (16, 2301), (16, 4095)):
        lengths = dict.fromkeys(range(count), length)
        allocated_pages = (length+127)//128+1
        current = {owner: tuple(1+owner*allocated_pages+j for j in range(allocated_pages))
                   for owner in range(count)}
        dense = dense_original(lengths, current)
        limits = _protected_page_limits(lengths, current)
        # Includes the unprotected suffix and every allocated unused page.
        for pages in current.values():
            for page in pages:
                for offset in range(128):
                    if ((page*128+offset in dense) != (offset < limits.get(page, 0))):
                        raise RuntimeError('Dense/page protection membership differs')
        functions = {'old_dense': lambda:dense_original(lengths, current),
                     'new_page_limits': lambda:_protected_page_limits(lengths, current)}
        for fn in functions.values():
            fn()
        iterations = max(10, min(500, 1000000//(count*length)))
        samples = {name: [] for name in functions}
        for repeat in range(9):
            names = list(functions)
            if repeat % 2:
                names.reverse()
            for name in names:
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    functions[name]()
                samples[name].append((time.perf_counter_ns()-started)/iterations/1000.)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        rows.append(dict(requests=count, prefix_tokens_each=length,
            old_dense_slot_entries=len(dense), new_protected_page_entries=len(limits),
            exact_membership_match=True, iterations_per_sample=iterations,
            actual_samples_us=samples, median_build_us=medians,
            helper_build_speedup=medians['old_dense']/medians['new_page_limits']))
    if any(sha(path) != wanted for path, wanted in EXPECTED.items()):
        raise RuntimeError('Protection source changed during HOST measurement')
    report = dict(schema='V85.synthetic-HOST-protection-build-micro.v1',
        created_UTC=datetime.now(timezone.utc).isoformat(), sys_executable=sys.executable,
        Python=sys.version, source_sha256={str(path):value for path, value in EXPECTED.items()},
        scope='Synthetic prefix protection data-structure construction only',
        rows=rows, source_inputs_unchanged=True, NPU_initialized=False,
        model_or_device_packages_imported=False, full_footprint_cost_measured=False,
        actual_NPU_cost_measured=False, formal_Goodput=False)
    output = BASE/'budget-diagnostics/V85_PREFIX_PROTECTION_HOST_MICRO_ATTEMPT1.json'
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'rows': [
        {key:value for key,value in row.items() if key != 'actual_samples_us'} for row in rows]}, indent=2))


if __name__ == '__main__':
    main()
