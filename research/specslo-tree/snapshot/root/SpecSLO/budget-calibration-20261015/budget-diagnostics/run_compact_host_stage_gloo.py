"""Actual CPU Gloo microcontrol; no Native/NPU/model or runtime installation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

P = Path(__file__).resolve().parent
TAG = 'CPU_GLOO_STAGE_TRANSPORT_20261016_v1'
OUT = P / TAG


def worker(rank, rendezvous, cores):
    os.sched_setaffinity(0, {cores[rank % len(cores)]})
    import torch
    import torch.distributed as dist
    from compact_host_stage_transport import CompactHostStageTransport
    assert 'torch_npu' not in sys.modules
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=4,
        init_method='file://'+rendezvous, timeout=datetime.timedelta(seconds=90))
    compact = CompactHostStageTransport(torch, dist, dist.group.WORLD)

    def original(row):
        result = [None]*4
        dist.all_gather_object(result, row)
        return result

    def row(sequence):
        return dict(rank=rank, sequence=sequence, stage='live-prefix-write-declaration', error=None)

    checks = []
    timings = []
    try:
        for case in range(9):
            own = row(17)
            if case == 1 and rank == 2: own['error'] = 'ValueError: expected counterexample'
            if case == 2 and rank == 1: own['stage'] = 'different-stage'
            if case == 3 and rank == 3: own['sequence'] = 18
            if case == 4 and rank == 3: own['rank'] = 2
            if case == 5 and rank == 1: own['stage'] = 'x'*65
            if case == 6 and rank == 0: own['stage'] = ('original', 'stage')
            if case == 7 and rank == 2: own['sequence'] = 2**80
            if case == 8 and rank == 1: own['stage'] = 'bad\ud800'
            expected, actual = original(own), compact.collect(own)
            assert actual == expected, (case, actual, expected)
            checks.append(dict(case=case, exact_original_rows=True))
        for mode in ('no_default_device_context', 'explicit_CPU_device_context'):
            torch.set_default_device(None if mode.startswith('no_') else 'cpu')
            for _ in range(10): original(row(1)); compact.collect(row(1))
            for repeat in range(5):
                methods = ('original', 'compact') if repeat % 2 == 0 else ('compact', 'original')
                for method in methods:
                    dist.barrier()
                    started = time.perf_counter()
                    for sequence in range(50):
                        last = original(row(sequence)) if method == 'original' else compact.collect(row(sequence))
                    seconds = time.perf_counter()-started
                    assert last == [dict(rank=r, sequence=49, stage='live-prefix-write-declaration', error=None)
                                    for r in range(4)]
                    timings.append(dict(mode=mode, repeat=repeat, method=method,
                        iterations=50, whole_collect_seconds=seconds, per_call_ms=seconds*20.))
            torch.set_default_device(None)
        report = dict(rank=rank, CPU_affinity=sorted(os.sched_getaffinity(0)),
            exact_row_checks=checks, timings=timings, compact_calls=compact.compact_calls,
            object_fallbacks=compact.object_fallbacks, Torch_version=torch.__version__,
            torch_npu_imported='torch_npu' in sys.modules, NPU_initialized=False,
            model_initialized=False, production_installed=False)
        with (OUT/f'rank{rank}.json').open('x') as stream: stream.write(json.dumps(report, indent=2)+'\n')
    finally:
        torch.set_default_device(None)
        dist.destroy_process_group()


def main():
    assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0'
    assert sys.version_info[:2] == (3, 11)
    OUT.mkdir(exist_ok=False)
    paths = [Path(__file__), P/'compact_host_stage_codec.py', P/'compact_host_stage_transport.py',
             P/'test_compact_host_stage_codec.py']
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    before = {str(f): sha(f) for f in paths}
    import unittest
    from test_compact_host_stage_codec import CodecTest
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(CodecTest))
    assert result.wasSuccessful() and result.testsRun == 6
    import torch.multiprocessing as mp
    cores = sorted(os.sched_getaffinity(0))[:4]
    start = time.perf_counter()
    mp.spawn(worker, args=(str(OUT/'rendezvous'), cores), nprocs=4, join=True)
    reports = [json.loads((OUT/f'rank{r}.json').read_text()) for r in range(4)]
    critical = []
    for mode in ('no_default_device_context', 'explicit_CPU_device_context'):
        samples = {}
        for method in ('original', 'compact'):
            samples[method] = [max(next(x['per_call_ms'] for x in r['timings']
                if x['mode']==mode and x['method']==method and x['repeat']==repeat)
                for r in reports) for repeat in range(5)]
        critical.append(dict(mode=mode, max_rank_samples_ms=samples,
            median_ms={method:statistics.median(values) for method,values in samples.items()}))
    after = {str(f): sha(f) for f in paths}
    assert before == after and not any(r['torch_npu_imported'] for r in reports)
    report = dict(schema='ROOT.compact-HOST-stage-actual-CPU-Gloo-microcontrol.v1',passed=True,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_HOST_argv=sys.argv,
        source_hashes_before=before,source_hashes_after=after,HOST_tests_run=6,
        actual_CPU_Gloo_ranks=4,exact_original_row_counterexamples_per_rank=9,
        actual_microbenchmark=critical,elapsed_seconds=time.perf_counter()-start,
        reports={str(OUT/f'rank{r}.json'):sha(OUT/f'rank{r}.json') for r in range(4)},
        NPU_started=False,model_initialized=False,production_installed=False,
        actual_Native_boundary_installed=False,whole_workload_gain_proved=False,formal_Goodput=False)
    with (P/(TAG+'.json')).open('x') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,microbenchmark=critical,elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__': main()
