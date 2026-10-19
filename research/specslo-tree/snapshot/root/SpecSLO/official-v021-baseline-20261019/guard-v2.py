"""Three explicitly requested, serial, source-bound old official MC2 runs."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path('/root/SpecSLO/official-v021-baseline-20261019')
BASE = Path('/root/SpecSLO/budget-calibration-20261015')
sys.path.insert(0, str(BASE / 'budget-diagnostics'))
from final_owned_processes import Owned, snapshot
from final_baseline_cell import native_verifier


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def execute(expected_manifest_sha):
    manifest_path = ROOT / 'SOURCE_MANIFEST-v2.json'
    require(sha(manifest_path) == expected_manifest_sha, 'Exact prepared source manifest required')
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads((ROOT / 'RECOVERY_PLAN.json').read_text())
    require(plan['repeat_runs'] == 3 and plan['MC2_ON'] is True, 'Exactly three real MC2_ON runs')

    def sources():
        for path, wanted in manifest['files'].items():
            require(sha(path) == wanted, 'Prepared official source changed: ' + path)

    sources()
    lock = (BASE / '.ROOT_NPU_EXECUTION.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from retained_paths import restore
    runtime_aliases = restore()
    controls = ROOT / 'controls-v2'
    controls.mkdir(exist_ok=False)
    record = dict(runtime_aliases=runtime_aliases, status='running', started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  schema='old-official-v021-MC2-3run-guard.v1', physical_cards=[2, 5, 6, 7],
                  source_manifest_sha256=expected_manifest_sha, rounds=[], errors=[], stability=None)

    def save():
        temporary = controls / 'receipt.tmp'
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(controls / 'receipt.json')

    def interrupted(signum, frame):
        raise RuntimeError('Guard interrupted: ' + str(signum))

    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    helper_path = Path('/root/SpecSLO/resume-20261012/run_baseline_calibration-v1.py')
    spec = importlib.util.spec_from_file_location('_old_official_owned_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.PHYSICAL = (2, 5, 6, 7)
    save()
    try:
        for ordinal in range(1, 4):
            tag = f'official021-MC2ON-20261019-v2-r{ordinal}'
            directory = controls / tag
            directory.mkdir()
            row = dict(tag=tag, ordinal=ordinal, status='starting', errors=[])
            record['rounds'].append(row)
            child = None
            owned = None
            try:
                sources()
                row['devices_before'] = helper.device_state()
                require(row['devices_before']['all_idle'], 'All four requested cards must be idle')
                row['pre_verifier'] = native_verifier(directory, 'pre')
                argv = ['bash', str(ROOT / 'run-v2.sh'), tag]
                row['argv'] = argv
                with (directory / 'child.log').open('x') as log:
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    owned = Owned(child.pid, snapshot())
                    row.update(child_pid=child.pid, status='running')
                    start = time.monotonic()
                    save()
                    while child.poll() is None:
                        owned.update(snapshot())
                        row['elapsed_seconds'] = time.monotonic() - start
                        require(row['elapsed_seconds'] < 2100, 'Bounded 2100-second baseline round')
                        save()
                        time.sleep(2)
                    row['child_exit_code'] = child.wait()
                require(row['child_exit_code'] == 0, 'Original official-only baseline entry failed')
                completed = subprocess.run([sys.executable, '-B', str(ROOT / 'audit_result.py'), str(ROOT / 'runs' / tag)],
                                           capture_output=True, text=True, timeout=180)
                (directory / 'independent-audit.log').write_text(completed.stdout + completed.stderr)
                require(completed.returncode == 0, 'Independent original-workload, source, MC2 and timestamp audit')
                audit = json.loads((ROOT / 'runs' / tag / 'independent-audit.json').read_text())
                require(audit['passed'] and not audit['errors'], 'Complete independent baseline audit')
                row['metrics'] = {k: audit[k] for k in ('paper_Goodput', 'raw_throughput', 'attained_requests',
                                                       'attained_by_class', 'TPOT_median_ms_by_class', 'e2e_seconds')}
                row['actual_result_complete'] = True
            except BaseException:
                row['errors'].append(traceback.format_exc())
            finally:
                row['owned_cleanup'] = owned.cleanup() if owned is not None else dict(before=[], after=[])
                row['owned_remaining'] = row['owned_cleanup']['after']
                if child is not None:
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        row['errors'].append('Owned baseline child did not exit')
                try:
                    sources()
                    row['sources_unchanged'] = True
                    row['post_verifier'] = native_verifier(directory, 'post')
                    row['devices_after'] = helper.device_state()
                    require(not row['owned_remaining'] and row['devices_after']['all_idle'], 'Owned cleanup and four idle cards')
                except BaseException:
                    row['errors'].append(traceback.format_exc())
                row['status'] = 'complete' if not row['errors'] and row.get('actual_result_complete') else 'failed'
                save()
            require(row['status'] == 'complete', 'Stop after any failed round; retain its actual evidence')
        rows = [v['metrics'] for v in record['rounds']]
        spread = lambda values: (max(values) - min(values)) / statistics.median(values)
        goodput = [v['paper_Goodput'] for v in rows]
        raw = [v['raw_throughput'] for v in rows]
        per_class = {kind: spread([v['TPOT_median_ms_by_class'][kind] for v in rows]) for kind in ('tight', 'normal', 'loose')}
        criteria = plan['stability_predeclared']
        stable = (spread(raw) <= criteria['raw_full_range_over_median_max']
                  and spread(goodput) <= criteria['Goodput_full_range_over_median_max']
                  and all(v <= criteria['per_class_TPOT_median_full_range_over_median_max'] for v in per_class.values()))
        record['stability'] = dict(passed=stable, observed_fresh_runs=3, Goodput_all=goodput,
                                   Raw_all=raw, Goodput_median=statistics.median(goodput), Raw_median=statistics.median(raw),
                                   Goodput_range_over_median=spread(goodput), Raw_range_over_median=spread(raw),
                                   per_class_TPOT_median_range_over_median=per_class, criteria=criteria,
                                   representative_round=record['rounds'][goodput.index(statistics.median(goodput))]['tag'],
                                   scope='Three observations on this exact machine/configuration; not universal stability')
        record['status'] = 'complete'
    except BaseException:
        record['errors'].append(traceback.format_exc())
        record['status'] = 'failed'
    finally:
        record['finished_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(json.dumps({k: record[k] for k in ('status', 'errors', 'stability')}))
    return 0 if record['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.source_manifest_sha256))
