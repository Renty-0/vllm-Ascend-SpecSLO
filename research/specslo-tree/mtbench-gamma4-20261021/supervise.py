"""Own one isolated acceptance run and retain exit/source/card evidence."""
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARDS = (2, 5, 6, 7)
LIMIT_SECONDS = 7200


def devices():
    values = {}
    for card in CARDS:
        result = subprocess.run(['npu-smi', 'info', '-t', 'proc-mem', '-i', str(card), '-c', '0'],
                                capture_output=True, text=True, timeout=30)
        values[str(card)] = dict(exit_code=result.returncode, output=result.stdout + result.stderr,
                                 idle=result.returncode == 0 and 'No process in device.' in result.stdout)
    return values


def save(value):
    path = HERE / 'receipt.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    files = [HERE / name for name in ('run_mtbench.py', 'launch.sh', 'supervise.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    receipt = dict(status='starting', started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source_hashes=hashes, physical_cards=list(CARDS), model_pair=['Qwen3-0.6B', 'Qwen3-32B'])
    process = None
    with Path('/root/SpecSLO/budget-calibration-20261015/.ROOT_NPU_EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt['devices_before'] = devices()
        assert all(row['idle'] for row in receipt['devices_before'].values())
        started = time.monotonic()
        try:
            with (HERE / 'run.log').open('x') as log:
                process = subprocess.Popen(['/bin/bash', str(HERE / 'launch.sh')], stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                receipt.update(status='running', child_pid=process.pid)
                save(receipt)
                while process.poll() is None:
                    if time.monotonic() - started > LIMIT_SECONDS:
                        raise TimeoutError('MT-Bench acceptance diagnostic exceeded7200seconds')
                    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) > 512 * 1024**2:
                        raise RuntimeError('Private diagnostic output exceeded512MiB')
                    time.sleep(5)
                receipt['child_exit_code'] = process.returncode
                receipt['status'] = 'complete' if process.returncode == 0 else 'failed'
        except BaseException as error:
            receipt.update(status='failed', error=repr(error))
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        finally:
            receipt['elapsed_seconds'] = time.monotonic() - started
            receipt['sources_unchanged'] = all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha for name, sha in hashes.items())
            receipt['devices_after'] = devices()
            receipt['all4idle'] = all(row['idle'] for row in receipt['devices_after'].values())
            receipt['finished_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save(receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'sources_unchanged', 'all4idle', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
