"""Root-only single baseline command; no model or test is run by preparing this file."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
WORK = HERE / 'baseline-full463-campaign-v1'
REBIND = HERE / 'baseline-physical-rebind-v1'
PHYSICAL = (2, 3, 4, 6)
EXPECTED = {
    str(WORK / 'prepared.json'): '3d677a844d7fc6221dd6b60581761278121e2758534992dace278b96febca207',
    str(WORK / 'run-plan.json'): 'be68b24a653b576e54edf60e9b7b6020a22f9fa2cb52a66af6ecbf8de45483c9',
    str(WORK / 'predeclared-source-and-result-ledger.json'):
        'bfe987747a1ee821cb2bdff5b19c26eeb6b84d3303c911ad0b47eaec0fe65f8e',
    str(WORK / 'input-lock.json'): '815091f1f6d3f7064543343c9d19081ff90355bbc4f3c43df96eba565eb7e195',
    str(REBIND / 'input-lock.json'): '06adbce426fcd566fb47aa02754d173e378cc7dfd1fc180e8b8f59229813b325',
    str(HERE / 'actual-container-device-mapping.json'):
        'dcaaf4a145100a14cd06a280ba7db74070e26d208e9a80b92f04fe98c9d0381f',
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def fresh(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'Refuse existing output: ' + str(path))


def hashes(names):
    result, errors = {}, []
    for name in names:
        try:
            result[name] = sha(name)
        except Exception as error:
            result[name] = None
            errors.append({'path': name, 'error': repr(error)})
    return result, errors


def capture(argv, timeout=30):
    """Read-only system command; even a timeout never sends a signal."""
    started = time.monotonic()
    row = {'argv': argv, 'started_UTC': now(), 'timeout_seconds': timeout}
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        row['pid'] = proc.pid
        try:
            output, _ = proc.communicate(timeout=timeout)
            row.update(exit_code=proc.returncode, output=output, timed_out=False)
        except subprocess.TimeoutExpired as error:
            output = error.output or ''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            row.update(exit_code=proc.poll(), output=output, timed_out=True, process_left_running=True)
    except Exception as error:
        row.update(exit_code=None, output='', error=repr(error), timed_out=False)
    row['elapsed_seconds'] = time.monotonic() - started
    return row


def device_state():
    info = capture(['npu-smi', 'info'])
    rows = [(int(device), name) for device, name in re.findall(
        r'^\|\s*(\d+)\s+(\S+)\s*\|', info['output'], re.MULTILINE) if not name.isdecimal()]
    hardware = info.get('exit_code') == 0 and not info.get('timed_out') and sorted(rows) == [
        (physical, '910B2') for physical in PHYSICAL]
    processes = {}
    for physical in PHYSICAL:
        row = capture(['npu-smi', 'info', '-t', 'proc-mem', '-i', str(physical), '-c', '0'])
        row['idle'] = bool(row.get('exit_code') == 0 and not row.get('timed_out') and re.search(
            r'No process in device\.|No running processes found', row['output'], re.IGNORECASE))
        processes[str(physical)] = row
    return {'hardware': info, 'parsed_hardware': rows, 'four_910B2': hardware,
            'processes': processes, 'all_idle': hardware and all(row['idle'] for row in processes.values())}


def owned_group(pgid):
    if pgid is None:
        return []
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            data = (directory / 'stat').read_text()
            fields = data[data.rindex(')') + 2:].split()
            # Original inner timeout can form a new process group inside our owned session.
            if (int(fields[2]) == pgid or int(fields[3]) == pgid) and fields[0] != 'Z':
                found.append({'pid': int(directory.name), 'state': fields[0], 'ppid': int(fields[1]),
                              'pgid': int(fields[2]), 'session_id': int(fields[3])})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(found, key=lambda row: row['pid'])


def artifact(path):
    path = Path(path)
    result = {'path': str(path), 'exists': path.exists(), 'is_symlink': path.is_symlink(),
              'is_file': path.is_file()}
    try:
        if path.is_file() and not path.is_symlink():
            result.update(bytes=path.stat().st_size, sha256=sha(path))
    except Exception as error:
        result['hash_error'] = repr(error)
    return result


def logged(argv, cwd, timeout, path, start_callback=None):
    fresh(path)
    row = {'argv': argv, 'cwd': cwd, 'timeout_seconds': timeout, 'log_path': str(path), 'started_UTC': now()}
    started = time.monotonic()
    with Path(path).open('x') as log:
        try:
            proc = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row.update(pid=proc.pid, pgid=proc.pid)
            if start_callback is not None:
                start_callback(row)
            try:
                proc.wait(timeout=timeout)
                row.update(exit_code=proc.returncode, timed_out=False, process_finished=True)
            except subprocess.TimeoutExpired:
                row.update(exit_code=proc.poll(), timed_out=True, process_finished=False,
                           process_left_running=True, supervisor_sent_signal=False)
            except BaseException:
                row.update(exit_code=proc.poll(), process_finished=proc.poll() is not None,
                           interrupted=True, error=traceback.format_exc(), supervisor_sent_signal=False)
        except BaseException:
            row.update(exit_code=None, process_finished=False, error=traceback.format_exc())
    row.update(completed_UTC=now(), elapsed_seconds=time.monotonic() - started,
               log=artifact(path), log_is_final=bool(row.get('process_finished')))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'tp4', 'nano'))
    mode = parser.parse_args().mode
    require(WORK.is_dir() and WORK.resolve() == WORK, 'Campaign directory must be original real path')
    stem = 'preflight' if mode == 'preflight' else mode + '-calibration-r1'
    receipt_path, log_path = WORK / (stem + '.root-launch.json'), WORK / (stem + '.root.log')
    for path in (receipt_path, log_path, *(WORK / f'{stem}.post-verifier-{n}.log' for n in range(2))):
        fresh(path)
    receipt = {'schema': 'specslo-baseline-full463-root-single-command-v1', 'status': 'running', 'mode': mode,
               'created_UTC': now(), 'executor': str(Path(__file__).resolve()), 'executor_sha256': sha(__file__),
               'expected_documents_sha256': EXPECTED, 'child': None, 'errors': [],
               'Goodput_or_full463_qualified': False, 'source_tests_model_executed_by_preparer': False,
               'supervisor_sends_signals_or_retries': False,
               'owned_process_scope': 'Child process group and full child session, including inner timeout groups.',
               'scope': 'Execution/identity/cleanup only; original gates and full463 output/Goodput need separate result audit.'}
    # Exclusive reservation protects all prior execution evidence. Updates touch only this owned file descriptor.
    stream = receipt_path.open('x')
    def save():
        stream.seek(0)
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    save()
    pins, rebind_pins, plan, selected = {}, {}, None, None
    started = time.monotonic()
    try:
        before, errors = hashes(EXPECTED)
        receipt.update(documents_sha256_before=before, document_hash_errors_before=errors)
        require(not errors and before == EXPECTED, 'Sealed campaign/prepared/plan/ledger/mapping changed')
        pins = read(WORK / 'input-lock.json')['files_sha256']
        rebind_pins = read(REBIND / 'input-lock.json')['files_sha256']
        require(len(pins) == 70 and len(rebind_pins) == 59 and all(pins.get(k) == v for k, v in rebind_pins.items()),
                'Expected70 pins and exact original59 subset required')
        before, errors = hashes(pins)
        receipt.update(files_sha256_before=before, file_hash_errors_before=errors)
        require(not errors and before == pins, 'Campaign/source inputs changed before launch')
        mapping = read(HERE / 'actual-container-device-mapping.json')
        expected_devices = [{'user_device_id': logical, 'container_logical_device_id': logical,
                             'physical_device_id': physical} for logical, physical in enumerate(PHYSICAL)]
        require(mapping['devices'] == expected_devices and mapping['device_count'] == 4
                and mapping['matches_requested_new_physical_assignment'] is True, 'Actual root ACL mapping changed')
        receipt['actual_mapping_receipt'] = mapping
        plan = read(WORK / 'run-plan.json')
        selected = plan['preflight'] if mode == 'preflight' else next(row for row in plan['runs'] if row['role'] == mode)
        require(selected['cwd'] == str(WORK), 'Unexpected campaign cwd')
        if mode != 'preflight':
            require(selected['root_receipt_path'] == str(receipt_path) and selected['root_log_path'] == str(log_path),
                    'Execution output paths differ from sealed plan')
            previous_preflight = WORK / 'preflight.root-launch.json'
            prior = read(previous_preflight)
            require(prior['mode'] == 'preflight' and prior['status'] == 'completed'
                    and prior['execution_checks_passed'] is True and prior['child']['exit_code'] == 0
                    and prior['documents_sha256_after'] == EXPECTED and prior['files_sha256_after'] == pins
                    and prior['executor_sha256'] == receipt['executor_sha256'], 'Fresh completed preflight required')
            receipt['fresh_preflight'] = artifact(previous_preflight)
            if mode == 'tp4':
                fresh(plan['result_root'])
            else:
                tp4 = next(row for row in plan['runs'] if row['role'] == 'tp4')
                previous = read(tp4['root_receipt_path'])
                require(previous['mode'] == 'tp4' and previous['status'] == 'completed'
                        and previous['child']['process_finished'] is True
                        and previous['child']['exit_code'] is not None
                        and previous['all_pins_unchanged'] and previous['all_documents_unchanged']
                        and previous['documents_sha256_after'] == EXPECTED and previous['files_sha256_after'] == pins
                        and previous['executor_sha256'] == receipt['executor_sha256']
                        and previous['original_post_verifiers_passed'], 'Prior TP4 completion/identity required')
                require(owned_group(previous['child'].get('pgid')) == [], 'Previous TP4 process group remains')
                receipt['previous_TP4_root_receipt'] = artifact(tp4['root_receipt_path'])
                receipt['previous_TP4_GP_not_required'] = True
                receipt['previous_cleanup_rechecked_now'] = True
            for path in selected['fresh_paths']:
                fresh(path)
            receipt['devices_before'] = device_state()
            require(receipt['devices_before']['all_idle'], 'Exactly four idle910B2 devices required before model')
        receipt.update(argv=selected['argv'], cwd=selected['cwd'], outer_timeout_seconds=selected['outer_timeout_seconds'],
                       entry_identity={'path': selected['argv'][1], 'sha256': sha(selected['argv'][1])})
        save()
        def started_child(row):
            receipt['child'] = dict(row)
            save()
        receipt['child'] = logged(selected['argv'], selected['cwd'], selected['outer_timeout_seconds'], log_path,
                                  start_callback=started_child)
        require(receipt['child'].get('process_finished') is True
                and receipt['child'].get('exit_code') == selected['required_exit'],
                'Original command failed/interrupted/timed out; all evidence retained')
    except BaseException:
        receipt['errors'].append({'phase': 'precheck_or_child', 'traceback': traceback.format_exc()})
    finally:
        # Rehash BEFORE invoking original verifiers; changed executable evidence is never executed.
        after, errors = hashes(pins)
        documents, doc_errors = hashes(EXPECTED)
        receipt.update(files_sha256_after=after, file_hash_errors_after=errors,
                       documents_sha256_after=documents, document_hash_errors_after=doc_errors,
                       all_pins_unchanged=bool(pins) and not errors and after == pins == receipt.get('files_sha256_before'),
                       all_documents_unchanged=not doc_errors and documents == EXPECTED == receipt.get('documents_sha256_before'))
        receipt['post_verifiers'] = []
        if receipt['all_pins_unchanged'] and receipt['all_documents_unchanged'] and plan is not None:
            post = plan['runs'][0] if selected is None or mode == 'preflight' else selected
            for index, argv in enumerate(post['post_verifiers']):
                try:
                    row = logged(argv, str(WORK), post['post_verifier_timeout_seconds_each'],
                                 WORK / f'{stem}.post-verifier-{index}.log')
                except BaseException:
                    row = {'argv': argv, 'exit_code': None, 'process_finished': False,
                           'error': traceback.format_exc()}
                receipt['post_verifiers'].append(row)
        else:
            receipt['errors'].append({'phase': 'post_verifiers', 'reason': 'Unsafe changed/unverified input; not executed'})
        receipt['original_post_verifiers_passed'] = len(receipt['post_verifiers']) == 2 and all(
            row.get('process_finished') is True and row.get('exit_code') == 0 for row in receipt['post_verifiers'])
        try:
            receipt['cleanup'] = device_state()
            receipt['owned_group_remaining'] = owned_group((receipt.get('child') or {}).get('pgid'))
        except BaseException:
            receipt['cleanup'] = {'all_idle': False, 'error': traceback.format_exc()}
            receipt['owned_group_remaining'] = None
        final, final_errors = hashes(pins)
        docs_final, docs_errors = hashes(EXPECTED)
        receipt.update(files_sha256_after=final, file_hash_errors_after=final_errors,
                       documents_sha256_after=docs_final, document_hash_errors_after=docs_errors)
        receipt['all_pins_unchanged'] = receipt['all_pins_unchanged'] and not final_errors and final == pins
        receipt['all_documents_unchanged'] = receipt['all_documents_unchanged'] and not docs_errors and docs_final == EXPECTED
        executor_after, executor_errors = hashes([str(Path(__file__).resolve())])
        receipt['executor_unchanged'] = not executor_errors and executor_after[str(Path(__file__).resolve())] == receipt['executor_sha256']
        receipt['artifacts'] = [artifact(path) for path in (selected or {}).get('required_actual_artifacts', [])]
        receipt['expected_artifacts_present'] = all(row['is_file'] and not row['is_symlink'] and row.get('sha256')
                                                    for row in receipt['artifacts'])
        child = receipt.get('child') or {}
        receipt['completed_UTC'] = now()
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['status'] = 'completed' if child.get('process_finished') is True else 'failed_or_incomplete'
        receipt['execution_checks_passed'] = bool(not receipt['errors'] and child.get('exit_code') == 0
            and child.get('process_finished') is True and receipt['all_pins_unchanged']
            and receipt['all_documents_unchanged'] and receipt['executor_unchanged']
            and receipt['original_post_verifiers_passed'] and receipt['expected_artifacts_present']
            and (mode == 'preflight' or receipt['cleanup']['all_idle']) and receipt['owned_group_remaining'] == [])
        receipt['blocks_next_model'] = not (receipt['cleanup']['all_idle']
            and receipt['owned_group_remaining'] == [] and child.get('process_finished') is True
            and receipt['all_pins_unchanged'] and receipt['all_documents_unchanged']
            and receipt['original_post_verifiers_passed'])
        save()
        stream.close()
        receipt_path.chmod(0o444)
    print(json.dumps({'mode': mode, 'receipt': str(receipt_path), 'status': receipt['status'],
                      'execution_checks_passed': receipt['execution_checks_passed'],
                      'child_exit_code': (receipt.get('child') or {}).get('exit_code'),
                      'Goodput_or_full463_qualified': False}), flush=True)
    return 0 if receipt['execution_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
