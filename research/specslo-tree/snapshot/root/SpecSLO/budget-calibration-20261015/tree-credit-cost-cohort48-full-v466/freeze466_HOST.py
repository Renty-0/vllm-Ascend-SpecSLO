"""Freeze a conditional candidate only after the current owned run ends."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import sys
import tarfile

P = Path(__file__).resolve().parent
B = P.parent
PRIOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-credit-cost-refill-current0127-root-v465-component-full463-ownAR-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with (B/'.ROOT_NPU_EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(PRIOR.read_text())
        if (previous['status'] not in ('complete', 'failed')
                or previous.get('sources_unchanged') is not True
                or previous.get('devices_after', {}).get('all_idle') is not True):
            raise RuntimeError('Full465 execution, source check and owned cleanup must finish first')
        line = next(s for s in (P/'startup466.sh').read_text().splitlines()
                    if s.startswith('export PYTHONPATH='))
        sys.path[:0] = line[len('export PYTHONPATH='):-len('${PYTHONPATH:-}')].split(':')
        from guard466 import execution_namespace
        from check466 import PARENT_FREEZE, PARENT_ARCHIVE
        helpers = execution_namespace()['original_helpers']()
        if helpers['helper']().device_state()['all_idle'] is not True:
            raise RuntimeError('Current0/1/2/7 devices must actually be idle')
        host = json.loads((P/'HOST466_RESULT.json').read_text())
        if (host['status'] != 'PASS_HOST466_COHORT48_GLOBAL64_AND_ACTUAL_TERMINAL_INTERFACES'
                or not host['actual_terminal_interfaces']['both_interfaces_pass']
                or host['NPU_used'] is not False or host['NN_permission'] is not False):
            raise RuntimeError('Actual source interface and policy checks required')
        for path, wanted in host['source_sha256'].items():
            if sha(path) != wanted:
                raise RuntimeError('Changed HOST-checked source: '+path)
        paths = [p for p in P.iterdir() if p.is_file() and (p.suffix in ('.py', '.sh')
                 or p.name in ('PLAN466.json', 'HOST466_RESULT.json'))]
        paths.append(PRIOR)
        files = {str(path):sha(path) for path in sorted(paths)}
        frozen = dict(schema='ROOT466.cohort48-full.freeze.v1',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=files,
            parent465_freeze=PARENT_FREEZE, parent465_archive=PARENT_ARCHIVE,
            physical_cards=[0,1,2,7], global_active_limit=64, maximum_new_cohort=48,
            maximum_empty_prefill_cohort=48, maximum_nonempty_prefill_cohort=32,
            predecessor_receipt_sha256=sha(PRIOR), original_model_callbacks_unchanged=True,
            terminal_audit_interfaces_corrected=True, NN_permission_inherited=False)
        path = P/'FREEZE466.json'
        with path.open('x') as stream:
            json.dump(frozen, stream, indent=2)
            stream.write('\n')
        members = dict(files)
        members[str(path)] = sha(path)
        archive = P/'SOURCE466.tar.gz'
        with tarfile.open(archive, 'x:gz') as stream:
            for name in sorted(members):
                stream.add(name, arcname=name.lstrip('/'), recursive=False)
        receipt = dict(status='HOST466_SOURCE_FROZEN_NO_NN_PERMISSION',
            source_freeze_SHA256=sha(path), source_archive_SHA256=sha(archive),
            own_source_members=len(members), NPU_model_launched=False, NN_permission_inherited=False)
        with (P/'SOURCE466_RECEIPT.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
