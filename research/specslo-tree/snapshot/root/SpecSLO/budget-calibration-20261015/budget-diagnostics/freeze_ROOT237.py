"""Freeze observer-role237; frozen model235 and default bootstrap236 unchanged."""
import ast, datetime, gzip, hashlib, json, subprocess, sys, tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
OLD = B/'tree-canonical-trace-kwdefaults-full463-own-AR-v236'
P = B/'tree-canonical-role-safe-full463-own-AR-v237'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


assert sha(OLD/'FREEZE236.json') == '9a1f51560862fe8d36d648eb7ae43356564d73feb7208380475539b2912496e8'
cpu = json.loads((P/'HOST_ROLES237_RESULT.json').read_text())
assert cpu['status'] == 'HOST_actual235_observer_draft_None_protocol_and_target_records_passed'
assert cpu['actual_original_forward_called_once_per_call'] and cpu['target_result_object_identity_preserved']
assert cpu['target_observer_code_unchanged'] and cpu['no_fake_target_output_or_metadata'] and not cpu['NPU_used']
prior_path = D/'ROOT_V236_TERMINAL_MATH159_PASS_DRAFT_OBSERVER_FAILURE.json'
prior = json.loads(prior_path.read_text())
assert prior['math_cases_per_target'] == 159 and prior['all4_idle_and_source_prepost_passed']
for field in ('receipt', 'failure_record'):
    assert sha(prior[field]['path']) == prior[field]['sha256']
files = json.loads((OLD/'FREEZE236.json').read_text())['files']
assert all(sha(path) == digest for path,digest in files.items())
files[str(OLD/'FREEZE236.json')] = sha(OLD/'FREEZE236.json')
for path in [*P.iterdir(), Path(__file__), prior_path]:
    if path.is_file() and path.suffix in ('.py', '.sh', '.json'):
        if path.suffix == '.py':
            ast.parse(path.read_text())
        files[str(path)] = sha(path)
for path in P.glob('*.sh'):
    subprocess.run(['bash', '-n', str(path)], check=True)
save(P/'FREEZE237.json', dict(schema='ROOT237.original-draft-None-forward-observer.v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=files,
    delta='draft rank restores original forward callback captured by235 observer; target callback unchanged;236 keyword defaults retained',
    scope='all fresh original startup/components159/warm/twozero/seal/full463/ownAR; no inherited current-instance model qualification'))
files = dict(files)
files[str(P/'FREEZE237.json')] = sha(P/'FREEZE237.json')
with (P/'SOURCE237.tar.gz').open('xb') as raw:
    with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for path in sorted(files):
                archive.add(path, arcname=path.lstrip('/'), recursive=False)
sys.path[:0] = [str(P)]
import guard237
guard237.EXPECTED_FREEZE = sha(P/'FREEZE237.json')
guard237.EXPECTED_ARCHIVE = sha(P/'SOURCE237.tar.gz')
guard237.prepared()
guard237.execution_namespace()
record = dict(status='role_source_frozen_HOST_both_roles_passed_device_unrun', directory=str(P),
              freeze_sha256=guard237.EXPECTED_FREEZE, archive_sha256=guard237.EXPECTED_ARCHIVE,
              source_members=len(files), frozen235_model_and_target_observer_unchanged=True,
              original_draft_None_return_protocol_restored=True, original236_keyword_defaults_retained=True,
              original_startup_and_current159_required=True, full463_and_independent_own_AR_required=True,
              NPU_qualified=False, whole_workload_AR_qualified=False, Goodput=None, adopted=False)
save(D/'ROOT_V237_SOURCE_PREPARED.json', record)
print(json.dumps(record))
