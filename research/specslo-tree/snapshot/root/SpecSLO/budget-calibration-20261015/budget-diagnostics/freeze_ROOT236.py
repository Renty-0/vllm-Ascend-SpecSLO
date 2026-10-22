"""Freeze the ABI-only236 bootstrap and all unchanged235 dependencies."""
import ast, datetime, gzip, hashlib, json, subprocess, sys, tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
OLD = B/'tree-canonical-contiguous-full463-own-AR-v235'
P = B/'tree-canonical-trace-kwdefaults-full463-own-AR-v236'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


assert sha(OLD/'FREEZE235.json') == 'b9d62c5fc915a3fd9711e01d2f3e2e6894dffe96b28d791628a92c737cab69ca'
cpu = json.loads((P/'HOST_DEFAULTS236_RESULT.json').read_text())
assert cpu['status'] == 'HOST_original_warm_default8_and_clone_ABI_passed'
assert cpu['actual_original_warm_loop_executed'] and cpu['original_default_max_attempts'] == 8
assert not cpu['real_backend_import_attempts'] and not cpu['real_model_or_device_operations']
assert all(v['after'] == v['original'] and v['original_code_identity'] for v in cpu['bootstrap']['functions'])
prior = json.loads((D/'ROOT_V235_TERMINAL_MATH159_PASS_TRACE_ABI_FAILURE.json').read_text())
assert prior['current_independent_math_cases_per_target'] == 159 and prior['source_prepost_and_owned_cleanup_passed']
assert all(sha(path) == digest for path, digest in prior['proofs_sha256'].items())
assert sha(prior['receipt']['path']) == prior['receipt']['sha256']
files = json.loads((OLD/'FREEZE235.json').read_text())['files']
assert all(sha(path) == digest for path, digest in files.items())
files[str(OLD/'FREEZE235.json')] = sha(OLD/'FREEZE235.json')
for path in [*P.iterdir(), Path(__file__), D/'ROOT_V235_TERMINAL_MATH159_PASS_TRACE_ABI_FAILURE.json']:
    if path.is_file() and path.suffix in ('.py', '.sh', '.json'):
        if path.suffix == '.py':
            ast.parse(path.read_text())
        files[str(path)] = sha(path)
for path in P.glob('*.sh'):
    subprocess.run(['bash', '-n', str(path)], check=True)
save(P/'FREEZE236.json', dict(schema='ROOT236.original-keyword-defaults-canonical-full.v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=files,
    delta='original235 code unchanged; restore original warm max_attempts8 and optional checkpointNone on private trace FunctionType clones before each worker starts',
    scope='same fresh original startup/components159/warm/twozero/seal/full463/ownAR; no inherited model qualification'))
files = dict(files)
files[str(P/'FREEZE236.json')] = sha(P/'FREEZE236.json')
with (P/'SOURCE236.tar.gz').open('xb') as raw:
    with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for path in sorted(files):
                archive.add(path, arcname=path.lstrip('/'), recursive=False)
sys.path[:0] = [str(P)]
import guard236
guard236.EXPECTED_FREEZE = sha(P/'FREEZE236.json')
guard236.EXPECTED_ARCHIVE = sha(P/'SOURCE236.tar.gz')
guard236.prepared()
guard236.execution_namespace()
record = dict(status='ABI_source_frozen_HOST_original_warm_executed_device_unrun', directory=str(P),
              freeze_sha256=guard236.EXPECTED_FREEZE, archive_sha256=guard236.EXPECTED_ARCHIVE,
              source_members=len(files), original235_code_unchanged=True,
              restored_original_keyword_defaults=True, original_startup_and_same_instance159_required=True,
              full463_and_independent_own_AR_required=True, NPU_qualified=False,
              whole_workload_AR_qualified=False, Goodput=None, adopted=False)
save(D/'ROOT_V236_SOURCE_PREPARED.json', record)
print(json.dumps(record))
