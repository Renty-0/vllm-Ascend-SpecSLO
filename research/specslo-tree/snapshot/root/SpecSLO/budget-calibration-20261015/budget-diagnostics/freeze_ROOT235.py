"""Freeze the complete local full235 experiment before any device launch."""
import ast, datetime, gzip, hashlib, json, subprocess, sys, tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
OLD = B/'tree-contiguous-branch-tail-final-metadata-math-v234'
P = B/'tree-canonical-contiguous-full463-own-AR-v235'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


cpu = json.loads((P/'CPU_BRANCH_TAIL235_RESULT.json').read_text())
assert cpu['status'] == 'CPU_exact_visible_KV_order_and_Native_ABI_passed' and cpu['cases'] == 207
assert cpu['branch_cases'] == 168 and cpu['max_actual_copy_slots'] <= 1792 and not cpu['NPU_used']
for path, digest in cpu['files_sha256'].items():
    assert sha(path) == digest
meta = json.loads((P/'CPU_METADATA_STATE231_RESULT.json').read_text())
assert meta['post_Native_padding_metadata_identity_checked'] and not meta['NPU_used']
live = json.loads((P/'HOST_FULL_LIVE235_RESULT.json').read_text())
assert live['status'] == 'HOST_original_runtime_and_full_LIVE_generation_inverse_passed'
assert live['generation']['effective_target_requests'] == 16 and live['generation']['effective_verification_candidates'] == 64
domain = json.loads((P/'HOST_KEY_CLOSURE235_RESULT.json').read_text())
assert domain['cases'] == 159 and domain['geometries'] == 2436 and domain['key_count'] == 17
for path, digest in domain['sources_sha256'].items():
    assert sha(path) == digest
terminal = json.loads((D/'ROOT_V234_TERMINAL_INDEPENDENT_BRANCH_MATH_AND_HANDOFF.json').read_text())
assert terminal['cases_per_target'] == 155 and terminal['all48_independent_prefixes_protected']
audit = json.loads((D/'ROOT_V234_INDEPENDENT_BRANCH_MATH_AUDIT.json').read_text())
assert audit['status'] == 'bounded_math_completed' and audit['positive_cases_per_target_rank'] == [155]*3
for path, digest in audit['source_sha256'].items():
    assert sha(path) == digest
files = json.loads((OLD/'FREEZE234.json').read_text())['files']
assert all(sha(path) == digest for path, digest in files.items())
files[str(OLD/'FREEZE234.json')] = sha(OLD/'FREEZE234.json')
extras = [Path(__file__), D/'ROOT_V234_TERMINAL_INDEPENDENT_BRANCH_MATH_AND_HANDOFF.json',
          D/'ROOT_V234_INDEPENDENT_BRANCH_MATH_AUDIT.json',
          D/'audit_ROOT234_independent_branch_math_HOST.py',
          D/'ROOT231_EFFECTIVE_C64_M80_KEY_DOMAIN_HOST.json',
          D/'enumerate_ROOT231_effective_key_domain_HOST.py']
for path in [*P.iterdir(), *extras]:
    if path.is_file() and path.suffix in ('.py', '.json', '.sh'):
        if path.suffix == '.py':
            ast.parse(path.read_text())
        files[str(path)] = sha(path)
for path in P.glob('*.sh'):
    subprocess.run(['bash', '-n', str(path)], check=True)
sys.path[:0] = [str(P), str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),
               str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),
               str(B/'tree-high-normal-cap4-native-families-v148')]
from runtime235 import project_source
projected, changes = project_source((B/'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text())
compile(projected, 'ROOT235.prelaunch.runtime-projection', 'exec')
assert len(changes) == 11
save(P/'FREEZE235.json', dict(schema='ROOT235.canonical-contiguous-full463-own-AR.v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=files,
    scope='fresh original201/44/114 and all218 components; same-instance159 canonical M80 math; effective N16/C64; original warm/twozero/seal/full463; own-output FULL9 AR after clocks'))
files = dict(files)
files[str(P/'FREEZE235.json')] = sha(P/'FREEZE235.json')
with (P/'SOURCE235.tar.gz').open('xb') as raw:
    with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for path in sorted(files):
                archive.add(path, arcname=path.lstrip('/'), recursive=False)
import guard235
guard235.EXPECTED_FREEZE = sha(P/'FREEZE235.json')
guard235.EXPECTED_ARCHIVE = sha(P/'SOURCE235.tar.gz')
guard235.prepared()
guard235.execution_namespace()
result = dict(status='source_frozen_CPU207_and_live_inverse_verified_device_unrun', directory=str(P),
    freeze_sha256=guard235.EXPECTED_FREEZE, archive_sha256=guard235.EXPECTED_ARCHIVE,
    source_members=len(files), same_instance_math_cases_planned=159,
    original_startup_and_components_required=True, original_sampler_unchanged=True,
    full463_and_independent_own_AR_required=True, whole_workload_AR_qualified=False,
    formal_Goodput=False, adopted=False)
save(D/'ROOT_V235_SOURCE_PREPARED.json', result)
print(json.dumps(result))
