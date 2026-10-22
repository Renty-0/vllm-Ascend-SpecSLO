"""Root's actual HOST/source/archive check before the V85 local diagnostic."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

BASE = Path('/root/SpecSLO/budget-calibration-20261015')
P = BASE/'tree-live-prefix-page-protection-v85'
OUTPUT = BASE/'budget-diagnostics/ROOT_V85_FINAL_CPU_AND_SOURCE_PIN_AUDIT.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    require(sys.version_info[:2] == (3, 11) and sys.dont_write_bytecode,
            'Retained Python3.11 -B is required')
    require(not OUTPUT.exists(), 'Root audit output already exists')
    sys.path.insert(0, str(P))
    import check_v75 as checker
    cpu = json.loads(checker.CPU.read_text())
    freeze = json.loads(checker.FREEZE.read_text())
    local, external = checker.candidate_files(), checker.external_CPU_files()
    checks = {}
    checks['actual_final_CPU'] = (
        cpu['status'] == 'passed' and cpu['phase'] == 'final-local-source-before-NPU'
        and cpu['modules'] == list(checker.CPU_MODULES) and len(cpu['modules']) == 29
        and cpu['tests_run'] >= 235 and cpu['failures'] == cpu['errors'] == 0
        and cpu['NPU_initialized'] is False
        and cpu['model_initialization_attempts'] == cpu['NPU_initialization_attempts'] == [])
    checks['exact_local_source_freeze'] = (
        local == cpu['input_hashes'] == cpu['input_hashes_after'] == freeze['files']
        and freeze['CPU_sha256'] == sha(checker.CPU)
        and freeze['online_qualified'] is False and freeze['local_numeric_qualified'] is False)
    checks['exact_external_CPU_closure'] = (
        external == cpu['external_input_hashes'] == cpu['external_input_hashes_after'])
    archive_path = Path(cpu['actual_local_source_archive'])
    require(archive_path.is_relative_to(P/'host-checks'), 'CPU archive outside candidate')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        wanted = {Path(name).name: value for name, value in local.items()}
        checks['actual_CPU_source_archive'] = (
            sha(archive_path) == cpu['actual_local_source_archive_sha256']
            and len(wanted) == len(local) == len(members)
            and {member.name for member in members} == set(wanted)
            and all(member.isfile() and hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                    == wanted[member.name] for member in members))
    ready = checker.require_local_launch('live-poisson-probe')
    checks['exact_bounded_action_readiness'] = (
        ready['action'] == 'live-poisson-probe' and ready['online_launch_allowed'] is False
        and ready['new_numeric_qualification_inferred'] is False
        and ready['identity']['freeze_sha256'] == sha(checker.FREEZE)
        and ready['identity']['CPU_sha256'] == sha(checker.CPU))
    checks['preparation_closed'] = (
        json.loads((P/'V85_PARENT_SOURCE.json').read_text())['source_preparation_only'] is False
        and json.loads((P/'V85_LOCAL_RUN_PLAN.json').read_text())['source_preparation_only'] is False
        and checker.LOCAL_NPU_ACTIONS_DISABLED is False)
    initialized = any(bool(getattr(sys.modules.get(name), '_initialized', False))
                      for name in ('torch_npu.npu',))
    checks['root_HOST_NPU_not_initialized'] = initialized is False
    require(all(checks.values()), 'Actual root source/CPU/archive check failed: '+repr(checks))
    result = dict(schema='V85.actual-final-CPU-and-source-pin-audit.v1',
        created_UTC=datetime.now(timezone.utc).isoformat(),
        identity=ready['identity'], checks=checks, all_checks_passed=True,
        CPU_path=str(checker.CPU), freeze_path=str(checker.FREEZE),
        tests_run=cpu['tests_run'], CPU_modules=len(cpu['modules']),
        local_files=len(local), external_files=len(external),
        CPU_source_archive=str(archive_path),
        CPU_source_archive_sha256=sha(archive_path),
        root_auditor_path=str(Path(__file__).resolve()),
        root_auditor_sha256=sha(__file__), NPU_initialized=False,
        new_mixed_qualified=False, independent_AR_qualified=False,
        formal_Goodput=False, online_qualified=False)
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
