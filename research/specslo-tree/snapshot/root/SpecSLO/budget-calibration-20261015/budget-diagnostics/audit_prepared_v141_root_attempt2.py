"""ROOT read-only closure audit for a frozen candidate; no model import."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--checker', required=True)
    parser.add_argument('--readiness-action')
    parser.add_argument('--minimum-tests', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 11)
    root = args.directory.resolve()
    frozen_path = root / ('PREPARED_SOURCE_HASHES_' + args.version + '.json')
    cpu_path = root / ('CPU_RESULT_' + args.version + '.json')
    frozen = json.loads(frozen_path.read_text())
    cpu = json.loads(cpu_path.read_text())
    assert cpu['status'] == 'passed' and cpu['tests_run'] >= args.minimum_tests
    assert cpu['failures'] == cpu['errors'] == 0
    assert cpu['NPU_initialized'] is cpu['model_initialized'] is False
    assert args.version == cpu['preparation_epoch'] == 'V141'
    assert cpu['schema'] == 'V141.three-actual-outer-config-TorchCPU-HOST.v1'
    assert cpu['Torch_imported'] is True
    assert cpu['NPU_or_ACL_queried'] is cpu['runtime_launcher_executed'] is False
    assert cpu['Torch_backend_autoload'] == '0' and cpu['actual_torch_threads'] == 1
    assert cpu['forbidden_import_attempts'] == cpu['forbidden_runtime_modules_loaded'] == []
    assert cpu['parent5_HOST_proof_reused'] is True and cpu['parent5_suite_rerun'] is False
    evidence = cpu['actual_configuration_evidence']
    assert evidence['actual_original_configuration_function_executed'] is True
    assert evidence['actual_API_classes_and_complete_post_init_executed'] is True
    assert evidence['actual_HF_configs_loaded_from_local_config_JSON'] is True
    assert evidence['real_torch_dtype_leaves'] is evidence['actual_parent_error_reproduced'] is True
    assert evidence['actual_outer_fields'] == 77
    assert evidence['NativeEngine_or_PEARLEngine_constructor_executed'] is False
    assert cpu['input_hashes'] == cpu['input_hashes_after'] == frozen['files']
    assert cpu['external_input_hashes'] == cpu['external_input_hashes_after'] == frozen['external_files']
    assert frozen['CPU_sha256'] == sha(cpu_path)
    expected = {**frozen['files'], **frozen['external_files'],
        str(cpu_path): sha(cpu_path), str(frozen_path): sha(frozen_path)}
    for name, digest in expected.items():
        assert sha(name) == digest, name
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        actual = {}
        retained_HOST_evidence = {}
        for member in members:
            assert member.isfile() and '..' not in Path(member.name).parts
            absolute = Path('/') / member.name
            path = absolute if str(absolute) in expected else root / member.name
            if str(path) not in expected:
                assert re.fullmatch(
                    r'CPU_V\d+\.log|host-attempts/attempt\d+-(?:CPU_RESULT_V\d+\.json|CPU_V\d+\.log)|host-attempts/attempt[123]\.json',
                    member.name), str(path)
                assert path.is_file() and path.resolve().is_relative_to(root)
                retained_HOST_evidence[str(path)] = sha(path)
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            assert digest == (expected | retained_HOST_evidence)[str(path)] == sha(path), str(path)
            actual[str(path)] = digest
        # The archive must include every local executable source and both
        # epoch evidence files. External sources may instead be hash-pinned.
        mandatory = set(frozen['files']) | {str(cpu_path), str(frozen_path)}
        assert mandatory <= set(actual), sorted(mandatory - set(actual))
    checker = root / args.checker
    argv = [sys.executable, '-B', str(checker)] + (
        ['--require-local', args.readiness_action] if args.readiness_action else ['--require-ready'])
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=60, cwd=root)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    readiness = json.loads(completed.stdout)
    assert readiness['identity']['CPU_sha256'] == sha(cpu_path)
    assert readiness['identity']['freeze_sha256'] == sha(frozen_path)
    assert readiness['scope']['formal_Goodput'] is readiness['scope']['extra4_adoption'] is readiness['actual_device_qualification'] is False
    for name, digest in expected.items():
        assert sha(name) == digest, name
    runtime_shells = {str(p): sha(p) for p in (
        Path('/root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh'),
        Path('/root/data/specslo-cann-runtime/cann-9.0.0/set_env.sh'),
        Path('/root/data/specslo-cann-runtime/nnal/atb/9.0.0/atb/set_env.sh'))}
    for name in frozen['files']:
        if name.endswith('.sh'):
            r = subprocess.run(['/bin/bash', '-n', name], capture_output=True, text=True, timeout=10)
            assert r.returncode == 0, r.stderr
    record = dict(schema='ROOT.frozen-candidate-pre-NPU-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='passed',
        directory=str(root), version=args.version, actual_python=sys.executable,
        CPU_sha256=sha(cpu_path), freeze_sha256=sha(frozen_path),
        actual_HOST_tests=cpu['tests_run'], local_source_count=len(frozen['files']),
        external_source_count=len(frozen['external_files']),
        archive=str(args.archive), archive_sha256=sha(args.archive),
        archive_members_all_bytes_checked=len(actual),
        retained_HOST_evidence=retained_HOST_evidence,
        all_local_sources_and_CPU_freeze_in_archive=True,
        all_pinned_sources_SHA_before_after_checked=True,
        readiness_argv=argv, readiness_exit_code=completed.returncode,
        readiness=readiness, runtime_shells=runtime_shells,
        NPU_started=False, model_initialized=False, formal_Goodput=False)
    with args.output.open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='passed', output=str(args.output), sha256=sha(args.output),
        local_sources=record['local_source_count'], external_sources=record['external_source_count'],
        archive_members=record['archive_members_all_bytes_checked'], NPU_started=False)))


if __name__ == '__main__':
    main()
