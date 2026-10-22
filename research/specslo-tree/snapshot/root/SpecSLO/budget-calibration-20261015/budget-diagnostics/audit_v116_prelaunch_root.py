"""ROOT independent prepared-source audit; never initializes Torch or a model."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import unittest

B = Path(__file__).resolve().parent.parent
P = B/'tree-s35-combined-controls-launchfix-v116'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 11)
    frozen = json.loads((P/'PREPARED_SOURCE_HASHES_V116.json').read_text())
    cpu = json.loads((P/'CPU_RESULT_V116.json').read_text())
    assert cpu['status'] == 'passed' and cpu['tests_run'] >= 15
    assert cpu['failures'] == cpu['errors'] == 0
    assert cpu['NPU_initialized'] is cpu['model_initialized'] is False
    assert cpu['initialization_attempts'] == []
    assert cpu['input_hashes'] == cpu['input_hashes_after'] == frozen['files']
    assert cpu['external_input_hashes'] == cpu['external_input_hashes_after'] == frozen['external_files']
    assert frozen['CPU_sha256'] == sha(P/'CPU_RESULT_V116.json')
    expected = {**frozen['files'], **frozen['external_files'],
                str(P/'CPU_RESULT_V116.json'): sha(P/'CPU_RESULT_V116.json'),
                str(P/'PREPARED_SOURCE_HASHES_V116.json'): sha(P/'PREPARED_SOURCE_HASHES_V116.json')}
    for path, wanted in expected.items():
        assert sha(path) == wanted, path
    archive_members = []
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        actual = {}
        for member in members:
            assert member.isfile() and '..' not in Path(member.name).parts
            absolute = Path('/')/member.name
            path = absolute if str(absolute) in expected else P/member.name
            assert path.exists(), str(path)
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            assert digest == sha(path), str(path)
            actual[str(path)] = digest
            archive_members.append(member.name)
        assert expected.items() <= actual.items(), sorted(set(expected)-set(actual))[:12]
    parent = json.loads((P/'V113_PARENT_SOURCE.json').read_text())
    parent_cpu = json.loads(Path(parent['CPU']).read_text())
    assert sha(parent['CPU']) == parent['CPU_sha256'] and parent_cpu['tests_run'] == 166
    assert parent_cpu['status'] == 'passed' and parent_cpu['NPU_initialized'] is False
    config = json.loads((P/'COMBINATION_V113.json').read_text())
    assert config['combined'] == dict(target_cpu_context_clean=True, hardware_family_normal_cap=True)
    assert config['V99_shadow_enabled'] is config['FULL_HOST_memo_enabled'] is False
    assert config['shared_Native_namespace'] is True
    component = config['component_actual_evidence']['V110']
    assert sha(component['receipt']) == component['receipt_sha256']
    factory = config['full_pipeline_factory_source_proof']
    assert sha(factory['path']) == factory['sha256']
    proof = json.loads(Path(factory['path']).read_text())
    assert proof['source_dataflow_closed_for_V113_actual_target_pipeline'] is True
    assert proof['global_factory_equivalence_proved'] is False
    # Execute the actual immutable launcher guards and argument statements.
    # The fixture intercepts only runtime/python tool boundaries, before any
    # model, extension, CANN environment or device enumeration can be invoked.
    sys.path.insert(0, str(P))
    import test_launcher_v116_host as launcher
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(launcher.LauncherTests))
    assert tests.wasSuccessful() and tests.testsRun == 2, stream.getvalue()
    launcher_proof = launcher.LAST_SHELL_PROOF
    assert launcher_proof['positive']['exit_code'] == 0
    assert len(launcher_proof['negative']) == 6
    assert all(item['exit_code'] == 2 and item['tool_argv'] == []
               for item in launcher_proof['negative'])
    assert not any(name == 'torch' or name.startswith(('torch.', 'torch_npu')) for name in sys.modules)
    independent = B/'budget-diagnostics/ROOT_V110_ACTUAL_FULL96-CPUcontext-v1_AUDIT.json'
    assert json.loads(independent.read_text())['status'] == 'passed'
    argv = ['bash', str(B/'serial-v1/runtime.sh'), '--environment-only',
            '/root/miniconda3/envs/vllm-hust-dev/bin/python3.11', '-B',
            str(P/'check_v113.py'), '--require-local', 'combined-full463']
    result = subprocess.run(argv, text=True, capture_output=True, timeout=60, cwd=P)
    assert result.returncode == 0, result.stdout+result.stderr
    readiness = json.loads(result.stdout)
    scripts = re.findall(r'^source (/[^ \n]+)', (B/'serial-v1/runtime.sh').read_text(), re.MULTILINE)
    assert scripts == ['/root/data/specslo-cann-runtime/cann-9.0.0/set_env.sh',
                       '/root/data/specslo-cann-runtime/nnal/atb/9.0.0/atb/set_env.sh']
    report = dict(schema='ROOT.V116.independent-prelaunch-source-HOST-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), all_checks_passed=True,
        actual_ROOT_HOST_argv=sys.argv, ROOT_auditor_sha256=sha(Path(__file__)),
        CPU_tests=cpu['tests_run'], CPU_sha256=sha(P/'CPU_RESULT_V116.json'),
        freeze_sha256=sha(P/'PREPARED_SOURCE_HASHES_V116.json'),
        static_files=len(frozen['files']), external_files=len(frozen['external_files']),
        archive=str(args.archive), archive_sha256=sha(args.archive),
        archive_bytes=args.archive.stat().st_size, archive_members_actual_byte_matches=len(archive_members),
        parent_actual_HOST_tests=166, unchanged_parent_tests_reexecuted=False,
        ROOT_actual_launcher_HOST_tests=2,
        ROOT_actual_launcher_transcript=launcher_proof,
        component_actual_receipt=component, component_independent_external_audit_sha256=sha(independent),
        full_pipeline_factory_proof=factory, global_factory_equivalence_proved=False,
        readiness_argv=argv, readiness_exit_code=result.returncode,
        readiness_identity=readiness['identity'],
        readiness_stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
        actual_sourced_runtime_shells_before={name:sha(name) for name in scripts},
        actual_S35_device_qualified=False, first_S35_actual_scopes_required=21,
        first_S35_actual_phases_required=26, NPU_started=False, formal_Goodput=False)
    destination = B/'budget-diagnostics/ROOT_V116_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
    with destination.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
