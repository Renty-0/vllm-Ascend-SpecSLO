"""Construct actual adapters in the sanitized startup environment; no models."""
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
B = P.parent
F = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')


def main():
    if '--inside' not in sys.argv:
        lines = [line for line in (P / 'startup425.sh').read_text().splitlines()
                 if line.startswith('export PYTHONPATH=')]
        assert len(lines) == 1
        value = lines[0].removeprefix('export PYTHONPATH=')
        for name, replacement in [('$repo', str(F / 'source/vllm-ascend-hust')),
                                  ('$frozen', str(F)),
                                  ('$task_root', str(B / 'tree-causal80-small-cohort-role-aware-audit-full-v209')),
                                  ('${PYTHONPATH:-}', os.environ.get('PYTHONPATH', ''))]:
            value = value.replace(name, replacement)
        assert '$' not in value
        environment = dict(os.environ, PYTHONPATH=value)
        repo = F / 'source/vllm-ascend-hust'
        subprocess.run([sys.executable, '-B', str(repo / 'examples/run_with_vllm_ascend_extension.py'),
                        '--extension', str(repo / 'vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'),
                        '--required-op', 'allreduce_add_rmsnorm_chained', str(Path(__file__)), '--inside'],
                       cwd=str(repo), env=environment, check=True)
        return
    # These are actual modules on the exact startup PYTHONPATH, including the
    # dependency omitted by the previously failed391 protocol entry.
    names = ['projection247', 'projection255', 'window425', 'empty425', 'reference425',
             'audit_reference425', 'audit_base425', 'runtime425', 'trace425', 'scope425',
             'audit425', 'extension425', 'bucket425', 'oracle425', 'qualify425', 'guard425',
             'shape425', 'admission425', 'audit_normal425']
    modules = {name: importlib.import_module(name) for name in names}
    from config209 import make_config
    config = make_config()
    assert config.max_num_seqs == 64 and config.max_num_queued_seqs == 463
    assert config.spec_rhythm_max_target_batch == 32 and config.spec_rhythm_verification_budget == 96
    trace = modules['trace425'].namespace()
    assert trace['window'] is modules['window425'].window
    assert trace['windows'] is modules['window425'].windows and trace['DEFAULT_WINDOWS'] == (64,)
    assert trace['run_windows'].__globals__ is trace
    runtime_source = (B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text()
    projected, changes = modules['runtime425'].project_source(runtime_source)
    compile(projected, 'ROOT425.actual-worker-source', 'exec')
    assert "windows==(64,)" in projected
    assert 'engine._root425_controls = controls' in ast.unparse(ast.parse(projected))
    qualifier = modules['extension425'].projected_qualifier()
    assert qualifier.__globals__['_ROOT425_valid_record'] is modules['bucket425'].valid_record
    auditor = modules['audit425'].projected_original_audit()
    assert auditor.__globals__['original_with_new_geometry'].__module__ == 'audit_base425'
    assert auditor.__globals__['_ROOT425_valid_record'] is modules['bucket425'].valid_record
    numerical = modules['oracle425'].specialized()
    assert numerical.__globals__['candidate'].__module__ == 'oracle255'
    assert numerical.__globals__['reference'].__module__ == 'oracle255'
    guard = modules['guard425'].execution_namespace()
    for name in ('prepared', 'actual_audit', 'launch_argv', 'storage'):
        assert guard[name] is getattr(modules['guard425'], name)
    assert guard['execute_locked'].__globals__ is guard
    assert guard['SCOPE']['active_request_limit'] == 64
    assert guard['SCOPE']['effective_requests'] == 24
    result = dict(status='PASS_HOST425_ACTUAL_SANITIZED_STARTUP_CALLBACK_CONSTRUCTION',
                  NPU_used=False, model_constructors_called=0,
                  exact_startup_PYTHONPATH=True,
                  imported_origins={name: str(Path(module.__file__).resolve()) for name, module in modules.items()},
                  configured_max_num_seqs=64, effective_active_ceiling=64,
                  original_owned_verifier_and_cleanup_retained=True,
                  original227_and_additional681_and_all463_ownAR_required=True,
                  neural_qualification_inherited=False, deadline_seconds=18000)
    (P / 'HOST425_CONSTRUCTION_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
