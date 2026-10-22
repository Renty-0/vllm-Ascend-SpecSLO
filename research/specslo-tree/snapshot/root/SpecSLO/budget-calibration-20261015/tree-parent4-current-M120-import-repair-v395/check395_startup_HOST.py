"""Test the exact repaired startup imports and original391 callback construction."""
import ast
import hashlib
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
        import guard395
        guard395.parent.EXPECTED_FREEZE = guard395.PARENT_FREEZE
        guard395.parent.EXPECTED_ARCHIVE = guard395.PARENT_ARCHIVE
        argv = guard395.launch_argv('HOST395-imports-only-no-model')
        assert argv.count(str(P / 'startup395.sh')) == 1
        environment = dict(os.environ)
        for argument in argv:
            if argument.startswith('SPECSLO_') and '=' in argument:
                key, value = argument.split('=', 1)
                environment[key] = value
        lines = [line for line in (P / 'startup395.sh').read_text().splitlines()
                 if line.startswith('export PYTHONPATH=')]
        assert len(lines) == 1
        value = lines[0].removeprefix('export PYTHONPATH=')
        for old, replacement in [('$repo', str(F / 'source/vllm-ascend-hust')),
                                 ('$frozen', str(F)),
                                 ('$task_root', str(B / 'tree-causal80-small-cohort-role-aware-audit-full-v209')),
                                 ('${PYTHONPATH:-}', os.environ.get('PYTHONPATH', ''))]:
            value = value.replace(old, replacement)
        assert '$' not in value
        environment['PYTHONPATH'] = value
        repo = F / 'source/vllm-ascend-hust'
        subprocess.run([sys.executable, '-B', str(repo / 'examples/run_with_vllm_ascend_extension.py'),
                        '--extension', str(repo / 'vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'),
                        '--required-op', 'allreduce_add_rmsnorm_chained', str(Path(__file__)), '--inside'],
                       env=environment, cwd=str(repo), check=True)
        return
    names = ['projection247', 'projection243', 'projection249', 'projection255',
             'metadata390', 'frontier392', 'target391', 'binding391', 'runtime391', 'runtime395',
             'actor391', 'component391', 'extension391', 'audit391', 'guard395']
    # Use the actual observer module's filename from the preserved392 directory.
    observer_root = B / 'tree-parent4-frontier-observer-before-capture-host-v392'
    observer_modules = [path.stem for path in observer_root.glob('*.py') if not path.name.startswith('check')]
    names.remove('frontier392')
    names.extend(observer_modules)
    modules = {name: importlib.import_module(name) for name in names}
    assert Path(modules['projection247'].__file__).resolve() == B / 'tree-static-q5-immutable-index-cache-host-v247/projection247.py'
    previous_startup = (B / 'tree-parent4-current-M120-target-protocol-v391/startup391.sh').read_text()
    assert 'tree-static-q5-immutable-index-cache-host-v247:' not in previous_startup
    assert 'tree-static-q5-immutable-index-cache-host-v247:' in (P / 'startup395.sh').read_text()
    from check391 import ready
    ready('a9a4bfaa81f9bc8f95a75a533612fb1bd8b8202638a6f65c6d35b8185fe8fa98')
    from config199 import make_config
    config = make_config()
    assert config.max_num_seqs == 64
    from vllm_ascend.spec_decode.pearl import api, native_engine
    constructors = [(api.PEARLEngine, api.PEARLEngine.__init__),
                    (native_engine.NativePearlEngine, native_engine.NativePearlEngine.__init__)]
    attempted = []
    def forbidden(*args, **kwargs):
        attempted.append(True)
        raise AssertionError('HOST395 forbids model constructors')
    try:
        for cls, original in constructors:
            cls.__init__ = forbidden
        actual = modules['runtime391'].build_runtime()
        assert set(actual) == {'worker', 'main'}
        assert actual['main'].__globals__['worker'] is modules['runtime391'].worker
        cloned = modules['target391'].cloned()
        assert cloned['metadata'] is modules['metadata390'].physical
        guard = modules['guard395'].execution_namespace()
        for name in ['prepared', 'storage', 'launch_argv']:
            assert guard[name] is getattr(modules['guard395'], name)
        assert guard['actual_audit'] is modules['guard395'].parent.actual_audit
        assert guard['execute_locked'].__globals__ is guard
    finally:
        for cls, original in constructors:
            cls.__init__ = original
    assert attempted == []
    result = dict(status='PASS_HOST395_EXACT_SANITIZED_STARTUP_AND_ORIGINAL391_BINDINGS',
                  NPU_used=False, model_constructors_called=0, constructors_blocked_in_HOST=True,
                  actual_startup_dependency_gap_reproduced_by_missing_original_directory=True,
                  repaired_projection247_import=True, configured_max_num_seqs=64,
                  original391_runtime_and_target_functions_preserved=True,
                  original391_SOURCE_SHA256=hashlib.sha256(
                      (B / 'tree-parent4-current-M120-target-protocol-v391/runtime391.py').read_bytes()).hexdigest(),
                  origins={name: str(Path(module.__file__).resolve()) for name, module in modules.items()},
                  production_permission=False, Raw=None, Goodput=None)
    (P / 'HOST395_STARTUP_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
