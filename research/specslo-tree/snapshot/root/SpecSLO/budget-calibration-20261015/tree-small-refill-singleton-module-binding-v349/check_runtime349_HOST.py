"""Compile the actual retained lifecycle and all original numerical bodies."""
import ast
import inspect
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B/'tree-coalesced-live-refill-host-v345'),
               str(B/'tree-high-normal-cap4-native-families-v148'),
               str(B/'tree-active-window32-16-local-v160')]
import runtime349
import guard349
import prefill349


def main():
    source = runtime349.worker_source()
    assert source.count('numeric_cases=16') == 2
    previous = sys.modules.get('check349')
    sys.modules['check349'] = SimpleNamespace(ready=lambda: None)
    try: actual = runtime349.build_runtime()
    finally:
        if previous is None: sys.modules.pop('check349')
        else: sys.modules['check349'] = previous
    assert inspect.signature(actual['worker']) == inspect.signature(runtime349.original.worker)
    assert actual['main'].__globals__['DONOR'] == guard349.DONOR
    ns = guard349.execution_namespace()
    assert ns['SCOPE']['finite_cases'] == 16 and ns['SCOPE']['independent_prefixes'] is None
    assert ns['SCOPE']['real_draft_generation'] is True
    projected = prefill349.project()
    local_imports = sorted({node.module for node in ast.walk(ast.parse(projected))
                            if isinstance(node, ast.ImportFrom) and node.module
                            and not node.module.startswith(('torch', 'vllm'))})
    assert 'subset345' in local_imports and 'subset349' not in local_imports
    for name in local_imports:
        assert importlib.util.find_spec(name) is not None, name
    import subset345
    assert Path(subset345.__file__).resolve() == B/'tree-coalesced-live-refill-host-v345/subset345.py'
    assert subset345.namespace()['prepare'].__module__ == 'subset209'
    assert 'actual_prefill_capture_routes=list(saved.routes)' in projected
    assert 'original-exact-singleton-qualification-before-tap' in projected
    assert 'warmup_cases=prewarm' in projected
    body = prefill349.namespace()
    assert body['capture_begin'].__module__ == 'capture349'
    assert inspect.signature(body['run_case']) == inspect.signature(prefill349.original.run_case)
    assert not any(name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for name in sys.modules)
    value = dict(status='passed_small_component_runtime_source_HOST',NPU_used=False,
        original311_lifecycle_inverse=True, original_numerical_bodies_and_matchers=True,
        prewarm_cases=4, current_cases=12, total_prefill_cases=16,
        singleton_graph_and_multiowner_eager_recorded_separately=True,
        all_local_numerical_module_imports_resolved=local_imports,
        no_model_qualification=True)
    (P/'HOST349_RUNTIME_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
