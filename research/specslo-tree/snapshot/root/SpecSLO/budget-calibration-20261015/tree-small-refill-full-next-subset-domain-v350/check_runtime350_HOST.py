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
import runtime350
import guard350
import prefill350


def main():
    source = runtime350.worker_source()
    assert source.count('numeric_cases=16') == 2
    previous = sys.modules.get('check350')
    sys.modules['check350'] = SimpleNamespace(ready=lambda: None)
    try: actual = runtime350.build_runtime()
    finally:
        if previous is None: sys.modules.pop('check350')
        else: sys.modules['check350'] = previous
    assert inspect.signature(actual['worker']) == inspect.signature(runtime350.original.worker)
    assert actual['main'].__globals__['DONOR'] == guard350.DONOR
    ns = guard350.execution_namespace()
    assert ns['SCOPE']['finite_cases'] == 16 and ns['SCOPE']['independent_prefixes'] is None
    assert ns['SCOPE']['real_draft_generation'] is True
    projected = prefill350.project()
    local_imports = sorted({node.module for node in ast.walk(ast.parse(projected))
                            if isinstance(node, ast.ImportFrom) and node.module
                            and not node.module.startswith(('torch', 'vllm'))})
    assert 'subset345' in local_imports and 'subset350' not in local_imports
    for name in local_imports:
        assert importlib.util.find_spec(name) is not None, name
    import subset345
    assert Path(subset345.__file__).resolve() == B/'tree-coalesced-live-refill-host-v345/subset345.py'
    assert subset345.namespace()['prepare'].__module__ == 'subset209'
    imports = [node.args[0].value for node in ast.walk(ast.parse(projected))
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == '__import__' and node.args
               and isinstance(node.args[0], ast.Constant)]
    assert 'next4_350' in imports
    for name in imports:
        assert importlib.util.find_spec(name) is not None, name
    import next4_350
    next_ns = next4_350.namespace()
    assert next_ns['generate'].__globals__['work_plan'] is next_ns['work_plan']
    assert next_ns['generate'].__globals__['eager_reference'] is next_ns['eager_reference']
    for n in (1,2,4,16):
        case = dict(N=n, budgets=[4]*n, ids=list(range(n)))
        plan = next_ns['work_plan'](case, 1)
        assert dict(plan.normal_budgets) == dict.fromkeys(range(n), 4)
    for n in (0,3,8,17):
        try: next_ns['work_plan'](dict(N=n,budgets=[4]*n,ids=list(range(n))), 1)
        except RuntimeError: pass
        else: raise AssertionError('undeclared nextdraft size accepted')
    assert 'actual_prefill_capture_routes=list(saved.routes)' in projected
    assert 'original-exact-singleton-qualification-before-tap' in projected
    assert 'warmup_cases=prewarm' in projected
    body = prefill350.namespace()
    assert body['capture_begin'].__module__ == 'capture350'
    assert inspect.signature(body['run_case']) == inspect.signature(prefill350.original.run_case)
    assert not any(name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for name in sys.modules)
    value = dict(status='passed_small_component_runtime_source_HOST',NPU_used=False,
        original311_lifecycle_inverse=True, original_numerical_bodies_and_matchers=True,
        prewarm_cases=4, current_cases=12, total_prefill_cases=16,
        singleton_graph_and_multiowner_eager_recorded_separately=True,
        all_local_numerical_module_imports_resolved=local_imports,
        actual_nextdraft_home_sizes_checked=[1,2,4,16],
        original_independent_nextdraft_bodies_and_matchers_retained=True,
        no_model_qualification=True)
    (P/'HOST350_RUNTIME_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
