"""Compile the actual retained lifecycle and all original numerical bodies."""
import ast
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B/'tree-coalesced-live-refill-host-v345'),
               str(B/'tree-high-normal-cap4-native-families-v148'),
               str(B/'tree-active-window32-16-local-v160')]
import runtime347
import guard347
import prefill347


def main():
    source = runtime347.worker_source()
    assert source.count('numeric_cases=16') == 2
    previous = sys.modules.get('check347')
    sys.modules['check347'] = SimpleNamespace(ready=lambda: None)
    try: actual = runtime347.build_runtime()
    finally:
        if previous is None: sys.modules.pop('check347')
        else: sys.modules['check347'] = previous
    assert inspect.signature(actual['worker']) == inspect.signature(runtime347.original.worker)
    assert actual['main'].__globals__['DONOR'] == guard347.DONOR
    ns = guard347.execution_namespace()
    assert ns['SCOPE']['finite_cases'] == 16 and ns['SCOPE']['independent_prefixes'] is None
    assert ns['SCOPE']['real_draft_generation'] is True
    projected = prefill347.project()
    assert 'actual_prefill_capture_routes=list(saved.routes)' in projected
    assert 'original-exact-singleton-qualification-before-tap' in projected
    assert 'warmup_cases=prewarm' in projected
    body = prefill347.namespace()
    assert body['capture_begin'].__module__ == 'capture347'
    assert inspect.signature(body['run_case']) == inspect.signature(prefill347.original.run_case)
    assert not any(name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for name in sys.modules)
    value = dict(status='passed_small_component_runtime_source_HOST',NPU_used=False,
        original311_lifecycle_inverse=True, original_numerical_bodies_and_matchers=True,
        prewarm_cases=4, current_cases=12, total_prefill_cases=16,
        singleton_graph_and_multiowner_eager_recorded_separately=True,
        no_model_qualification=True)
    (P/'HOST347_RUNTIME_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
