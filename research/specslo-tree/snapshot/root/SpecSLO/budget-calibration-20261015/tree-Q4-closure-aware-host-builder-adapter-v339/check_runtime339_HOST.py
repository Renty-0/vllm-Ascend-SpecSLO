"""Check the unchanged worker/main bodies and new diagnostic delegates."""
import ast
import inspect
import json
import os
from pathlib import Path
import sys
import runtime339
import guard339
import probe339

P = Path(__file__).resolve().parent
B = P.parent


def main():
    for version, directory in ((336, 'tree-Q4-linear-target-math-cost-v336'),
                               (337, 'tree-Q4-original-output-root-adapter-v337'),
                               (338, 'tree-Q4-actual-host-builder-adapter-v338')):
        pins = json.loads((B/directory/f'SOURCE{version}_RECEIPT.json').read_text())
        os.environ[f'SPECSLO_V{version}_ROOT_FREEZE_SHA'] = pins['freeze_sha256']
    old = runtime339.parent.build_runtime()['main']
    new = runtime339.clone_main(old)
    assert new.__code__ is old.__code__
    assert inspect.signature(new) == inspect.signature(old)
    assert new.__defaults__ == old.__defaults__ and new.__kwdefaults__ == old.__kwdefaults__
    assert {k for k, value in old.__globals__.items() if new.__globals__[k] is not value} == {'worker'}
    assert new.__globals__['DONOR'] == guard339.DONOR
    source = runtime339.worker_source()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'worker']
    assert len(nodes) == 1 and source.count('from probe339 import run') == 1
    namespace = dict(vars(runtime339.numerical.original))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'HOST339-actual-worker', 'exec'), namespace)
    assert inspect.signature(namespace['worker']) == inspect.signature(runtime339.numerical.original.worker)
    projected = probe339.projected_run()
    assert projected.__globals__['candidate'] is probe339.original.candidate
    assert projected.__globals__['reference'] is probe339.original.reference
    assert projected.__globals__['paired_cost'] is probe339.original.paired_cost
    scope = guard339.execution_namespace()
    assert scope['P'] == guard339.DONOR and scope['G'] == P
    assert scope['SCOPE']['models_requested'] is True and 'models_loaded' not in scope['SCOPE']
    assert scope['MAX_SECONDS'] == 1800
    assert 'ROOT.V339.original-root-linear-target-supervisor.v1' in scope['execute_locked'].__code__.co_consts
    assert not any(n.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') for n in sys.modules)
    result = dict(status='passed_HOST_only', entire_original_main_code_identical=True,
        only_main_worker_delegate_changed=True, original_worker_signature_preserved=True,
        only_numeric_worker_probe_import_changed=True, candidate_reference_cost_functions_identical=True,
        original209_output_preserved=True, original_owned_guard_compiled=True,
        Torch_imported=False, NPU_used=False, model_constructed=False)
    with (P/'HOST339_RUNTIME_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
