"""Compile the actual retained worker/main projection and compare signatures."""
import ast
import inspect
import json
from pathlib import Path
import sys
import runtime336

P = Path(__file__).resolve().parent


def main():
    source = Path(runtime336.original.__file__).read_text()
    projected, changes = runtime336.project_source(source)
    nodes = [n for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name in ('worker', 'main')]
    assert len(nodes) == 2
    namespace = dict(vars(runtime336.original), DONOR=P)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'HOST336-retained-loader', 'exec'), namespace)
    original = runtime336.original.worker
    actual = namespace['worker']
    assert inspect.signature(actual) == inspect.signature(original)
    assert actual.__defaults__ == original.__defaults__
    assert actual.__kwdefaults__ == original.__kwdefaults__
    assert namespace['DONOR'] == P and namespace['P'] == runtime336.OLD
    assert not any(n.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') for n in sys.modules)
    result = dict(status='passed_original_loader_projection_HOST',
        exact_source_replacement_counts={before:count for before, after, count in changes},
        worker_and_main_compiled=True, original_worker_defaults_preserved=True,
        complete_signature_and_defaults_equality_checked=True,
        private_output_and_original_fixture_paths_checked=True,
        Torch_imported=False, NPU_used=False, model_constructed=False)
    with (P/'HOST336_RUNTIME_PROJECTION_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
