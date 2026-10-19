"""Execute original output-root guard prefix and compare the retained lifecycle."""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import runtime337
import guard337

P = Path(__file__).resolve().parent


def main():
    pins = json.loads((runtime337.PARENT/'SOURCE336_RECEIPT.json').read_text())
    os.environ['SPECSLO_V336_ROOT_FREEZE_SHA'] = pins['freeze_sha256']
    original = runtime337.parent.build_runtime()['main']
    adapted = runtime337.clone_main(original)
    assert adapted.__code__ is original.__code__
    assert inspect.signature(adapted) == inspect.signature(original)
    assert adapted.__defaults__ == original.__defaults__ and adapted.__kwdefaults__ == original.__kwdefaults__
    changed = {k for k, v in original.__globals__.items() if adapted.__globals__[k] is not v}
    assert changed == {'DONOR', 'worker'}
    assert adapted.__globals__['DONOR'] == runtime337.DONOR
    assert adapted.__globals__['P'] == runtime337.parent.OLD
    import progress209
    source = Path(progress209.__file__).read_text()
    node = copy.deepcopy(next(n for n in ast.parse(source).body
                              if isinstance(n, ast.FunctionDef) and n.name == 'configure'))
    boundary = next(i for i, n in enumerate(node.body)
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_OUTPUT' for t in n.targets))
    node.body = node.body[:boundary]+[ast.Return(value=ast.Name(id='output', ctx=ast.Load()))]
    namespace = dict(Path=Path, P=progress209.P, require=progress209.require, _OUTPUT=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 __file__+'::actual-constructor-root-and-lifetime-guard-prefix', 'exec'), namespace)
    rejected = False
    with tempfile.TemporaryDirectory(prefix='HOST337-outside-', dir=P) as outside:
        try:
            namespace['configure'](outside)
        except RuntimeError:
            rejected = True
    assert rejected
    with tempfile.TemporaryDirectory(prefix='HOST337-allowed-', dir=runtime337.DONOR) as inside:
        assert namespace['configure'](inside) == Path(inside)
    scope = guard337.execution_namespace()
    assert scope['P'] == runtime337.DONOR and scope['G'] == P
    assert scope['SCOPE']['models_requested'] is True and 'models_loaded' not in scope['SCOPE']
    assert scope['MAX_SECONDS'] == 1800 and scope['MAX_AGGREGATE_BYTES'] == 2560*1024**2
    assert 'ROOT.V337.original-root-linear-target-supervisor.v1' in scope['execute_locked'].__code__.co_consts
    assert not any(n.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') for n in sys.modules)
    result = dict(status='passed_HOST_only', original_configure_guard_prefix_executed=True,
        original336_output_rejected=True, original209_output_accepted=True,
        entire_parent_main_code_identical=True, changed_main_globals=sorted(changed),
        defaults_and_signature_exact=True, original_fixture_directory_preserved=True,
        declared_models_requested_not_loaded=True, original_owned_guard_compiled=True,
        parent_numeric_source_unchanged=True, model_constructed=False, NPU_used=False,
        Torch_imported=False, no_model_qualification_claimed=True)
    with (P/'HOST337_OUTPUT_ROOT_AND_LIFECYCLE_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
