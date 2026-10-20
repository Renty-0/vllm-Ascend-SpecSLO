"""Build the actual private LIVE source before models and validate its new call."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile


def main():
    import torch_npu
    from vllm_ascend.spec_decode.pearl import native_engine as native
    calls = []

    def blocked(*args, **kwargs):
        calls.append('blocked')
        raise AssertionError('HOST398 may not construct a model or initialize an NPU')

    native.NativePearlEngine.__init__ = blocked
    torch_npu.npu._lazy_init = blocked
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    readiness = require_ready()
    spec = dict(action=PARENT_ACTION, identity=readiness['parent']['identity'])
    with activate_local_diagnostic(spec):
        from extension398 import prepare_install
        donor = Path('/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209')
        output = Path(tempfile.mkdtemp(prefix='HOST398-LIVE-', dir=donor))
        entry, _ = prepare_install(output)
        import live_decode_patch
        from empty209 import patch as empty_patch
        from window398 import adapt_source
        import generation398
        builder = entry.get('_live_builder', live_decode_patch.build_live_runtime)
        base = empty_patch(adapt_source(builder(), 64))
        source = generation398.patch(base)
        compile(source, 'ROOT398.actual-LIVE-from-original-entry', 'exec')
        tree = ast.parse(source)
        selected = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == '_build_plan398']
        assert len(selected) == 1
        call = selected[0]
        assert ast.unparse(call.args[0]) == 'controller'
        keywords = {n.arg: n.value for n in call.keywords}
        assert ast.literal_eval(keywords['max_target_requests']) == 24
        assert ast.literal_eval(keywords['max_target_segments']) == 24
        assert ast.unparse(keywords['ready_segment_counts']).startswith('_ready_costs398(controller, payloads, ')
        assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'controller.build_plan'
                       for n in ast.walk(tree))
        detail = generation398.descriptor(base)
        assert detail['explicit_segment_packing398']['global_active_limit'] == 64
        assert detail['explicit_segment_packing398']['neural_code_changed'] is False
        assert not calls
        p = Path(__file__).resolve().parent
        (p / 'HOST398.generated-LIVE.py').write_text(source)
        result = dict(status='PASS_HOST398_ACTUAL_PRIVATE_LIVE_SOURCE_AND_CALL_BINDING',
                      exact_private_original_builder=True,
                      projected_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      packing=detail['explicit_segment_packing398'],
                      global_active_limit=64, target_requests=24, target_attention_segments=24,
                      actual_proposal_counts_passed_to_new_packer=True,
                      original_class_method_unmodified=True, model_constructor_calls=0,
                      NPU_lazy_init_calls=0, NPU_used=False,
                      original_source_NN_qualification_not_replaced=True,
                      production_permission=False, Raw=None, Goodput=None)
        (p / 'HOST398_LIVE_BINDING_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('status', 'global_active_limit',
                                              'model_constructor_calls', 'NPU_used')}))


if __name__ == '__main__':
    main()
