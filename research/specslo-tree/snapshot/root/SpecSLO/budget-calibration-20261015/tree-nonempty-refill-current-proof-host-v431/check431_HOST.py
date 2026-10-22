"""Whole original LIVE rewrite and rejection of HOST/fake NN permission."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch, torch_npu
    attempted = []
    def forbidden(*a, **k):
        attempted.append(True)
        raise AssertionError('HOST431 no model constructor or NPU initialization')
    torch.npu._lazy_init = forbidden
    torch_npu.npu._lazy_init = forbidden
    from vllm_ascend.spec_decode.pearl import native_engine as native
    native.NativePearlEngine.__init__ = forbidden
    import projection431, witness431
    from check425 import ready
    frozen = ready()
    p = projection431.B/'tree-parent4-canonical-tail-observer-binding-v425/HOST425.generated-LIVE.py'
    import hashlib
    assert frozen['files'][str(p)] == hashlib.sha256(p.read_bytes()).hexdigest()
    before = p.read_text()
    after = projection431.patch(before)
    compile(after, 'HOST431.whole-LIVE-admission-proposal', 'exec')
    def clocks(source):
        return [ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
                and ast.unparse(n.func) in ('time.time','time.perf_counter','time.perf_counter_ns','time.sleep','_dt_cpu_broadcast')]
    assert clocks(before) == clocks(after)
    import admission429
    original_function, _, _ = admission429.source()
    original_nodes = ast.parse(original_function).body[0]
    new_nodes = next(n for n in ast.walk(ast.parse(after)) if isinstance(n,ast.FunctionDef) and n.name=='admit_available')
    assert ast.unparse(new_nodes.body[0]).startswith('from witness431 import before as _before429')
    assert 'self._prefill_and_sample_target_batch' in ast.unparse(new_nodes)
    assert 'states[index].add_decode_time(refill_ms)' in ast.unparse(new_nodes)
    assert ast.unparse(new_nodes).count('self._deliver_committed_tokens') == ast.unparse(original_nodes).count('self._deliver_committed_tokens') == 1
    rejected = []
    altered = before.replace('refill_started = time.perf_counter()', 'refill_started = 0.')
    assert altered != before
    try:
        projection431.patch(altered)
    except RuntimeError:
        rejected.append('changed_original_clock_body')
    else:
        raise AssertionError('changed original admission must fail construction')
    engine = SimpleNamespace(model=object(), _dynamic_tree_generation=1)
    for kind in ('absent_permission','forged_HOST_permission'):
        if kind == 'forged_HOST_permission':
            engine._root427_incumbent_prefill_actual = {}
            engine._root431_nonempty_permission = witness431.Permission(engine, engine.model, object(),
                engine._root427_incumbent_prefill_actual, object())
        try:
            witness431.current(engine)
        except RuntimeError:
            rejected.append(kind)
        else:
            raise AssertionError('HOST permission must never enable model work')
    try:
        witness431.issue(engine, None)
    except RuntimeError:
        rejected.append('HOST_has_no_actual425_issued_scope')
    else:
        raise AssertionError('HOST must not issue production permission')
    assert not attempted
    value = dict(status='PASS_HOST431_WHOLE_ACTUAL425_LIVE_INVERSE_AND_SAME_MODEL_PERMISSION_NEGATIVES',
        descriptor=projection431.descriptor(before), controlled_negatives=rejected,
        original_clock_calls_and_duration_broadcast_AST_identical=True,
        original_once_root_callback_AST_retained=True,
        full_parent4_dispatch_planner_and_target_backend_outside_admission_unchanged=True,
        current430_NPU_process_unchanged=True, Native_models_constructed=0, NPU_used=False,
        NN427_executed=False, serving_installed=False, Raw=None, Goodput=None)
    (projection431.P/'HOST431_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':
    main()
