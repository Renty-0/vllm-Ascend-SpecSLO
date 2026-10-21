"""Scope provenance and callback corruption controls using HOST-only doubles."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

P = Path(__file__).parent
sys.path.append(str(P.parent / 'tree-static-q5-M120-SLO-first-full463-own-AR-v257'))


def main():
    import scope409 as scope
    import scope_wide409 as wide
    def need(value):
        if value is not True:
            raise RuntimeError('ROOT409 HOST scope regression')
    def rejects(fn):
        try:
            fn()
        except (RuntimeError, AttributeError):
            return
        raise RuntimeError('ROOT409 HOST invalid provenance accepted')
    draft = object()
    native = ModuleType('vllm_ascend.spec_decode.pearl.native_engine')
    native.NativePearlEngine = SimpleNamespace(draft_tree_forward=draft)
    pearl = ModuleType('vllm_ascend.spec_decode.pearl')
    pearl.native_engine = native
    sys.modules['vllm_ascend'] = ModuleType('vllm_ascend')
    sys.modules['vllm_ascend.spec_decode'] = ModuleType('vllm_ascend.spec_decode')
    sys.modules['vllm_ascend.spec_decode.pearl'] = pearl
    sys.modules['vllm_ascend.spec_decode.pearl.native_engine'] = native
    def engine():
        state = SimpleNamespace(prepare=object(), uniform_KV_bucket409_enabled=True)
        result = SimpleNamespace(is_draft=False, cache_allocation=None, cache_block_tables=None,
            _best257_draft409=draft, _branch_tail_state=state,
            graph_runner=SimpleNamespace(graph_cache_sealed=False, run_target_greedy=object()),
            _uniform_KV_bucket409_proof=dict(cases=681, same_model_instance=True,
                original227_FULL_math_preserved=True),
            _S32_409_current_proof=dict(cases=200, same_model_instance=True, Native_flags_assigned=False))
        result._S32_409_proof_owner = result
        return result
    inherited = SimpleNamespace(require=lambda current: frozenset())
    original_issue = scope.original.issue
    original_current = scope.original.current
    original_wide_local = wide.local_proof
    try:
        scope.original.issue = lambda current, component_proof: inherited
        scope.original.current = lambda current: dict(original=True)
        broken = engine()
        broken._S32_409_proof_owner = object()
        rejects(lambda: original_wide_local(broken))
        broken = engine()
        broken._S32_409_current_proof['cases'] = 199
        rejects(lambda: original_wide_local(broken))
        # Positive scope chaining uses HOST doubles; it is not model evidence.
        wide.local_proof = lambda current: (current._S32_409_current_proof, {})
        active = engine()
        permit = scope.issue(active, component_proof=object())
        need(permit.require(active) == frozenset())
        rejects(lambda: permit.require(engine()))
        rejects(lambda: scope.issue(active, component_proof=object()))
        prior = active._branch_tail_state.prepare
        active._branch_tail_state.prepare = object()
        rejects(lambda: permit.require(active))
        active._branch_tail_state.prepare = prior
        prior = active.graph_runner.run_target_greedy
        active.graph_runner.run_target_greedy = object()
        rejects(lambda: permit.require(active))
        active.graph_runner.run_target_greedy = prior
        native.NativePearlEngine.draft_tree_forward = object()
        rejects(lambda: permit.require(active))
        native.NativePearlEngine.draft_tree_forward = draft
        original_wide = active._S32_409_current_proof
        active._S32_409_current_proof = dict(original_wide)
        rejects(lambda: permit.require(active))
        active._S32_409_current_proof = original_wide
        prior = active._uniform_KV_bucket409_proof
        active._uniform_KV_bucket409_proof = dict(prior)
        rejects(lambda: permit.require(active))
        active._uniform_KV_bucket409_proof = prior
        need(permit.require(active) == frozenset())
        for field, bad in [('cache_allocation', object()), ('cache_block_tables', object())]:
            damaged = engine()
            setattr(damaged, field, bad)
            rejects(lambda: scope.issue(damaged, component_proof=object()))
        damaged = engine()
        damaged.graph_runner.graph_cache_sealed = True
        rejects(lambda: scope.issue(damaged, component_proof=object()))
        for field, value in [('cases', 680), ('same_model_instance', False),
                             ('original227_FULL_math_preserved', False)]:
            damaged = engine()
            damaged._uniform_KV_bucket409_proof[field] = value
            rejects(lambda: scope.issue(damaged, component_proof=object()))
        scope.original.issue = lambda current, component_proof: (_ for _ in ()).throw(RuntimeError('original gate fails'))
        rejects(lambda: scope.issue(engine(), component_proof=object()))
        need('torch' not in sys.modules and 'torch_npu' not in sys.modules)
        result = dict(status='PASS_HOST409_SCOPE_CALLBACK_AND_PROVENANCE_CORRUPTIONS',
            NPU_used=False, uses_HOST_doubles=True, original_scope_failure_retained=True,
            qualification_flag_assignments=False, production_qualified=False)
        (P / 'HOST409_SCOPE_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    finally:
        scope.original.issue, scope.original.current = original_issue, original_current
        wide.local_proof = original_wide_local


if __name__ == '__main__':
    main()
