"""Scope provenance and callback corruption controls using HOST-only doubles."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

P = Path(__file__).parent
sys.path.append(str(P.parent / 'tree-static-q5-M120-SLO-first-full463-own-AR-v257'))


def main():
    import scope402 as scope
    def need(value):
        if value is not True:
            raise RuntimeError('ROOT402 HOST scope regression')
    def rejects(fn):
        try:
            fn()
        except (RuntimeError, AttributeError):
            return
        raise RuntimeError('ROOT402 HOST invalid provenance accepted')
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
        state = SimpleNamespace(prepare=object(), uniform_KV_bucket402_enabled=True)
        return SimpleNamespace(is_draft=False, cache_allocation=None, cache_block_tables=None,
            _best257_draft402=draft, _branch_tail_state=state,
            graph_runner=SimpleNamespace(graph_cache_sealed=False, run_target_greedy=object()),
            _uniform_KV_bucket402_proof=dict(cases=681, same_model_instance=True,
                original227_FULL_math_preserved=True))
    inherited = SimpleNamespace(require=lambda current: frozenset())
    original_issue = scope.original.issue
    original_current = scope.original.current
    try:
        scope.original.issue = lambda current, component_proof: inherited
        scope.original.current = lambda current: dict(original=True)
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
        prior = active._uniform_KV_bucket402_proof
        active._uniform_KV_bucket402_proof = dict(prior)
        rejects(lambda: permit.require(active))
        active._uniform_KV_bucket402_proof = prior
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
            damaged._uniform_KV_bucket402_proof[field] = value
            rejects(lambda: scope.issue(damaged, component_proof=object()))
        scope.original.issue = lambda current, component_proof: (_ for _ in ()).throw(RuntimeError('original gate fails'))
        rejects(lambda: scope.issue(engine(), component_proof=object()))
        need('torch' not in sys.modules and 'torch_npu' not in sys.modules)
        result = dict(status='PASS_HOST402_SCOPE_CALLBACK_AND_PROVENANCE_CORRUPTIONS',
            NPU_used=False, uses_HOST_doubles=True, original_scope_failure_retained=True,
            qualification_flag_assignments=False, production_qualified=False)
        (P / 'HOST402_SCOPE_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    finally:
        scope.original.issue, scope.original.current = original_issue, original_current


if __name__ == '__main__':
    main()
