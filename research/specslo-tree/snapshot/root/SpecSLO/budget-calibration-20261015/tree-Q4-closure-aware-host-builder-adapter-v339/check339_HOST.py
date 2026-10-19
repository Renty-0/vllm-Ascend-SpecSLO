"""Reproduce the old bypass and run the actual retained Native frontend on CPU."""
import ast
import copy
import __future__
import json
from pathlib import Path
import sys
import time
import types
from typing import Any, Sequence

P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(B/'tree-Q4-linear-target-math-cost-v336'))
import runtime336
sys.path.insert(0, str(B/'tree-Q4-actual-host-builder-adapter-v338'))


def fixture():
    from fixture326_HOST import build
    env = build()
    from vllm_ascend.spec_decode.pearl.tree import pack_selected_tree_plan, TreeSpeculationPlan
    from host_target_metadata import make_host_tree_attention_metadata, install as install_host
    engine, torch = env['engine'], env['torch']
    for name in ('_allocate_cache', '_activate_cache_sequence', '_ensure_cache_capacity',
                 '_release_cache', '_cache_slot_mapping', '_prepare_attention_metadata'):
        method = getattr(engine, name)
        setattr(engine, name, types.MethodType(method.__func__, engine))
    engine._release_cache()
    engine._allocate_cache([[31]*97], enable_prefix_caching=False, reserve_sequence_capacity=True)
    for owner in range(463, 527):
        engine._activate_cache_sequence(owner, [0], enable_prefix_caching=False)
    cfg = dict(vars(engine.config))
    cfg.update(enforce_eager=False, spec_rhythm_tree_width=2, spec_rhythm_tree_depth=4)
    engine.config = types.SimpleNamespace(**cfg)
    engine.is_draft = False
    engine.draft_vocab_size = engine.target_vocab_size = 100
    model = engine.model
    model._mc2_chained_row_counts = {96, 120, 128}
    model._mc2_deferred_row_counts = set()
    model_calls = []
    def make(this, plans, roots, draft, block_tables, *args, **kwargs):
        model_calls.append(1)
        return make_host_tree_attention_metadata(engine, plans, roots, draft, *args, **kwargs)
    model.make_tree_attention_metadata = types.MethodType(make, model)
    engine.graph_runner.last_target_execution = types.SimpleNamespace(
        used_aclgraph=True, mode='HOST-fixture-only', fallback_reason=None)
    engine.graph_runner.run_target_greedy = lambda xs, ps, ms, vocabulary_size: (torch.zeros_like(xs[0]),)
    path = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
    tree = ast.parse(path.read_text())
    node = copy.deepcopy(next(n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == 'target_tree_forward'))
    native = types.ModuleType('HOST339_actual_Native_frontend')
    native.__dict__.update(torch=torch, time=time, Sequence=Sequence, Any=Any,
        TreeSpeculationPlan=TreeSpeculationPlan, envs=types.SimpleNamespace(
        VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH=True), pack_selected_tree_plan=pack_selected_tree_plan,
        attention_backend_identity=lambda metadata: 'HOST-FULL' if metadata.tree_attention else 'HOST-causal')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec',
        flags=__future__.annotations.compiler_flag, dont_inherit=True), native.__dict__)
    cls = type('ActualFrontendCPU339', (), {'target_tree_forward':native.target_tree_forward})
    native.NativePearlEngine = cls
    from root_padding_forward231 import host_installer
    host_installer(install_host)(native)
    # Execute the actual verification wrapper and the exact installed entry
    # accounting wrapper, rather than only testing the bare host frontend.
    runner = engine.graph_runner
    runner.graph_cache_sealed = True  # HOST fixture cannot qualify a graph.
    class Runner:
        def _record_execution(self, *args, **kwargs):
            return None
    native.NativeACLGraphRunner = Runner
    import verification_qualification
    verification_qualification.install(native)
    entry_path = B/'tree-high-normal-cap4-native-families-v148/tree_entry.py'
    entry_tree = ast.parse(entry_path.read_text())
    checked = next(n for n in ast.walk(entry_tree) if isinstance(n, ast.FunctionDef) and n.name == 'checked_target')
    factory_node = ast.FunctionDef(
        name='factory', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='target'), ast.arg(arg='account_real_target_plans')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[copy.deepcopy(checked), ast.Return(ast.Name(id='checked_target', ctx=ast.Load()))], decorator_list=[])
    wrapper_namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory_node], type_ignores=[])), str(entry_path), 'exec'), wrapper_namespace)
    engine._HOST339_account_calls = []
    def account(this, plans):
        assert this is engine
        this._HOST339_account_calls.append([plan.candidate_budget for plan in plans])
    cls.target_tree_forward = wrapper_namespace['factory'](cls.target_tree_forward, account)
    engine.target_tree_forward = types.MethodType(cls.target_tree_forward, engine)
    return engine, model_calls


def invoke(engine, layout):
    from geometry336 import LAYOUTS
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    width, groups = LAYOUTS[layout]
    engine._q4_layout336 = layout
    ids = [0]+[463+j for j in range(groups-1)]
    plans = [cached_cpu_tree_speculation_plan(1, width-1, 97 if owner == 0 else 0,
             4096, candidate_budget=width-1) for owner in ids]
    return engine.target_tree_forward(plans, [41]*groups, [[43]*(width-1) for _ in ids],
        sequence_ids=ids, return_logits=False, real_tree_count=1)


def main():
    import metadata336
    engine, ignored = fixture()
    metadata336.install(engine)
    reproduced = False
    try:
        invoke(engine, 'Q4-S24')
    except AttributeError as error:
        assert "'NoneType' object has no attribute 'shape'" in str(error)
        reproduced = True
    assert reproduced and ignored == []
    import metadata339
    engine, ignored = fixture()
    original = engine.target_tree_forward.__func__
    old338_rejected = False
    import metadata338
    try:
        metadata338.install(engine)
    except RuntimeError as error:
        assert 'actual retained HOST builder' in str(error)
        old338_rejected = True
    assert old338_rejected
    activation = metadata339.install(engine)
    assert engine.target_tree_forward.__func__.__code__ is original.__code__
    outputs = []
    for layout in ('Q4-S24', 'Q4-S32', 'Q5-S24-control'):
        out = invoke(engine, layout)
        assert out['used_tree_attention'] is True and out['used_aclgraph'] is True
        assert out['logical_query_count'] == (5 if layout == 'Q5-S24-control' else 4)
        assert engine._q4_native_calls336[-1]['mask_shape'][2] == 10
        outputs.append(out['query_count'])
    assert outputs == [96, 128, 120] and ignored == []
    assert len(engine._HOST339_account_calls) == 3
    assert len(metadata339.functions(original)) >= 5
    assert 'torch_npu' not in sys.modules
    result = dict(status='passed_actual_retained_frontend_CPU', old337_bypass_reproduced=True, old338_wrapper_failure_reproduced=True, actual_checked_accounting_preserved=True,
        model_callback_never_called_in_actual_fast_frontend=True,
        actual_frontend_code_unchanged=True, actual_HOST_builder_rebound=True,
        physical_rows=outputs, activation=activation, fake_graph_is_not_NPU_evidence=True,
        real_model_constructed=False, NPU_used=False, model_math_qualified=False)
    with (P/'HOST339_ACTUAL_FRONTEND_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
