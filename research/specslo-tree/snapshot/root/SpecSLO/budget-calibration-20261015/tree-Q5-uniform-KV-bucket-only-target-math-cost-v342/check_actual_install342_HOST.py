"""Run the exact premodel installer; bind its actual closures without weights."""
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import runtime342

P = Path(__file__).resolve().parent
B = P.parent

from check209 import require_ready
from base209 import PARENT_ACTION
from diagnostic_authorization import activate_local_diagnostic

readiness = require_ready()
spec = dict(action=PARENT_ACTION, identity=readiness['parent']['identity'])
output = B/'tree-causal80-small-cohort-role-aware-audit-full-v209/HOST342-exact-installer-v1'
output.mkdir(exist_ok=False)
with activate_local_diagnostic(spec):
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from extension218 import prepare_install
    from install234 import install_entry
    entry, _ = prepare_install(str(output))
    policy = install_entry(entry, native)
    import metadata341
    original = native.NativePearlEngine.target_tree_forward
    graph = metadata341.functions(original)
    assert any(f.__name__ == 'checked_target' for f in graph)
    assert any(f.__name__ == 'qualified' for f in graph)
    assert any('host-and-root-padding' in f.__code__.co_filename for f in graph)
    from vllm_ascend.spec_decode.pearl import native_graph,native_model
    assert native_model.run_native_fused_infer_attention is native_graph.run_native_fused_infer_attention
    causal_proof=None
    model = SimpleNamespace(_mc2_chained_row_counts={96, 120, 128},
        _mc2_deferred_row_counts=set(), make_tree_attention_metadata=lambda: None)
    engine = SimpleNamespace(is_draft=False, model=model,
        graph_runner=SimpleNamespace(run_target_greedy=lambda *args: None))
    engine.target_tree_forward = MethodType(original, engine)
    activation = metadata341.install(engine)
    assert native.NativePearlEngine.target_tree_forward is original
    assert engine.target_tree_forward.__func__ is not original
    new_graph = metadata341.functions(engine.target_tree_forward.__func__)
    assert [f.__code__ for f in graph] == [f.__code__ for f in new_graph]
    from geometry342 import cases
    from geometry336 import make_plans
    import json
    fixture = json.loads((B/'tree-M160-S32-N32-target-math-cost-v326/FIXTURE326.json').read_text())
    account = native.NativePearlEngine._specslo_account_real_target_plans
    for case in cases():
        account(engine, make_plans(fixture['prefixes'], case['owners'], case['width']))
    result = dict(status='passed_actual_premodel_installer_HOST',
        NPU_used=False, model_constructed=False, real_target_executed=False,
        actual_install234_executed=True, checked_and_qualified_wrappers_retained=True,
        complete_actual_wrapper_codes_identical=True, class_method_unchanged=True,
        original_accounting_all28_cases_passed=True,causal_proof=causal_proof, activation=activation)
    with (P/'HOST342_ACTUAL_INSTALL_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
