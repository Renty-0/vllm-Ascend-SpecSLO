"""Actual frozen installer/accounting closure, all532 declared inputs, no models."""
import json
from pathlib import Path
from types import SimpleNamespace
import runtime329

P=Path(__file__).resolve().parent
B=P.parent
from check209 import require_ready
from base209 import PARENT_ACTION
from diagnostic_authorization import activate_local_diagnostic

original=require_ready()
spec=dict(action=PARENT_ACTION,identity=original['parent']['identity'])
output=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/HOST329-capacity-preflight-v3'
output.mkdir(exist_ok=False)
with activate_local_diagnostic(spec):
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from extension218 import prepare_install
    entry,_=prepare_install(str(output))
    from install234 import install_entry
    policy=install_entry(entry,native)
    account=native.NativePearlEngine._specslo_account_real_target_plans
    import oracle326
    from plan326 import make
    cases=oracle326.cases()
    before=dict(C=policy.candidate_roof,Q=policy.target_queries,N=policy.max_target_requests)
    blocked=next(c for c in cases if c['name']=='N25-fork5-anchor0')
    engine=SimpleNamespace()
    plans=[make(b,127,blocked['anchor'])for b in blocked['budgets']]
    try:
        account(engine,plans)
    except RuntimeError as error:
        assert 'Root-inclusive target verification budget exceeded'in str(error)
    else:
        raise AssertionError('old actual accounting did not reproduce observed cap')
    from capacity329 import bind
    proof=bind(policy,native)
    for case in cases:
        plans=[make(b,127,case['anchor'])for b in case['budgets']]
        account(engine,plans)
    assert len(cases)==532 and engine._dynamic_tree_current_target_max_queries==160
    rejected=[]
    for label,values in [('C129',[8]*16+[1]),('N33',[1]*33),('groups33',[5]+[4]*31),('branches13',[5]*13),('B9',[9]),('empty',[])]:
        try:
            policy.check_target(values)
        except (ValueError,RuntimeError):
            rejected.append(label)
        else:
            raise AssertionError('invalid capacity accepted '+label)
    from budget_policy import TreeBudgetPolicy
    fresh=TreeBudgetPolicy(policy.model)
    assert(fresh.candidate_roof,fresh.target_queries,fresh.max_target_requests)==(96,128,32)
    cfg=runtime329.make_config()
    native_cfg=cfg.to_native()
    assert cfg.spec_rhythm_verification_budget==native_cfg.spec_rhythm_verification_budget==128
    result=dict(passed=True,actual_frozen_installer_executed=True,actual_accounting_callback_all532_cases_passed=True,
        old_capacity_failure_reproduced=True,old=before,capacity_binding=proof,invalid_limits_rejected=rejected,
        original_policy_class_defaults_unchanged=True,constructor_candidates128_verified=True,
        model_weights_loaded=False,NPU_model_executed=False,Goodput=None)
    with(P/'HOST329_ACTUAL_INSTALL_CAPACITY_RESULT.json').open('x')as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
