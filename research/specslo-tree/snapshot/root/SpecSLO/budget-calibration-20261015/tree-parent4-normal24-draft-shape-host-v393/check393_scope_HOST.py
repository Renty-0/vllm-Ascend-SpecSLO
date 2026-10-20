"""Exercise actual finite N24 helper scopes and selected-budget negatives."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
P=Path(__file__).parent;B=P.parent
sys.path[:0]=[str(B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'),
    str(B/'tree-parent4-current-M120-target-protocol-v391'),
    str(B/'tree-parent4-best376-target-binding-host-v390')]
import base199,scope393,live393,budget393


def main():
    trials=[]
    for case in scope393.cases():
        e=SimpleNamespace(_v173_qualified_component=object())
        ids=list(range(24))+list(case.eager_ids)
        plans=[SimpleNamespace(candidate_budget=4)for _ in ids]
        sources={i:(SimpleNamespace(candidate_budget=4),[1,2,3,4])for i in case.eager_ids}
        repairs=[(0,128,[7])]*case.repair_count
        frontiers=[(i,132,9)for i in case.eager_ids]
        native=SimpleNamespace(_next_linear_draft_graph_bucket=lambda n,cap:scope393.bucket(n))
        with live393.diagnostic(e,case):
            live393.validate(e,native,plans,ids,sources,repairs,frontiers,4)
            for wrong in (3,5):
                try:live393.validate(e,native,plans,ids,sources,repairs,frontiers,wrong)
                except RuntimeError:pass
                else:raise AssertionError('shared step scope must reject')
        assert not hasattr(e,'_v199_diagnostic')
        trials.append(dict(case=case.name,scope_positive=True,wrong_steps_rejected=True,context_revoked=True))
    work=budget393.work_plan(scope393.Case(8,1))
    try:replace(work,available_selected_candidates=127)
    except ValueError as error:assert 'limit'in str(error).lower()
    else:raise AssertionError('selected128 cannot fit available127')
    budget393.clear()
    assert 'torch'not in sys.modules
    value=dict(status='PASS_HOST393_ACTUAL_FINITE_SCOPE_AND_GLOBAL_BUDGET_NEGATIVES',
        NPU_used=False,Torch_imported=False,CPU_scope_carriers_only=True,
        cases=trials,global_selected_limit_negative=True,model_qualification=False,
        runtime_prepared=False,production_permission=False,Raw=None,Goodput=None)
    (P/'HOST393_SCOPE_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],cases=len(trials),NPU_used=False)))


if __name__=='__main__':main()
