"""Actual private typed grants and genuine through-root owner remapping."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'))
import base199
import scope393,budget393,draft393,provider393
import provider199,split_budget_v132


def main():
    original=split_budget_v132.DraftWorkLimit
    actual=[]
    for case in scope393.cases():
        work=budget393.work_plan(case)
        assert sum(work.normal_budgets.values())==96
        assert sum(work.eager_budgets.values())==4*case.eager
        assert work.allocated_selected_candidates==96+4*case.eager<=128<=192
        ids=list(range(24))+list(case.eager_ids)
        work.require_work(ids,[4]*len(ids),[False]*24+[True]*case.eager,
            [SimpleNamespace(candidate_budget=4)for _ in ids])
        work.require_published_tickets([SimpleNamespace(request_index=i,gamma=4)for i in ids])
        assert len(set(ids))==len(ids)and set(case.eager_ids)<=set(range(24,48))
        counts=draft393.counts(case)
        assert len(counts)==64 and all(counts[i]==9 for i in case.eager_ids)
        assert counts[:24]==[4+case.repair_vector[i]for i in range(24)]
        assert all(counts[i]==4 for i in set(range(24,48))-set(case.eager_ids))
        assert counts[48:]==[0]*16
        contexts=draft393.contexts(case)
        old=provider199.source_contexts(case)
        for j in range(8):assert contexts[24+j]['token_ids']==old[16+j]['token_ids'][:129]
        assert all(len(row['token_ids'])==129+case.repair_vector[i]for i,row in enumerate(contexts))
        actual.append(dict(N=24,E=case.eager,R=case.repair_count,chain_M=scope393.bucket(case.count),
            F=scope393.bucket(case.eager),U=case.union_M,normal_C=96,eager_C=4*case.eager,
            selected_C=work.allocated_selected_candidates))
    assert len(actual)==16 and {x['chain_M']for x in actual}=={28,32}
    assert split_budget_v132.DraftWorkLimit is original and original().eager_per_request==3
    budget393.clear();assert split_budget_v132.DraftWorkLimit is original
    source=(P/'draft393.py').read_text()
    tree=ast.parse(source)
    assert not any(isinstance(n,ast.Constant)and type(n.value)is int and n.value in (16,20,32)for n in ast.walk(tree))
    projection=json.loads((P/'SOURCE_PROJECTION393.json').read_text())
    inverse=source
    for old,new,count in reversed(projection['draft_changes']):inverse=inverse.replace(new,old)
    assert inverse==Path(projection['draft_source']).read_text()
    assert 'torch'not in sys.modules
    result=dict(status='PASS_HOST393_ACTUAL_TYPED_GRANTS_SOURCE_OWNER_AND_ROLE_INVERSE',NPU_used=False,
        Torch_imported=False,actual_original_cap3_unchanged=True,actual64_prefix_contexts=True,
        parent_draft_role_projection_inverse_exact=True,cases=actual,model_qualification=False,
        runtime_prepared=False,production_permission=False,Raw=None,Goodput=None)
    (P/'HOST393_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cases=len(actual),NPU_used=False)))


if __name__=='__main__':main()
