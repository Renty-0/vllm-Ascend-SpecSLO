"""Original selected-tree packer and original candidate DTO on the new pool."""
import importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def main():
    sys.path[:0]=[str(P),str(B/'tree-deferred-FULL-mask-host-projection-v259'),str(B/'tree-alternate-chain-target-host-v269')]
    from fixture259 import build
    env=build();tree=env['tree'];torch=env['torch']
    spec=importlib.util.spec_from_file_location('vllm_ascend.spec_decode.pearl.tree_budget',N/'tree_budget.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from plans282 import raw,select,PARENTS
    from plan269 import make
    cases=[]
    for prefix in(0,127,128,1505,2555):
        requests={}
        for b in range(1,9):
            plan=raw(b,prefix);assert plan.candidate_budget==b and tuple(plan.parent_indices.tolist())==PARENTS
            requests[b]=module.TreeCandidateRequest(parents=plan.parent_indices.tolist(),
                token_ids=list(range(10,18)),conditional_confidences=[.7]*8,
                max_candidates=b,minimum_candidates=b)
        selection=select(requests,sum(range(1,9)))
        for b in range(1,9):
            chosen=selection.selected_indices[b];assert chosen==tuple(range(b))
            packed=tree.pack_selected_tree_plan(raw(b,prefix),chosen);expected=make(b,prefix,0)
            assert packed.candidate_budget==expected.candidate_budget==b
            for name in('parent_indices','positions','cache_positions','attention_mask'):
                assert torch.equal(getattr(packed,name),getattr(expected,name)),(b,prefix,name)
            assert all(p<0 or p in chosen for i,p in enumerate(PARENTS[:b]))
            cases.append(dict(B=b,prefix=prefix,selected=list(chosen)))
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_original_candidate_DTO_and_selected_packer_B1to8_passed',
        cases=cases,case_count=len(cases),exact_declared269_target_plans=True,
        ancestor_closed=True,all_admitted_grants_preserved=True,unit_candidate_charges_preserved=True,
        NPU_used=False,model_executed=False,Goodput=None,adopted=False)
    with(P/'HOST_PLANS282_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
if __name__=='__main__':main()
