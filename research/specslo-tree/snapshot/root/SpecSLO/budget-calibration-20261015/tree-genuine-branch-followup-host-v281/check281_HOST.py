"""Original missing-suffix and packed-root protocol after alternate commits."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
    sys.path.insert(0,str(B/'tree-high-normal-cap4-native-families-v148'))
    from fast_tree_draft_v5 import missing_committed_suffix
    from root_seed import layout
    from history281 import advance
    cases=[]
    for n in(1,4,8,12):
        for mode in(0,127,128,1505,2550,'mixed'):
            for accepted in range(5):
                entries=[];controls=[]
                for owner in range(n):
                    p=(1505 if owner%2==0 else 2550)if mode=='mixed'else mode
                    primary=[10,20,30,40];raw=primary+[50,60,70,80];root=99;corrected=88
                    control=advance([103+owner]*p,root,raw,accepted,corrected)
                    state=dict(safe_before=p,tokens={p+j:v for j,v in enumerate([root]+primary[:3])})
                    start,repair=missing_committed_suffix(state,control['new_prefix'],control['history'][-5:])
                    assert start==control['first_missing']and repair==control['repair_tokens']
                    assert len(repair)==accepted and control['root_seed_required']is(accepted>0)
                    assert control['new_prefix']<=2555
                    entries.append((owner,control['new_prefix'],start,repair,corrected));controls.append(control)
                packed=layout(entries,4096)
                assert packed.repair_queries==n*accepted and len(packed.tokens)==n*(accepted+1)
                assert packed.root_indices==tuple((i+1)*(accepted+1)-1 for i in range(n))
                assert packed.tokens==tuple(t for c in controls for t in c['packed_inputs'])
                assert packed.positions==tuple(t for c in controls for t in c['packed_positions'])
                assert packed.sequence_ids==tuple(i for i in range(n)for _ in range(accepted+1))
                cases.append(dict(N=n,mode=mode,accepted=accepted,repair_queries=packed.repair_queries,
                    first_physical_real_rows=len(packed.tokens),next_prefixes=[c['new_prefix']for c in controls]))
    text=(P/'followup281.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='one')
    for item in ast.walk(node):
        if isinstance(item,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=item.targets if isinstance(item,ast.Assign)else[item.target]
            assert all(not(isinstance(t,ast.Attribute)and t.attr=='_dynamic_tree_known_KV')for t in targets)
    for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
    assert not any(name.split('.')[0]in('torch','torch_npu')for name in sys.modules)
    result=dict(status='HOST_original_deferred_suffix_and_packed_root_followup120_passed',
        case_count=len(cases),cases=cases,original_missing_suffix_function_executed=True,
        original_root_seed_layout_executed=True,branch_reject0_and_accepted1to4=True,
        transition_does_not_manually_change_candidate_ledger=True,
        teacher_forced_commit_controls_only=True,
        actual_NPU_followup_cases_required=20,NPU_used=False,
        real_model_qualified=False,Goodput=None,adopted=False)
    with(P/'HOST281_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
if __name__=='__main__':main()
