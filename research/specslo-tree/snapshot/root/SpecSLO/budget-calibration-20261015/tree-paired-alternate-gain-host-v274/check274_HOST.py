"""Original greedy-verdict proof of alternate-chain marginal gains.

All predictions here are synthetic HOST cases. They provide no empirical
model acceptance curve, component timing or serving-performance evidence.
"""
import ast,collections,copy,hashlib,itertools,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
SOURCE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
def main():
    import torch
    torch.set_num_threads(1)
    text=SOURCE.read_text();nodes=[]
    for name in('_verify_tree_outputs_host','_tree_row_requires_kv_compaction'):
        node=copy.deepcopy(next(n for n in ast.walk(ast.parse(text))if isinstance(n,ast.FunctionDef)and n.name==name))
        node.decorator_list=[];nodes.append(node)
    ns=dict(torch=torch,TreeVerificationOutput=collections.namedtuple('CPUOriginalVerdict','outputs accepted'))
    exec(compile('from __future__ import annotations\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))),str(SOURCE)+'::unchanged-original-greedy','exec'),ns)
    verify=ns['_verify_tree_outputs_host'];counts=collections.Counter();witnesses={}
    def call(tokens,parents,predictions):
        plan=types.SimpleNamespace(candidate_budget=len(parents),parent_indices=torch.tensor(parents))
        return verify([tokens],[plan],torch.tensor(predictions),4,torch.device('cpu'))
    for b in range(5,9):
        for anchor in range(9-b):
            parents=[-1,0,1,2,anchor-1]+list(range(4,b-1))
            tokens=[10*(i+1)for i in range(b)]
            options=[[tokens[i]for i,p in enumerate(parents)if p==owner]+[99,100]for owner in range(-1,b)]
            for predictions in itertools.product(*options):
                verdict=call(tokens,parents,predictions);base=call(tokens[:4],parents[:4],predictions[:5])
                path=[x for x in verdict.accepted[0].tolist()if x>=0]
                actual=[x for x in verdict.outputs[0].tolist()if x>=0]
                old=[x for x in base.outputs[0].tolist()if x>=0]
                k=sum(node>=4 for node in path)
                assert len(actual)-len(old)==k and actual[:len(old)]==old
                for m in range(1,4):
                    shorter=call(tokens[:m],parents[:m],predictions[:m+1])
                    out=[x for x in shorter.outputs[0].tolist()if x>=0]
                    assert len(out)==min(len(old),m+1)and old[:len(out)]==out
                    counts['primary_counterfactual_verdicts']+=1
                for m in range(5,b+1):
                    truncated=call(tokens[:m],parents[:m],predictions[:m+1])
                    out=[x for x in truncated.outputs[0].tolist()if x>=0]
                    assert len(out)==len(old)+min(k,m-4) and actual[:len(out)]==out
                    counts['counterfactual_verdicts']+=1
                if k<b-4:
                    assert all(not(k>=j)for j in range(b-3,5))
                counts['original_prediction_worlds']+=1
                counts[f'B{b}-anchor{anchor}-side{k}']+=1
                witnesses.setdefault(f'{b}:{anchor}:{k}',dict(B=b,anchor=anchor,parents=parents,
                    accepted=verdict.accepted[0].tolist(),outputs=verdict.outputs[0].tolist(),
                    same_prefix_B4=len(old),accepted_side_nodes=k))
    sys.path.insert(0,str(B/'tree-high-normal-cap4-native-families-v148'))
    from budget_policy import TreeBudgetPolicy
    from paired274 import begin_generation,observe,observe_primary,snapshot
    # Fixture curves initialize the original policy, not a measured machine g.
    model=dict(curves={c:{str(i):min(5.,1.+.5*i)for i in range(1,9)}
        for c in('coding','chat','summarization')})
    policy=TreeBudgetPolicy(model);begin_generation(policy,1)
    requests={c:types.SimpleNamespace(slo_class=c,output_token_budget=256,delivered_tokens=1)
        for c in('tight','normal','loose')}
    for cls,request in requests.items():
        for witness in witnesses.values():
            row=observe(policy,request,parents=witness['parents'],accepted=witness['accepted'],
                outputs=witness['outputs'],remaining_before=255)
            assert row['same_prefix_B4']==witness['same_prefix_B4']
            assert row['accepted_side_nodes']==witness['accepted_side_nodes']
        curve=[policy.gain(request,i)for i in range(4,9)]
        assert all(a<=z for a,z in zip(curve,curve[1:]))and curve[-1]<=5
        before=copy.deepcopy(policy.observed)
        w=witnesses['8:0:4']
        assert observe(policy,request,parents=w['parents'],accepted=w['accepted'],outputs=w['outputs'],remaining_before=4)is None
        assert policy.observed==before
    # A short fully accepted side path cannot label deeper success as zero.
    fresh=TreeBudgetPolicy(model);begin_generation(fresh,1);w=witnesses['5:0:1']
    observe(fresh,requests['tight'],parents=w['parents'],accepted=w['accepted'],outputs=w['outputs'],remaining_before=255)
    layers=snapshot(fresh)['by_class_anchor']['tight:0']['layers']
    assert layers[0]==dict(samples=1,hits=1)and layers[1:]==[dict(samples=0,hits=0)]*3
    try:begin_generation(fresh,2)
    except RuntimeError:pass
    else:raise AssertionError('warm/previous observations leaked into new generation')
    fresh.observed.clear() # Simulate the original fresh_decode boundary.
    begin_generation(fresh,2)
    assert snapshot(fresh)['by_class_anchor']=={}
    try:begin_generation(fresh,1)
    except ValueError:pass
    else:raise AssertionError('stale generation accepted')
    primary=TreeBudgetPolicy(model);begin_generation(primary,1)
    for k in range(5):
        accepted=list(range(k))+[-1]*(4-k);outputs=list(range(1,k+2))+[-1]*(4-k)
        old4=primary.observed.get(('tight',4))
        result=observe_primary(primary,requests['tight'],parents=[-1,0,1,2],
            accepted=accepted,outputs=outputs,remaining_before=255)
        assert result=={str(n):min(k+1,n+1)for n in range(1,4)}
        assert primary.observed.get(('tight',4))==old4
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_original_B5to8_paired_counterfactual_and_censoring_proved',
        original_source=str(SOURCE),original_source_SHA=hashlib.sha256(text.encode()).hexdigest(),
        counts=dict(counts),witnesses=witnesses,
        identity='g_B = same-prefix g4 + number of accepted alternate nodes; g_m = g4 + min(accepted alternate nodes,m-4) for4<m<=B',
        exact_original_verifier_body=True,all_qualified_anchors=True,
        right_censoring_preserved=True,terminal_horizon_excluded=True,
        actual_generation_reset_and_stale_rejection=True,
        root_predictor_excludes_other_anchor_samples=True,
        HOST_policy_fixture_not_empirical=True,NPU_used=False,
        empirical_g_determined=False,production_installed=False,Goodput=None,adopted=False)
    with(P/'HOST274_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('witnesses','counts')},ensure_ascii=False))
    print(json.dumps(dict(counts)))
if __name__=='__main__':main()
