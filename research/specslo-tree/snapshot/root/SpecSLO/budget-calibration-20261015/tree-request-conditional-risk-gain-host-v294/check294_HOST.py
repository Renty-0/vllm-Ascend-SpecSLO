"""Actual Native greedy verdicts test censoring, request separation and reset."""
import ast,collections,copy,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path.append(str(B/'tree-high-normal-cap4-native-families-v148'))
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
def main():
    import torch
    torch.set_num_threads(1)
    node=copy.deepcopy(next(n for n in ast.walk(ast.parse(N.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='_verify_tree_outputs_host'))
    node.decorator_list=[]
    ns=dict(torch=torch,TreeVerificationOutput=collections.namedtuple('OriginalVerdict','outputs accepted'))
    exec(compile('from __future__ import annotations\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))),str(N)+'::original-verifier','exec'),ns)
    def world(b,kind,k=0):
        parents=[-1,0,1,2,-1,4,5,6][:b];tokens=[10*(i+1)for i in range(b)];pred=[99]*(b+1)
        if kind=='primary':
            for depth in range(k):pred[depth]=10*(depth+1)
        elif kind=='alternate':
            pred[0]=50
            for depth in range(k-1):pred[5+depth]=60+10*depth
        else:assert kind=='other'
        plan=types.SimpleNamespace(candidate_budget=b,parent_indices=torch.tensor(parents))
        verdict=ns['_verify_tree_outputs_host']([tokens],[plan],torch.tensor(pred),4,torch.device('cpu'))
        return dict(parents=parents,accepted=verdict.accepted[0].tolist(),outputs=verdict.outputs[0].tolist(),remaining_before=255)
    from budget_policy import TreeBudgetPolicy
    import risk294 as risk
    values={'1':1.5,'2':2.,'3':2.5,'4':3.,'5':3.25}
    model=dict(tree_depth=4,curves={c:dict(values)for c in('coding','chat','summarization')})
    def req(i):return types.SimpleNamespace(request_index=i,slo_class='tight',output_token_budget=256,delivered_tokens=1)
    def policy():
        p=TreeBudgetPolicy(model);risk.begin_generation(p,1);return p
    # Every exposed primary/alternate path, including original eager B3.
    path_cases=0
    for b in range(1,9):
        for k in range(min(4,b)+1):
            event=world(b,'primary',k);bb,c,g,side,shadow=risk.events(event['parents'],event['accepted'],event['outputs'])
            assert bb==b and g==k+1 and side==0
            assert c['samples'][:4]==[float(j<=min(b,4)and k>=j-1)for j in range(1,5)]
            assert c['hits'][:4]==[float(j<=min(b,4)and k>=j)for j in range(1,5)]
            if b<4:assert shadow is None
            path_cases+=1
        if b>=5:
            for k in range(1,b-3):
                event=world(b,'alternate',k);bb,c,g,side,shadow=risk.events(event['parents'],event['accepted'],event['outputs'])
                assert (g,side,shadow)==(k+1,k,1)
                assert c['samples'][4:]==[float(j<=b-4 and k>=j-1)for j in range(1,5)]
                path_cases+=1
    # Identical class, very different request-specific marginal tree gain.
    p=policy();easy=req(0);hard=req(1)
    for i in range(96):
        risk.observe(p,easy,**world(8,'primary',4));risk.observe(p,hard,**world(8,'alternate',4))
    eg=risk.prediction(p,easy);hg=risk.prediction(p,hard)
    assert eg['g'][4]>4.5 and eg['g'][8]-eg['g'][4]<.25
    assert hg['g'][4]<1.5 and hg['g'][8]-hg['g'][4]>2.8
    # A request contributes zero observations to its own class prior.
    alone=policy();r=req(8)
    for _ in range(48):risk.observe(alone,r,**world(8,'primary',4))
    assert risk.prediction(alone,r)['other_request_class_probabilities']==risk.priors(alone.base['tight'])
    # Short B5 reveals no continuation trials, whether its alternate succeeds
    # or fails. Prior tails remain unchanged instead of learning false zeros.
    short=policy();r=req(5)
    for kind,k,count in [('primary',4,160),('other',0,80),('alternate',1,80)]:
        event=world(5,kind,k)
        for _ in range(count):risk.observe(short,r,**event)
    state=risk.snapshot(short)
    assert state['classes']['tight']['samples'][5:]==[0.]*3
    assert state['requests']['5']['recent']['samples'][5:]==[0.]*3
    assert risk.prediction(short,r)['conditional_probabilities'][5:]==[.5]*3
    # Recent hazards adapt after a phase change within one request.
    shifted=policy();r=req(9)
    for _ in range(96):risk.observe(shifted,r,**world(8,'primary',4))
    before=shifted.gain(r,4)
    for _ in range(96):risk.observe(shifted,r,**world(8,'alternate',4))
    after=shifted.gain(r,4)
    assert before>4.5 and after<2.0 and shifted.gain(r,8)>4.5
    # Terminal horizons do not train either primary or alternate hazards.
    before_state=copy.deepcopy(risk.snapshot(shifted));before_observed=copy.deepcopy(shifted.observed)
    event=world(8,'alternate',4);event['remaining_before']=4
    risk.observe(shifted,r,**event)
    after_state=risk.snapshot(shifted)
    assert shifted.observed==before_observed and after_state['classes']==before_state['classes']
    assert after_state['requests']==before_state['requests']
    q=policy();r=req(7);event=world(4,'primary',4)
    risk.observe_primary(q,r,**event);assert ('tight',4)not in q.observed
    q.observe(r,4,5);assert q.observed['tight',4]==(1,5)
    try:risk.begin_generation(q,2)
    except RuntimeError:pass
    else:raise AssertionError('missing original reset allowed')
    q.observed.clear();risk.begin_generation(q,2);assert risk.snapshot(q)['requests']=={}
    try:risk.begin_generation(q,1)
    except ValueError:pass
    else:raise AssertionError('stale generation accepted')
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_original_Native_verdicts_request_conditional_censoring_and_phase_adaptation_passed',
        legal_path_cases=path_cases,original_verifier_executed=True,easy_g4=eg['g'][4],easy_g8=eg['g'][8],
        hard_g4=hg['g'][4],hard_g8=hg['g'][8],same_class_requests_separated=True,
        class_prior_excludes_own_raw_counts=True,unobserved_continuations_not_failed=True,
        within_request_phase_change_g4=[before,after],original_actual_budget_observe_once=True,
        terminal_excluded=True,fresh_reset_and_stale_generation_checked=True,
        synthetic_fixture_not_machine_measurement=True,NPU_used=False,runtime_installed=False,Goodput=None)
    with(P/'HOST294_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
