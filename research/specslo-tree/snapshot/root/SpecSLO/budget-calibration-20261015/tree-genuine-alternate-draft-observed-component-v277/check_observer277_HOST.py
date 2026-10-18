"""Regress the missing-observer failure using the unchanged original hook."""
import ast,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(B/'tree-high-normal-cap4-native-families-v148'))
    import fast_tree_draft_v5 as fast,tail_hidden_observer as observer
    original=fast.execute_top2;graph=types.SimpleNamespace(_CAPTURED_TASKS=None)
    class ArithmeticFixture:
        def __call__(self,tokens,pos,meta):
            return torch.stack((tokens.float()+pos.float(),tokens.float()-pos.float(),pos.float()+1,torch.ones_like(pos).float()),dim=1)
        def compute_logits(self,hidden):
            return torch.cat((hidden,-hidden),dim=1)
    checks=[]
    for n in(1,2,16):
        for steps in(1,2,3,4):
            key=(f'draft-greedy:8|steps:{steps}|paged|confidence',n)
            runner=types.SimpleNamespace(model=ArithmeticFixture(),last_draft_entry_key=key)
            inputs=torch.arange(n,dtype=torch.long)
            positions=[torch.arange(n,dtype=torch.long)+3+s for s in range(steps)];meta=[None]*steps
            expected=original(runner,inputs,positions,meta,8,return_confidence=True)
            try:observer.snapshot(runner,key,steps)
            except RuntimeError:pass
            else:raise AssertionError('missing original observer not rejected')
            fast.execute_top2=original;observer.install(graph)
            graph._CAPTURED_TASKS=None
            eager=fast.execute_top2(runner,inputs,positions,meta,8,return_confidence=True)
            graph._CAPTURED_TASKS=[]
            captured=fast.execute_top2(runner,inputs,positions,meta,8,return_confidence=True)
            actual,intervals,eager_intervals=observer.snapshot(runner,key,steps)
            assert len(actual)==len(intervals)==len(eager_intervals)==steps
            assert torch.equal(expected,eager)and torch.equal(expected,captured)
            assert all(torch.equal(a,z)for a,z in zip(actual,runner._v35_last_eager_tail_hidden,strict=True))
            checks.append(dict(N=n,steps=steps,original_encoding_unchanged=True,all_hidden_available=True))
    fast.execute_top2=original
    tree=ast.parse((P/'runtime277.py').read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
    installed=[n.lineno for n in ast.walk(worker)if isinstance(n,ast.Call)and ast.unparse(n.func)=='tail_hidden_observer.install']
    constructors=[n.lineno for n in ast.walk(worker)if isinstance(n,ast.Call)and ast.unparse(n.func)=='native.NativePearlEngine']
    assert len(installed)==len(constructors)==1 and installed[0]<constructors[0]
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_original_observer_capture_eager_snapshot_regression_passed',
        cases=checks,case_count=12,missing_observer_negative_controls_rejected=True,
        original_hook_body_executed=True,observer_installed_before_any_real_constructor=True,
        CPU_arithmetic_fixture_only=True,NN_model_used=False,NPU_used=False,
        Goodput=None,adopted=False)
    with(P/'HOST_OBSERVER277_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
if __name__=='__main__':main()
