"""Check actual cloned callback resolution and reject invalid local bindings."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P=Path(__file__).parent;B=P.parent
for folder in ('tree-bounded-parent4-shared4-eager-diagnostic-v199',
        'tree-parent4-best376-target-binding-host-v390',
        'tree-static-q5-M120-node-math-v255','tree-static-q5-immutable-index-cache-host-v247',
        'tree-best257-uniform-KV-buckets-full-v376'):
    sys.path.insert(0,str(B/folder))
import base199
import actor199,actor391,target199,target391,binding391
from metadata390 import physical


def main():
    for name in ('padded','clone_output','original_call','own_compact','rebuild_history','qualify'):
        fn=getattr(target391,name);before=getattr(target199,name)
        if name!='original_call':
            assert fn.__code__ is before.__code__ and fn.__closure__ is before.__closure__
        if name=='original_call':continue
        assert fn.__globals__['metadata']is physical
        assert fn.__globals__['target_shape']is target391.target_shape
        assert fn.__globals__['original_call']is target391.original_call
        assert fn.__globals__['rebuild_history']is target391.rebuild_history
    import inspect
    assert inspect.getclosurevars(target391.original_call).nonlocals['base'].__code__ is target199.original_call.__code__
    assert actor391.run.__globals__['run_case'].__globals__['target']is target391
    assert actor391.run.__globals__['run_case'].__code__ is actor199.run_case.__code__
    for n in range(1,25):
        value=target391.target_shape([4]*n)
        assert value['M']==120 and value['S']==24 and len(value['scratch'])==24-n
        assert value['real_Q']==5*n and value['C']==4*n
    for budgets in ([],[4]*25,[3],[5],[True]):
        try:target391.target_shape(budgets)
        except RuntimeError:pass
        else:raise AssertionError('out of scope shape')
    rejected=[]
    for case in ('no_baseline','live_cache','sealed'):
        engine=SimpleNamespace(_v173_qualified_component=None if case=='no_baseline'else {'real_fresh':True},
            cache_allocation=object()if case=='live_cache'else None,
            graph_runner=SimpleNamespace(graph_cache_sealed=case=='sealed'))
        try:binding391.install(engine,SimpleNamespace())
        except RuntimeError as error:rejected.append(str(error))
        else:raise AssertionError('invalid local target activation')
    assert 'torch'not in sys.modules
    result=dict(status='PASS_HOST391_ACTUAL_CALLBACK_GLOBALS_AND_SCOPE_REJECTIONS',
        target_proof_code_objects_preserved=True,actual_actor_target_delegate=True,
        actual_full_history_and_original_numeric_matcher_preserved=True,
        original173_cache_seal_rejections=rejected,shape_cases=24,
        NPU_used=False,Torch_imported=False,model_qualification=False,
        production_permission=False,Raw=None,Goodput=None)
    (P/'HOST391_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
