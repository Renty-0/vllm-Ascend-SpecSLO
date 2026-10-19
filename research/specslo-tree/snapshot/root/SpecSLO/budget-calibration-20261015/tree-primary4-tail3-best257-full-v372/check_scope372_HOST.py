"""Pure Python callback/scope fixtures; no Native model or graph qualification."""
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
sys.path.append(str(Path(__file__).resolve().parent.parent/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'))
import live372
import scope372

HERE=Path(__file__).resolve().parent

def main():
    output=HERE/'HOST372_SCOPE_RESULT.json'
    assert not output.exists()
    class Runner:
        def run_draft_greedy(self):pass
        def _execute_draft(self):pass
    class Engine:
        is_draft=True
        _canonical257_phase='qualified'
        _dynamic_tree_generation=5
        def __init__(self):
            self.graph_runner=Runner()
            self.graph_runner.draft_entries={}
            self.graph_runner.disabled_entry_keys=set()
        def draft_tree_forward(self,plans,*args,**kwargs):
            return dict(graph_calls=4,query_count=len(plans)*4,num_draft_tokens=[4]*len(plans),eager_frontier_tokens={})
    engine=Engine();other=Engine()
    original=Engine.draft_tree_forward
    state=SimpleNamespace(calls=[])
    def proposed(current,plans,*args,**kwargs):
        assert current is engine
        state.calls.append(dict(actual_model_steps=3))
        return dict(graph_calls=2,query_count=len(plans)*4,num_draft_tokens=[4]*len(plans),eager_frontier_tokens={})
    entries={('HOST-private-'+str(i),i+1):SimpleNamespace(runtime_validated=True)for i in range(64)}
    engine.graph_runner.draft_entries.update(entries)
    q=dict(original=original,proposed=proposed,state=state,
        proof=dict(cases=306,fresh_same_model_instance=True),entries=entries)
    native=SimpleNamespace(NativePearlEngine=Engine)
    live372.install(engine,native,q)
    assert engine.draft_tree_forward([None]*2)['graph_calls']==2
    assert engine.draft_tree_forward([None]*33)['graph_calls']==4
    assert other.draft_tree_forward([None]*2)['graph_calls']==4
    counts=q['counts']['5']
    assert counts['calls']==2 and counts['tail_applied']==counts['fallback']==1
    assert counts['larger_than32_fallback']==1 and counts['max_logical_owners']==33
    class Inherited:
        def __init__(self):self.calls=0;self.valid=True
        def require(self,current):
            self.calls+=1
            return frozenset()if self.valid else frozenset({'HOST-reject'})
    inherited=Inherited()
    scope=scope372.Scope(engine,inherited,q,scope372._ISSUER)
    modules={}
    for name in ['vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl']:
        module=ModuleType(name);module.__path__=[];modules[name]=module
    module=ModuleType('vllm_ascend.spec_decode.pearl.native_engine')
    module.NativePearlEngine=Engine
    modules[module.__name__]=module
    modules['vllm_ascend.spec_decode.pearl'].native_engine=module
    rejected=[]
    def reject(label):
        try:scope.require(engine)
        except RuntimeError:rejected.append(label)
        else:raise AssertionError('invalid callback/scope accepted: '+label)
    with patch.dict(sys.modules,modules):
        assert scope.require(engine)==frozenset()and inherited.calls>0
        inherited.valid=False;reject('original_scope');inherited.valid=True
        Engine.draft_tree_forward=original;reject('draft_callback');Engine.draft_tree_forward=q['forward']
        key=next(iter(entries));entry=engine.graph_runner.draft_entries.pop(key)
        reject('resident_entry_identity');engine.graph_runner.draft_entries[key]=entry
        old=entries[key];bad=SimpleNamespace(runtime_validated=False)
        entries[key]=bad;engine.graph_runner.draft_entries[key]=bad
        reject('natural_qualification_required');entries[key]=old;engine.graph_runner.draft_entries[key]=old
        engine.graph_runner.disabled_entry_keys.add(key);reject('disabled_entry');engine.graph_runner.disabled_entry_keys.clear()
        q['proof']['cases']=305;reject('missing_neural_cases');q['proof']['cases']=306
        assert scope.require(engine)==frozenset()
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST_actual_scope_callback_identity_and_bounded_adoption',
        larger_than32_uses_original=True,other_instances_use_original=True,
        original_scope_delegate_exercised=True,rejected_mutations=rejected,
        purely_fictional_scope_objects=True,Native_graph_or_model_created=False,
        NPU_used=False,model_qualified=False,Raw=None,Goodput=None)
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
