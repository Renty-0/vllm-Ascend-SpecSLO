"""Reproduce constructor-time loss with the original observer, then repair it."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'))
import base199
import observer392


def classes():
    graph=SimpleNamespace(_CAPTURED_TASKS=None)
    class Model:
        def __init__(self):self.hooks=[];self.calls=0
        def register_forward_hook(self,hook):self.hooks.append(hook);return hook
        def __call__(self,*args):
            self.calls+=1;hidden=object()
            for hook in self.hooks:hook(self,args,hidden)
            return hidden
    class Runner:
        def __init__(self,model,*,is_target_worker):self.model=model;self.original_qualification_marker=object()
        def _execute_target(self,inputs,positions,metadata,vocabulary_size,*,output_kind='greedy'):
            return self.model(inputs[0],positions[0],metadata[0])
    class Engine:
        def __init__(self,*,draft=True):
            self.is_draft=draft;self.model=Model()
            self.graph_runner=Runner(self.model,is_target_worker=not draft)
            r=self.graph_runner;self.marker=r.original_qualification_marker
            meta=SimpleNamespace(use_fused_infer_attention=False,attention_mask=None)
            graph._CAPTURED_TASKS=[object()]
            self.capture=r._execute_target([[1]],[None],[meta],151936)
            graph._CAPTURED_TASKS=None
            self.eager=r._execute_target([[1]],[None],[meta],151936)
    return SimpleNamespace(NativeACLGraphRunner=Runner,NativePearlEngine=Engine),graph


def main():
    ns={};base199.functions(base199.V151/'observer151.py',ns,names=('install',))
    old,oldgraph=classes();ns['install'](old,oldgraph);prior=old.NativePearlEngine()
    key=('target-greedy:151936|steps:1|paged',(1,))
    assert prior.graph_runner._v151_target_hidden.get(key)is None
    assert prior.graph_runner._v151_eager_target_hidden.get(key)is None
    new,graph=classes();observer392.install(new,graph);actual=new.NativePearlEngine()
    r=actual.graph_runner
    assert r._v151_target_hidden[key]is actual.capture
    assert r._v151_eager_target_hidden[key]is actual.eager
    assert actual.capture is not actual.eager and actual._v151_hidden_hook is r._v151_hidden_hook
    assert r.original_qualification_marker is actual.marker
    assert actual.model.calls==prior.model.calls==2 and len(actual.model.hooks)==len(prior.model.hooks)==1
    target=new.NativePearlEngine(draft=False)
    assert not target.model.hooks and not hasattr(target.graph_runner,'_v151_target_hidden')
    assert 'torch'not in sys.modules
    value=dict(status='PASS_HOST392_ACTUAL_ORIGINAL_OBSERVER_CONSTRUCTOR_REFERENCE_LOSS_REPRODUCED',
        original_failure_reproduced=True,early_capture_and_eager_references_preserved=True,
        engine_postconstructor_does_not_clear_references=True,model_call_count_unchanged=True,
        target_role_untouched=True,Native_qualification_flag_modified=False,
        NPU_used=False,Torch_imported=False,numerical_qualification=False,Raw=None,Goodput=None)
    (P/'HOST392_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
