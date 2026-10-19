"""Actual installed wrapper, AST inverse and separate-frontier CPU oracle."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
LEGACY=BASE/'tree-high-normal-cap4-native-families-v148'
sys.path.append(str(LEGACY))
sys.path.append(str(BASE/'tree-primary4-top2-chain-host-v364'))

def main():
    import torch
    torch.set_num_threads(1)
    from check364_HOST import ModelDouble,SOURCE as GRAPH_SOURCE
    from spine373 import execute,SCALE
    from forward373 import installed_fast,project_fast
    import root_seed,fast_tree_draft_v5
    class Runner:
        def _execute_draft(self,*args,**kwargs):pass
    class Engine:
        def _allocate_cache(self,*args,**kwargs):pass
        def draft_tree_forward(self,*args,**kwargs):pass
    native=SimpleNamespace(NativePearlEngine=Engine,NativeACLGraphRunner=Runner)
    root_seed.install(native)
    legacy=Engine.draft_tree_forward
    fast=installed_fast(legacy)
    projected=project_fast(fast,lambda*args:None,lambda*args:None,lambda*args:None)
    assert callable(projected)
    try:installed_fast(Engine.__init__)
    except RuntimeError:pass
    else:raise AssertionError('unrelated entry accepted')
    graph=ast.parse(GRAPH_SOURCE.read_text())
    matcher_node=copy.deepcopy(next(n for n in ast.walk(graph)if isinstance(n,ast.FunctionDef)and n.name=='_draft_outputs_match'))
    matcher_node.decorator_list=[]
    namespace=dict(torch=torch)
    exec(compile('from __future__ import annotations\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=[matcher_node],type_ignores=[]))),str(GRAPH_SOURCE)+'::original-unmodified-checker','exec'),namespace)
    matcher=namespace['_draft_outputs_match']
    cases=0;different_argmax_from_top2=0
    for logical in (1,4,12,24,32):
        physical=1<<(logical-1).bit_length()
        for vocabulary in (32,151936):
            for steps in (1,2,3,4):
                def model():
                    m=ModelDouble(torch,vocabulary)
                    m.compute_greedy_tokens=lambda hidden,vocabulary:m.compute_logits(hidden)[:,:vocabulary].argmax(dim=-1)
                    return m
                actual_model,reference_model=model(),model()
                roots=torch.arange(physical,dtype=torch.long)%vocabulary
                positions=[torch.full((physical,),127+i,dtype=torch.long)for i in range(steps+1)]
                slots=torch.arange(physical);slots[logical:]=-1
                metas=[SimpleNamespace(slot_mapping=slots.clone())for _ in positions]
                actual=execute(actual_model,roots,positions,metas,vocabulary)
                hidden=reference_model(roots,positions[0],metas[0])
                first=reference_model.compute_greedy_tokens(hidden,vocabulary)
                root=torch.stack((first*(SCALE+1),torch.zeros_like(first)),dim=-1)[:,None,:]
                tail=fast_tree_draft_v5.execute_top2(SimpleNamespace(model=reference_model),torch.where(slots>=0,first,torch.zeros_like(first)),positions[1:],metas[1:],vocabulary,return_confidence=True)
                expected=torch.cat((root,tail),dim=1)
                assert torch.equal(actual,expected)
                assert all(torch.equal(a,b)for a,b in zip(actual_model.inputs,reference_model.inputs,strict=True))
                assert matcher(actual,expected,metas,logical)
                changed=actual.clone();changed[0,0,0]^=1
                assert not matcher(changed,expected,metas,logical)
                cases+=1
    assert 'torch_npu'not in sys.modules
    output=HERE/'HOST373_RESULT.json';assert not output.exists()
    value=dict(status='PASS_HOST_actual_RootSeed_FastV5_closure_and_fused_frontier_chain_oracle',
        cases=cases,frontier_argmax_then_original_top2_exact=True,
        all_dependent_CPU_model_inputs_equal=True,original_int64_checker_used=True,
        complete_actual_FastV5_function_AST_inverse=True,normal_and_mixed_fallback_design=True,
        model_precision_not_changed=True,original_confidence_encoding_retained=True,
        NPU_used=False,model_qualified=False,Raw=None,Goodput=None,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in HERE.glob('*.py')})
    with output.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))
if __name__=='__main__':main()
