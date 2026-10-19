"""Replay the actual Native eager constructor through mixed candidate selection."""
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

P=Path(__file__).resolve().parent
B=P.parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')

def main():
    sys.path[:0]=[str(P),str(B/'tree-deferred-FULL-mask-host-projection-v259'),str(B/'tree-alternate-chain-target-host-v269')]
    from fixture259 import build
    env=build();tree=env['tree'];torch=env['torch']
    spec=importlib.util.spec_from_file_location('vllm_ascend.spec_decode.pearl.tree_budget',N/'tree_budget.py')
    dto=importlib.util.module_from_spec(spec);sys.modules[spec.name]=dto;spec.loader.exec_module(dto)
    from plans333 import raw,select
    from plan269 import make
    source=(N/'native_engine.py').read_text()
    native=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine')
    method=next(n for n in native.body if isinstance(n,ast.FunctionDef)and n.name=='_spec_rhythm_tree_eager_plan')
    ns=dict(cached_cpu_tree_speculation_plan=tree.cached_cpu_tree_speculation_plan)
    exec(compile('from __future__ import annotations\n'+ast.unparse(method),str(N/'native_engine.py')+'::original-eager-constructor','exec'),ns)
    engine=types.SimpleNamespace(config=types.SimpleNamespace(spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4,max_model_len=4096),
        _tree_primary_path_length=lambda plan:len(tree.tree_primary_path(plan)))
    cases=[]
    for prefix in (0,127,128,1505,2555):
        parent=make(3,prefix,0)
        state=types.SimpleNamespace(token_ids=[0]*(prefix+1))
        eager=ns['_spec_rhythm_tree_eager_plan'](engine,parent,state,3)
        assert tuple(eager.parent_indices.tolist())==(-1,0,1,2,-1,0,1,2)
        assert eager.prefix_len==prefix+4
        requests={}
        for b in range(1,9):
            plan=raw(b,prefix)
            requests[b]=dto.TreeCandidateRequest(parents=plan.parent_indices.tolist(),token_ids=list(range(8)),
                conditional_confidences=[.7]*8,max_candidates=b,minimum_candidates=1)
        requests[72]=dto.TreeCandidateRequest(parents=eager.parent_indices.tolist(),token_ids=list(range(8)),
            conditional_confidences=[.6]*8,max_candidates=3,minimum_candidates=0)
        selected=select(requests,39)
        assert selected.selected_indices[72]==(0,1,2)
        packed=tree.pack_selected_tree_plan(eager,selected.selected_indices[72])
        expected=make(3,eager.prefix_len,0)
        for name in ('parent_indices','positions','cache_positions','attention_mask'):
            assert torch.equal(getattr(packed,name),getattr(expected,name)),name
        assert selected.total_candidates==39
        for b in range(1,9):assert selected.selected_indices[b]==tuple(range(b))
        # The previously frozen selector must fail on the exact same inputs.
        old=B/'tree-dynamic-root-alternate-paired-prefix-grid-full-v284/plans284.py'
        old_spec=importlib.util.spec_from_file_location('_ROOT333_old_selector',old)
        old_module=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old_module)
        try:old_module.select(requests,39)
        except ValueError as exc:assert 'exact primary4/root-alternate4 topology' in str(exc)
        else:raise AssertionError('old full463 failure was not reproduced')
        for cap,minimum in ((4,0),(8,0),(3,1)):
            bad=dto.TreeCandidateRequest(parents=eager.parent_indices.tolist(),token_ids=list(range(8)),
                conditional_confidences=[.6]*8,max_candidates=cap,minimum_candidates=minimum)
            try:select({72:bad},96)
            except ValueError:pass
            else:raise AssertionError('unqualified eager topology allowed')
        cases.append(dict(prefix=prefix,eager_prefix=eager.prefix_len,normal_grants=list(range(1,9)),
            eager_grant=3,old_failure_reproduced=True,packed_eager_exact_declared_chain=True))
    result=dict(status='HOST_actual_Native_eager3_and_normalB1to8_mixed_selection_regression_passed',cases=cases,
        original_eager_method_sha256=hashlib.sha256(ast.unparse(method).encode()).hexdigest(),
        old_failure_reproduced_all_prefixes=True,unqualified_eager_shapes_rejected=True,
        original_eager_candidate_charges_and_prefix_preserved=True,NPU_used=False,Goodput=None)
    with(P/'HOST_EAGER333_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
