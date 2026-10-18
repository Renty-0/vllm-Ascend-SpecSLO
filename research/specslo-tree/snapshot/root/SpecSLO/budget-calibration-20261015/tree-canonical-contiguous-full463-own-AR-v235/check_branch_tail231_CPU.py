"""Original-tree visible-KV order oracle plus original Native FIA mask ABI."""
import ast,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent;N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def main():
    import torch
    torch.set_num_threads(1)
    for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
        module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
    spec=importlib.util.spec_from_file_location('vllm_ascend.spec_decode.pearl.tree',N/'tree.py');tree=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tree;spec.loader.exec_module(tree)
    predicate=next(n for n in ast.parse((N/'native_model.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_is_contiguous_linear_tree_plan')
    module=types.ModuleType('vllm_ascend.spec_decode.pearl.native_model');module.__dict__['torch']=torch
    exec(compile('from __future__ import annotations\n'+ast.unparse(predicate),str(N/'native_model.py')+'::CPU-original-linear-predicate','exec'),module.__dict__)
    sys.modules[module.__name__]=module
    sys.path[:0]=[str(P),str(B/'tree-high-normal-cap4-native-families-v148')]
    from neural_oracle231 import cases,prepare
    from padding231 import pad
    from branch_tail231 import layout,COPY_SLOTS
    node=copy.deepcopy(next(n for n in ast.parse((N/'native_graph.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_validate_tree_fia_mask'))
    ns=dict(torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(N/'native_graph.py')+'::CPU-original-mask-validator','exec'),ns)
    fixtures=[]
    for case in cases():
        plans,rows=prepare(case);fixtures.append((case['name'],plans))
    for prefix in(0,1,124,125,126,127,128,255,511,4000,4087):
        for anchor in range(4):
            raw=tree.cached_cpu_tree_speculation_plan(2,4,prefix,4096,candidate_budget=8)
            plans=[tree.pack_selected_tree_plan(raw,[0,1,2,3,4+anchor])];fixtures.append((f'edge{prefix}-anchor{anchor}',plans))
    for anchor in range(4):
        raw=tree.cached_cpu_tree_speculation_plan(2,4,128,4096,candidate_budget=8)
        fixtures.append((f'max13forks-prefix128-anchor{anchor}',[tree.pack_selected_tree_plan(raw,[0,1,2,3,4+anchor])for _ in range(13)]))
    observations=[];branch_cases=0
    for label,plans in fixtures:
        n=len(plans);engine=types.SimpleNamespace(config=types.SimpleNamespace(max_model_len=4096),_tree_graph_scratch_sequence_ids=tuple(range(463,527)))
        pp,_,_,ids,_=pad(engine,plans,[41+i for i in range(n)],[[103+j for j in range(p.candidate_budget)]for p in plans],list(range(n)),[],[])
        tables=[[-1]*32 for _ in range(527)];next_page=0
        for owner,plan in zip(ids,pp,strict=True):
            pages=(max(plan.cache_positions.tolist())+128)//128
            tables[owner][:pages]=list(range(next_page,next_page+pages));next_page+=pages
        for owner in engine._tree_graph_scratch_sequence_ids:
            if tables[owner][0]<0:tables[owner][0]=next_page;next_page+=1
        assert next_page<=544
        actual=layout(pp,ids,tables,engine._tree_graph_scratch_sequence_ids)
        if not actual['enabled']:
            observations.append(dict(case=label,copy_enabled=False));continue
        branch_cases+=1;src=actual['source'];dst=actual['dest'];assert len(src)==len(dst)==len(set(dst))==COPY_SLOTS
        copied=dict(zip(dst,src,strict=True))
        original=[]
        for plan,owner in zip(pp,ids,strict=True):
            for row in plan.attention_mask:
                visible=(~row).nonzero().flatten().tolist()
                original.append([tables[owner][p//128]*128+p%128 for p in visible])
        seen=[]
        for group in actual['groups']:
            for j,row_id in enumerate(group['original_rows']):
                mask=group['masks'][j];visible=(~mask[:group['KV']]).nonzero().flatten().tolist()
                slots=[group['tables'][p//128]*128+p%128 for p in visible]
                values=[copied.get(s,s)for s in slots]
                assert values==original[row_id],(label,row_id)
                seen.append(row_id)
        assert seen==list(range(80))
        protected_pages={page for owner in ids for page in tables[owner]if page>=0}
        assert all(slot//128 not in protected_pages for slot in dst)
        ns['_validate_tree_fia_mask'](80,actual['mask'],torch.tensor(actual['tables'],dtype=torch.int32),actual['actual_seq_lengths_q'],actual['sequence_lens'],128)
        assert actual['mask'].shape[2]==10 and actual['real_copy_slots']<=COPY_SLOTS
        observations.append(dict(case=label,copy_enabled=True,branches=actual['branches'],real_copy_slots=actual['real_copy_slots'],branch_tail_pages=actual['branch_tail_pages'],identity_padding_pages=actual['identity_padding_pages'],attention_segments=actual['attention_segments']))
    assert 'torch_npu'not in sys.modules
    result=dict(status='CPU_exact_visible_KV_order_and_Native_ABI_passed',cases=len(fixtures),branch_cases=branch_cases,fixed_copy_slots=COPY_SLOTS,
        max_actual_copy_slots=max(x.get('real_copy_slots',0)for x in observations),all_real_and_current_dummy_prefix_pages_protected=True,
        all_copy_destinations_unique=True,original_query_order_preserved=True,original_Native_FULL_mask_validator_passed=True,
        files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),P/'branch_tail231.py',P/'padding231.py']},
        details=observations,NPU_used=False,runtime_installed=False,neural_qualified=False,performance_qualified=False)
    with(P/'CPU_BRANCH_TAIL235_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='details'}))
if __name__=='__main__':main()
