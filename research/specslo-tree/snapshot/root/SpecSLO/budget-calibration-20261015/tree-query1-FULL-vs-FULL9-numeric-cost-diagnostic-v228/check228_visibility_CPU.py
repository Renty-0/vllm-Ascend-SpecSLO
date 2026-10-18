"""Independent original-tree visibility/page checks for query1 metadata."""
import ast,copy,importlib.util,json,sys,types
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Any,Sequence
P=Path(__file__).resolve().parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def need(x,m):
    if not x:raise AssertionError(m)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    import torch
    torch.set_num_threads(1)
    tree=load('_ROOT228_original_tree',N/'tree.py')
    ns=dict(torch=torch,dataclass=dataclass,replace=replace,Any=Any,Sequence=Sequence)
    nodes=[copy.deepcopy(n)for n in ast.parse((N/'native_model.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='NativeAttentionMetadata']
    nodes.extend(copy.deepcopy(n)for n in ast.parse((N/'native_graph.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_validate_tree_fia_mask')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(N)+'::original-CPU','exec'),ns)
    from canonical228 import canonical
    cases=[]
    for position in(0,127,128,511,4000):
        raw=tree.cached_cpu_tree_speculation_plan(2,4,position,4096,candidate_budget=8)
        selected=[list(range(b))for b in range(1,9)]+[[0,1,2,3,4+a]for a in range(4)]
        for ids in selected:
            plan=tree.pack_selected_tree_plan(raw,ids);physical=plan.cache_positions.tolist();q=len(physical)
            tables=torch.arange(32,dtype=torch.int32).reshape(1,32)
            original=ns['NativeAttentionMetadata'](slot_mapping=torch.tensor(physical,dtype=torch.int32),context_lens=torch.tensor([p+1 for p in physical],dtype=torch.int32),
                block_tables=tables,request_block_tables=tables,actual_seq_lengths_q=(q,),sequence_lens=(max(physical)+1,),
                tree_attention=True,use_fused_infer_attention=True,tree_attention_mask=plan.attention_mask[None,None,:,:])
            result=canonical(original)
            need(result.actual_seq_lengths_q==tuple(range(1,q+1)),'exact one-token segmentation')
            need(result.sequence_lens==tuple(p+1 for p in physical),'own physical extents')
            need(result.slot_mapping is original.slot_mapping and result.context_lens is original.context_lens,'unchanged KV write coordinates')
            for row,n in enumerate(result.sequence_lens):
                mask=result.tree_attention_mask[row,0,0]
                need(torch.equal(mask,plan.attention_mask[row]),'exact original complete visibility mask')
                need(not bool((~mask[n:]).any()),'no visible ancestor/query truncated')
                need(not bool(mask[physical[row]]),'own query remains visible')
                need(torch.equal(result.request_block_tables[row],tables[0]),'same exact physical cache pages')
            ns['_validate_tree_fia_mask'](q,result.tree_attention_mask,result.request_block_tables,result.actual_seq_lengths_q,result.sequence_lens,128)
            cases.append(dict(position=position,selected=ids,q=q))
    invalid=[]
    for label,meta in [('not_FULL',replace(original,tree_attention=False)),('missing_partition',replace(original,actual_seq_lengths_q=())),('query_exceeds_mask',replace(original,actual_seq_lengths_q=(q+1,))),('invalid_context',replace(original,context_lens=torch.zeros(q,dtype=torch.int32)))]:
        try:canonical(meta)
        except ValueError:invalid.append(label)
        else:raise AssertionError('invalid metadata accepted '+label)
    need('torch_npu'not in sys.modules,'strictly CPU only')
    out=P/'CPU228_VISIBILITY_RESULT.json';need(not out.exists(),'fresh independent output')
    out.write_text(json.dumps(dict(status='passed',cases=len(cases),negative_cases=invalid,details=cases,
        original_Native_mask_validator_passed=True,original_linear_and_all_four_side_anchor_masks_preserved=True,
        Torch_CPU=True,NPU_used=False,neural_qualified=False,performance_qualified=False),indent=2)+'\n')
    print(json.dumps(dict(status='passed',cases=len(cases),negative_cases=len(invalid),NPU_used=False)))
if __name__=='__main__':main()
