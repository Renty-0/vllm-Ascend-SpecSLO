"""M128 complete row mapping and actual branch KV visibility, CPU only."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,copy,dataclasses,hashlib,importlib.util,itertools,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path[:0]=[str(B/'tree-static-q5-immutable-index-cache-host-v247')]
from mapper249 import envelope
from projection249 import plan,transform

accepted=rejected=0
for n in range(1,25):
    for budgets in itertools.combinations_with_replacement(range(1,6),n):
        if sum(budgets)>96:continue
        real=n+sum(budgets);m,s,padding=envelope(n,real,'full9'if 5 in budgets else'causal')
        sizes=[]
        for b in budgets:sizes.extend([5,1]if b==5 else[b+1])
        sizes.extend(padding);ends=[];offset=0
        for q in sizes:offset+=q;ends.append(offset)
        supported=n+budgets.count(5)<=25
        try:value=plan(ends,real)
        except ValueError:
            assert not supported;rejected+=1;continue
        assert supported and sorted(value['physical_to_original'])==list(range(128))
        assert [value['physical_to_original'][i]for i in value['original_to_physical']]==list(range(128))
        assert all(value['physical_to_original'][v['new_row']]==v['old_row']for v in value['real_moves'])
        assert all(v['new_row']<125 for v in value['real_moves'])
        accepted+=1

import torch
torch.set_num_threads(1)
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
    mod=types.ModuleType(name);mod.__path__=[];sys.modules[name]=mod
spec=importlib.util.spec_from_file_location('vllm_ascend.spec_decode.pearl.tree',N/'tree.py')
tree=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tree;spec.loader.exec_module(tree)
predicate=next(n for n in ast.parse((N/'native_model.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_is_contiguous_linear_tree_plan')
module=types.ModuleType('vllm_ascend.spec_decode.pearl.native_model');module.__dict__['torch']=torch
exec(compile('from __future__ import annotations\n'+ast.unparse(predicate),str(N/'native_model.py')+'::HOSTpredicate','exec'),module.__dict__)
sys.modules[module.__name__]=module
sys.path[:0]=[str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-high-normal-cap4-native-families-v148')]
from padding249 import pad
from branch_tail249 import layout
node=copy.deepcopy(next(n for n in ast.parse((N/'native_graph.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_validate_tree_fia_mask'))
ns=dict(torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(N/'native_graph.py')+'::HOSTvalidator','exec'),ns)
@dataclasses.dataclass
class Meta:
    slot_mapping:object;context_lens:object;block_tables:object;request_block_tables:object
    tree_attention_mask:object;actual_seq_lengths_q:object;sequence_lens:object
    tree_attention:bool=True;use_fused_infer_attention:bool=True


def naive(inputs,positions,meta,real,zero,capacity):
    p=plan(meta.actual_seq_lengths_q,real);index=torch.tensor(p['physical_to_original'])
    tables=torch.full((26,32),zero,dtype=torch.int32);mask=torch.ones((26,1,capacity,4096),dtype=torch.bool)
    for group,source in enumerate(p['physical_group_sources']):
        if source is not None:tables[group]=torch.where(meta.request_block_tables[source]>=0,meta.request_block_tables[source],zero)
    for move in p['real_moves']:
        g,q=divmod(move['new_row'],5);mask[g,0,q]=meta.tree_attention_mask[move['old_group'],0,move['old_query']]
    for row in p['phantom_rows']:g,q=divmod(row,5);mask[g,0,q,0]=False
    return(inputs.index_select(0,index),positions.index_select(0,index),dataclasses.replace(meta,
        slot_mapping=meta.slot_mapping.index_select(0,index),context_lens=meta.context_lens.index_select(0,index),
        block_tables=tables,request_block_tables=tables,tree_attention_mask=mask,
        actual_seq_lengths_q=tuple(range(5,126,5))+(128,),sequence_lens=(2560,)*26),
        torch.tensor(p['original_to_physical']),p)


def main():
    patterns=[]
    for n in range(1,25):
        for label,budgets in [('all1',[1]*n),('all4',[4]*n),('mixed',[(i%5)+1 for i in range(n)]),('fork',[5]*n)]:
            if sum(budgets)>96 or n+budgets.count(5)>25:continue
            for anchor in(range(4)if 5 in budgets else(0,)):patterns.append((f'N{n}-{label}-a{anchor}',budgets,anchor,127))
    patterns.extend((f'boundary{prefix}-N{n}-a{anchor}',[5]*n,anchor,prefix)
                    for prefix in(0,128,1505,2555)for n in(1,8,12)for anchor in range(4))
    patterns.extend((f'boundary{prefix}-N24-all4',[4]*24,0,prefix)for prefix in(0,128,1505,2555))
    checked=0;copy_rows=0;identity_cases=0;root_only_cases=0
    for label,budgets,anchor,prefix in patterns:
        raw=tree.cached_cpu_tree_speculation_plan(2,4,prefix,4096,candidate_budget=8)
        plans=[];rows=[]
        for i,b in enumerate(budgets):
            chosen=list(range(min(b,4)))+([4+anchor]if b==5 else[])
            plans.append(tree.pack_selected_tree_plan(raw,chosen));rows.append([103+17*i+29*j for j in chosen])
        real=sum(b+1 for b in budgets);n=len(plans)
        engine=types.SimpleNamespace(config=types.SimpleNamespace(max_model_len=4096),_tree_graph_scratch_sequence_ids=tuple(range(463,527)))
        pp,_,_,ids,count=pad(engine,plans,list(range(n)),rows,list(range(n)),[],[])
        assert all(a is b for a,b in zip(pp[:n],plans,strict=True))and count==n
        root_only_cases+=any(p.candidate_budget==0 for p in pp[n:])
        tables=[[-1]*32 for _ in range(527)];page=0
        for owner,p in zip(ids,pp,strict=True):
            used=(max(p.cache_positions.tolist())+128)//128;tables[owner][:used]=range(page,page+used);page+=used
        for owner in range(463,527):
            if tables[owner][0]<0:tables[owner][0]=page;page+=1
        assert page<=544
        desc=layout(pp,ids,tables,engine._tree_graph_scratch_sequence_ids)
        if desc['enabled']:
            masks=desc['mask'];bt=desc['tables'];ends=desc['actual_seq_lengths_q'];lengths=desc['sequence_lens']
            copied=dict(zip(desc['dest'],desc['source'],strict=True));copy_rows+=1
            assert tables[526][0]not in{v//128 for v in desc['dest']}
        else:
            masks=torch.ones((len(pp),1,9,4096),dtype=torch.bool);bt=[tables[i]for i in ids];ends=[];lengths=[];offset=0;copied={}
            for g,p in enumerate(pp):
                q=p.candidate_budget+1;masks[g,0,:q]=p.attention_mask;offset+=q;ends.append(offset);lengths.append(max(p.cache_positions.tolist())+1)
        original=[];slots=[];positions=[];contexts=[]
        for owner,p in zip(ids,pp,strict=True):
            for mask,physical,logical in zip(p.attention_mask,p.cache_positions.tolist(),p.positions.tolist(),strict=True):
                visible=(~mask).nonzero().flatten().tolist()
                original.append([tables[owner][k//128]*128+k%128 for k in visible])
                slots.append(tables[owner][physical//128]*128+physical%128);positions.append(logical);contexts.append(physical+1)
        meta=Meta(torch.tensor(slots),torch.tensor(contexts,dtype=torch.int32),torch.tensor(bt,dtype=torch.int32),torch.tensor(bt,dtype=torch.int32),masks,tuple(ends),tuple(lengths))
        capacity=11 if desc['enabled']else 10;args=(torch.arange(128)+17,torch.tensor(positions),meta,real,tables[526][0])
        actual=transform(*args,mask_capacity=capacity);expected=naive(*args,capacity)
        for a,b in zip(actual[:2]+actual[3:4],expected[:2]+expected[3:4],strict=True):assert torch.equal(a,b)
        for field in dataclasses.fields(Meta):
            a,b=getattr(actual[2],field.name),getattr(expected[2],field.name)
            assert torch.equal(a,b)if isinstance(a,torch.Tensor)else a==b
        x,pos,out,inverse,projection=actual;assert torch.equal(x.index_select(0,inverse),args[0])
        identity_cases+=projection['physical_to_original']==list(range(128))
        for move in projection['real_moves']:
            g,q=divmod(move['new_row'],5);visible=(~out.tree_attention_mask[g,0,q]).nonzero().flatten().tolist()
            physical=[int(out.request_block_tables[g,k//128])*128+k%128 for k in visible]
            assert [copied.get(v,v)for v in physical]==original[move['old_row']],(label,move)
            assert int(out.slot_mapping[move['new_row']])==slots[move['old_row']]
        ns['_validate_tree_fia_mask'](128,out.tree_attention_mask,out.request_block_tables,out.actual_seq_lengths_q,out.sequence_lens,128)
        checked+=1
    assert root_only_cases>0 and identity_cases>0 and copy_rows>0 and 'torch_npu'not in sys.modules
    result=dict(status='HOST_actual_M128_tree_visibility_and_Native_ABI_passed',supported_histograms=accepted,
        explicitly_unsupported_histograms=rejected,actual_tree_cases=checked,branch_copy_cases=copy_rows,
        identity_cases=identity_cases,root_only_dummy_cases=root_only_cases,all128_rows_bijective=True,
        all_real_ancestral_visible_KV_and_write_slots_preserved=True,optimized_transform_equals_independent_naive=True,
        private_zero_page_never_overwritten_by_branch_copy=True,NPU_used=False,neural_qualified=False,production_installed=False)
    with(P/'HOST_PROJECTION249_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
