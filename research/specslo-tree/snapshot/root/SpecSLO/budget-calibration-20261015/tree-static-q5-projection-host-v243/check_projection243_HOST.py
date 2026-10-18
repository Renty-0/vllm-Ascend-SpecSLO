"""Exhaustive budget shapes and actual-tree KV visibility oracle, CPU only."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,copy,json,hashlib,types,sys,importlib.util,itertools
from pathlib import Path
from dataclasses import dataclass
P=Path(__file__).resolve().parent;B=P.parent
from projection243 import plan,transform

accepted=rejected=0
for n in range(1,17):
    for budgets in itertools.combinations_with_replacement(range(1,6),n):
        if sum(budgets)>64:continue
        sizes=[]
        for b in budgets:sizes.extend([5,1]if b==5 else[b+1])
        real=sum(sizes);remain=80-real
        while remain:sizes.append(min(5,remain));remain-=min(5,remain)
        ends=[];cursor=0
        for q in sizes:cursor+=q;ends.append(cursor)
        supported=n+budgets.count(5)<=16
        try:result=plan(ends,real)
        except ValueError:
            assert not supported;rejected+=1;continue
        assert supported and sorted(result['physical_to_original'])==list(range(80))
        assert [result['physical_to_original'][i]for i in result['original_to_physical']]==list(range(80))
        assert all(result['physical_to_original'][m['new_row']]==m['old_row']for m in result['real_moves'])
        accepted+=1

import torch
torch.set_num_threads(1)
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
for name in ('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
    module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
spec=importlib.util.spec_from_file_location('vllm_ascend.spec_decode.pearl.tree',N/'tree.py')
tree=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tree;spec.loader.exec_module(tree)
predicate=next(x for x in ast.parse((N/'native_model.py').read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='_is_contiguous_linear_tree_plan')
module=types.ModuleType('vllm_ascend.spec_decode.pearl.native_model');module.__dict__['torch']=torch
exec(compile('from __future__ import annotations\n'+ast.unparse(predicate),str(N/'native_model.py')+'::HOSTpredicate','exec'),module.__dict__)
sys.modules[module.__name__]=module
sys.path[:0]=[str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-high-normal-cap4-native-families-v148')]
from neural_oracle231 import cases,prepare
from padding231 import pad
from branch_tail231 import layout
node=copy.deepcopy(next(x for x in ast.parse((N/'native_graph.py').read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='_validate_tree_fia_mask'))
ns=dict(torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(N/'native_graph.py')+'::HOSTvalidator','exec'),ns)
@dataclass
class Meta:
    slot_mapping:object;context_lens:object;block_tables:object;request_block_tables:object
    tree_attention_mask:object;actual_seq_lengths_q:object;sequence_lens:object
    tree_attention:bool=True;use_fused_infer_attention:bool=True

fixtures=[]
for case in cases():fixtures.append((case['name'],prepare(case)[0]))
for prefix in (0,127,128,1505,2555):
    for anchor in range(4):
        raw=tree.cached_cpu_tree_speculation_plan(2,4,prefix,4096,candidate_budget=8)
        fixtures.append((f'long{prefix}-side{anchor}',[tree.pack_selected_tree_plan(raw,[0,1,2,3,4+anchor])]))
details=[];unsupported=0
for label,plans in fixtures:
    real=sum(p.candidate_budget+1 for p in plans);count=len(plans)
    engine=types.SimpleNamespace(config=types.SimpleNamespace(max_model_len=4096),_tree_graph_scratch_sequence_ids=tuple(range(463,527)))
    pp,_,_,ids,_=pad(engine,plans,list(range(count)),[[17+j for j in range(p.candidate_budget)]for p in plans],list(range(count)),[],[])
    tables=[[-1]*32 for _ in range(527)];page=0
    for owner,p in zip(ids,pp,strict=True):
        k=(max(p.cache_positions.tolist())+128)//128;tables[owner][:k]=list(range(page,page+k));page+=k
    for owner in engine._tree_graph_scratch_sequence_ids:
        if tables[owner][0]<0:tables[owner][0]=page;page+=1
    assert page<=544
    desc=layout(pp,ids,tables,engine._tree_graph_scratch_sequence_ids)
    if desc['enabled']:
        masks=desc['mask'];bt=desc['tables'];ends=desc['actual_seq_lengths_q'];lengths=desc['sequence_lens'];copied=dict(zip(desc['dest'],desc['source'],strict=True))
    else:
        masks=torch.ones((len(pp),1,9,4096),dtype=torch.bool);bt=[tables[i]for i in ids];ends=[];lengths=[];cursor=0
        for g,p in enumerate(pp):
            q=p.candidate_budget+1;masks[g,0,:q]=p.attention_mask;cursor+=q;ends.append(cursor);lengths.append(max(p.cache_positions.tolist())+1)
        copied={}
    original=[];slots=[];positions=[];contexts=[]
    for owner,p in zip(ids,pp,strict=True):
        for row,position,logical in zip(p.attention_mask,p.cache_positions.tolist(),p.positions.tolist(),strict=True):
            visible=(~row).nonzero().flatten().tolist()
            original.append([tables[owner][k//128]*128+k%128 for k in visible])
            slots.append(tables[owner][position//128]*128+position%128);positions.append(logical);contexts.append(position+1)
    meta=Meta(torch.tensor(slots),torch.tensor(contexts,dtype=torch.int32),torch.tensor(bt,dtype=torch.int32),torch.tensor(bt,dtype=torch.int32),masks,tuple(ends),tuple(lengths))
    expected=len(plans)+sum(any(p.parent_indices.tolist().count(v)>1 for v in set(p.parent_indices.tolist()))for p in plans)<=16
    try:x,pos,out,inverse,projection=transform(torch.arange(80)+17,torch.tensor(positions),meta,real,tables[526][0])
    except ValueError:
        assert not expected;unsupported+=1;continue
    assert expected and torch.equal(x.index_select(0,inverse),torch.arange(80)+17)
    assert torch.equal(out.slot_mapping.index_select(0,inverse),meta.slot_mapping)
    assert torch.equal(pos.index_select(0,inverse),torch.tensor(positions))
    for move in projection['real_moves']:
        old,new=move['old_row'],move['new_row'];g,q=divmod(new,5)
        visible=(~out.tree_attention_mask[g,0,q]).nonzero().flatten().tolist()
        physical=[int(out.request_block_tables[g,k//128])*128+k%128 for k in visible]
        assert [copied.get(v,v)for v in physical]==original[old],(label,old)
        assert int(out.slot_mapping[new])==slots[old]
    for new in projection['phantom_rows']:
        g,q=divmod(new,5);assert (~out.tree_attention_mask[g,0,q]).nonzero().flatten().tolist()==[0]
    ns['_validate_tree_fia_mask'](80,out.tree_attention_mask,out.request_block_tables,out.actual_seq_lengths_q,out.sequence_lens,128)
    details.append(dict(case=label,real_queries=real,primary_side_groups=projection['supported_primary_side_groups'],copy_enabled=desc['enabled']))
assert 'torch_npu'not in sys.modules
result=dict(status='HOST_actual_tree_visibility_and_Native_ABI_passed',budget_histograms_supported=accepted,budget_histograms_explicitly_unsupported=rejected,
            actual_tree_cases_passed=len(details),actual_tree_cases_explicitly_unsupported=unsupported,all80_input_position_and_write_slot_rows_bijective=True,
            all_real_ancestral_KV_orders_exact=True,all_phantoms_finite_one_key_mask=True,original_Native_mask_validator_passed=True,
            NPU_initialized=False,neural_qualified=False,performance_qualified=False,production_installed=False,
            source_sha256=hashlib.sha256((P/'projection243.py').read_bytes()).hexdigest(),details=details)
(P/'HOST_PROJECTION243_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='details'}))
