"""Actual original metadata builder and FIA ABI on CPU external doubles."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ast,copy,dataclasses,hashlib,importlib.util,json,sys,types
from pathlib import Path
import torch
torch.set_num_threads(1)
P=Path(__file__).resolve().parent;B=P.parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
    m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
spec=importlib.util.spec_from_file_location('vllm_ascend.spec_decode.pearl.tree',N/'tree.py')
tree=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tree;spec.loader.exec_module(tree)
native=types.ModuleType('vllm_ascend.spec_decode.pearl.native_model');sys.modules[native.__name__]=native
native.__dict__.update(torch=torch,dataclass=dataclasses.dataclass)
nodes=[copy.deepcopy(n)for n in ast.parse((N/'native_model.py').read_text()).body
       if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in('NativeAttentionMetadata','_is_contiguous_linear_tree_plan','make_tree_fia_mask')]
exec(compile('from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])),str(N/'native_model.py')+'::HOST_metadata_only', 'exec'),native.__dict__)
sys.path[:0]=[str(B/'tree-high-normal-cap4-native-families-v148')]
import host_target_metadata as host
host._uses_fused_attention=lambda device,attention:True # Explicit CPU ABI seam, no actual device claim.
node=copy.deepcopy(next(n for n in ast.parse((N/'native_graph.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_validate_tree_fia_mask'))
ns=dict(torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(N/'native_graph.py')+'::HOST_validator','exec'),ns)
from frames248 import pack


def engine_for(count,position):
    tables=[[-1]*32 for _ in range(527)];page=0;used=(position+5+127)//128
    for owner in range(count):tables[owner][:used]=range(page,page+used);page+=used
    for owner in range(463,527):tables[owner][0]=page;page+=1
    assert page<=544
    return types.SimpleNamespace(cache_allocation=types.SimpleNamespace(block_tables=tables),
        cache_block_tables=torch.tensor(tables,dtype=torch.int32),_tree_cache_capacity=544*128,
        model=types.SimpleNamespace(max_model_len=4096,embed_tokens=types.SimpleNamespace(weight=torch.empty(1)),
            layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(block_size=128,uses_paged_attention=True))]))


def main():
    checked=[]
    for rows,counts in[(80,(1,11,16)),(128,(1,16,22,24))]:
        for n in counts:
            for position in(0,127,128,1505,2555):
                for reference in(False,True):
                    engine=engine_for(n,position);tokens=[[17+i,21+i,31+i,41+i,51+i]for i in range(n)]
                    x,pos,meta=pack(engine,list(range(n)),[position]*n,tokens,rows=rows,reference=reference)
                    groups=16 if rows==80 else 26
                    assert len(x)==rows and tuple(meta.tree_attention_mask.shape)==(groups,1,10,4096)
                    assert list(meta.actual_seq_lengths_q)==(list(range(5,81,5))if rows==80 else list(range(5,126,5))+[128])
                    assert meta.sequence_lens==(2560,)*groups
                    assert x[:5*n].tolist()==[v for row in tokens for v in row]
                    assert pos[:5*n].tolist()==list(range(position,position+5))*n
                    assert meta.slot_mapping[:5*n].tolist()==[engine.cache_allocation.block_tables[i][p//128]*128+p%128 for i in range(n)for p in range(position,position+5)]
                    for group in range(n):
                        for query in range(5):
                            expected=([0]if reference and query else list(range(position+query+1)))
                            assert (~meta.tree_attention_mask[group,0,query]).nonzero().flatten().tolist()==expected
                    for group in range(n,groups):
                        qcount=3 if rows==128 and group==25 else 5
                        for q in range(qcount):assert (~meta.tree_attention_mask[group,0,q]).nonzero().flatten().tolist()==[0]
                        assert meta.tree_attention_mask[group,0,qcount:].all()
                    ns['_validate_tree_fia_mask'](rows,meta.tree_attention_mask,meta.request_block_tables,meta.actual_seq_lengths_q,meta.sequence_lens,128)
                    checked.append(dict(M=rows,N=n,position=position,reference=reference))
    for bad in(2556,2560):
        try:pack(engine_for(1,2555),[0],[bad],[[1,2,3,4,5]],rows=128)
        except ValueError:pass
        else:raise AssertionError('real five-query extent must reject unsupported keys')
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_original_builder_and_Native_FIA_ABI_passed',cases=len(checked),
        all_real_tokens_positions_write_slots_and_visible_keys_checked=True,
        uniform25_q5_plus_nonreal3_tail=True,unsupported_real_extent_rejected=True,
        no_source_Native_validator_changed=True,CPU_external_double_only=True,NPU_used=False,
        source_sha256=hashlib.sha256((P/'frames248.py').read_bytes()).hexdigest(),details=checked)
    with(P/'HOST_FRAMES248_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='details'}))


if __name__=='__main__':main()
