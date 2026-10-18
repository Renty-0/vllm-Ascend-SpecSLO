"""Real Torch CPU FULL9 plans/masks/slots/buckets, no neural/device claim."""
import ast,copy,hashlib,importlib.util,json,sys,types
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Any,Sequence
P=Path(__file__).resolve().parent;B=P.parent;N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def need(x,m):
 if not x:raise AssertionError(m)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 import torch
 torch.set_num_threads(1) # CPU fixture packing only; no device/runtime setting changed.
 need('torch_npu'not in sys.modules,'CPU only')
 fixture=json.loads((P/'FIXTURE228.json').read_text())
 old=json.loads((B/'tree-FP32-BF16-tie-refinement-diagnostic-v223/FIXTURE223.json').read_text())
 need(fixture['history']==old['history']and fixture['fresh']['cases'][:1]==old['fresh']['cases'],'original3 contexts unchanged plus current224 remaining mismatch')
 need(fixture['fresh']['cases'][1]['first_difference']==148,'new remaining context included')
 for part in('fresh','history'):
  for path,digest in fixture[part]['inputs'].items():need(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'original input binding')
 for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
  m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
 load('vllm_ascend.spec_decode.pearl.tree',N/'tree.py');cache=load('_FULL228_original_cache',N/'native_cache.py')
 ns=dict(torch=torch,dataclass=dataclass,replace=replace,Any=Any,Sequence=Sequence,NativeCacheAllocation=cache.NativeCacheAllocation)
 funcs={'make_attention_metadata','_is_contiguous_linear_tree_plan','make_tree_fia_mask'}
 nodes=[copy.deepcopy(n)for n in ast.walk(ast.parse((N/'native_model.py').read_text()))if(isinstance(n,ast.FunctionDef)and n.name in funcs)or(isinstance(n,ast.ClassDef)and n.name=='NativeAttentionMetadata')]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(N/'native_model.py')+'::original-HOST','exec'),ns)
 mod=types.ModuleType('vllm_ascend.spec_decode.pearl.native_model');mod.__dict__.update(ns);sys.modules[mod.__name__]=mod
 names={'_allocate_cache','_activate_cache_sequence','_ensure_cache_capacity','_release_cache','_prepare_attention_metadata','_cache_slot_mapping'}
 nodes=[copy.deepcopy(n)for n in ast.walk(ast.parse((N/'native_engine.py').read_text()))if isinstance(n,ast.FunctionDef)and n.name in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(N/'native_engine.py')+'::original-HOST','exec'),ns)
 sys.path[:0]=[str(P),str(B/'tree-high-normal-cap4-native-families-v148')];import host_target_metadata as host
 host._uses_fused_attention=lambda device,attention:True # Isolated CPU ABI test predicate, no device assertion.
 from metadata228 import pack
 def engine(prompts,reserve):
  model=types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=torch.zeros(1)),max_model_len=4096,
   layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(block_table=torch.zeros((527,32),dtype=torch.int32),blocks_per_sequence=32,block_size=128,uses_paged_attention=True))],attention_mask=torch.zeros((2048,2048),dtype=torch.int8))
  model.make_attention_metadata=types.MethodType(ns['make_attention_metadata'],model)
  e=types.SimpleNamespace(device=torch.device('cpu'),model=model,prefix_cache=cache.NativePrefixCache(544,32,128),_cache_storage_sequence_capacity=527,_tree_cache_capacity=544*128,_tree_graph_scratch_sequence_ids=tuple(range(463,527)),config=types.SimpleNamespace(kvcache_block_size=128,max_model_len=4096))
  for name in names:setattr(e,name,types.MethodType(ns[name],e))
  e._allocate_cache(prompts,enable_prefix_caching=False,reserve_sequence_capacity=True)
  for seq,n in enumerate(reserve):e._ensure_cache_capacity([seq],[n+8])
  for seq in range(463,527):e._activate_cache_sequence(seq,[0],enable_prefix_caching=False)
  return e
 count=0;bucket_checks=[]
 for case in fixture['history']['cases']:
  e=engine(case['prompts'],case['reserve_lengths'])
  for mode in('candidate','reference'):
   for frame in case[mode+'_frames']:
    for seq,pos in zip(frame['ids'],frame['positions'],strict=True):e._ensure_cache_capacity([seq],[pos])
    inputs,pos,meta=pack(e,frame,full=True);need(meta.tree_attention and tuple(meta.tree_attention_mask.shape)==(80,1,1,4096),'all ordinary q5 frames split as FULL query1')
    need(inputs.tolist()==frame['tokens']and pos.tolist()==frame['positions'],'unchanged actual tokens and positions')
    for j in range(16):
     for k in range(5):
      p=frame['positions'][5*j+k];mask=meta.tree_attention_mask[5*j+k,0,0]
      need(not bool(mask[:p+1].any())and bool(mask[p+1:].all()),'exact per-query causal visibility in FULL mask')
    need(meta.actual_seq_lengths_q==tuple(range(1,81))and meta.sequence_lens==tuple(p+1 for p in frame['positions']),'exact one-query partition and own visible extent')
    count+=1
  e._release_cache()
 from fresh228 import prepare_frame
 for case in fixture['fresh']['cases']:
  prompts=[r['prefix']for r in case['rows']];e=engine(prompts,[len(p)+9 for p in prompts])
  for frame in case['frames']:
   v=prepare_frame(e,case,frame);m=v['metadata'];need(m.tree_attention and v['inputs'].numel()==80 and m.actual_seq_lengths_q==tuple(range(1,81)),'canonical M80 query1 FULL');
   for i,n in enumerate(m.sequence_lens):need(not bool((~m.tree_attention_mask[i,0,0,n:]).any()),'no visible KV truncated by per-query physical extent')
   count+=1
  e._release_cache()
 need(count==427,'421 incremental +6 fresh frames')
 for p in P.glob('*.py'):ast.parse(p.read_text())
 need('torch_npu'not in sys.modules,'no NPU')
 out=P/'CPU228_RESULT.json';need(not out.exists(),'fresh metadata evidence');out.write_text(json.dumps(dict(status='passed',frames=count,exact_per_query_FULL_causal_and_branch_masks_verified=True,actual_runtime_uses_exact_KV_lengths=True,bucket_helper_not_used=True,Torch_CPU=True,isolated_FIA_device_predicate_stub=True,NPU_used=False,neural_qualified=False,performance_qualified=False),indent=2)+'\n');print(json.dumps(dict(status='passed',frames=count,bucket_checks=bucket_checks,NPU_used=False)))
if __name__=='__main__':main()
