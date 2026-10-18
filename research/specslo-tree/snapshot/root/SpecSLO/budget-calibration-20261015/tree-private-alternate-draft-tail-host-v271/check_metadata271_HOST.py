"""Actual original Native PA padding plus private-tail metadata checks."""
import ast,copy,dataclasses,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
SOURCE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')
def main():
 sys.path[:0]=[str(B/'tree-deferred-FULL-mask-host-projection-v259'),str(B/'tree-fused-branch-KV-row-copy-v262')]
 from fixture259 import build
 env=build();engine=env['engine'];torch=env['torch']
 node=copy.deepcopy(next(n for n in ast.walk(ast.parse(SOURCE.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='_pad_inputs'))
 node.decorator_list=[];ns=dict(torch=torch,replace=dataclasses.replace)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE)+'::unchanged-CPU-padding','exec'),ns)
 pad=ns['_pad_inputs']
 for owner in range(500,524):
  if engine.cache_allocation.block_tables[owner][0]<0:engine._activate_cache_sequence(owner,[0],enable_prefix_caching=False)
 from layout271 import prepare
 from metadata271 import build as metadata
 results=[]
 for n in (1,2,4,8,12):
  owners=list(range(n));bucket=next(b for b in (1,2,4,8,16,32)if b>=n)
  for prefix in (0,127,128,1505,2555):
   reserve=[(owner,pos)for owner in owners for pos in list(range(0,prefix+4,128))+[prefix+3]]
   engine._ensure_cache_capacity([x[0]for x in reserve],[x[1]for x in reserve])
   desc=prepare(engine.cache_allocation.block_tables,owners,[prefix]*n)
   primary=[41+i for i in owners];alternate=[500+i for i in owners]
   engine._dynamic_tree_known_KV={i:dict(safe_before=prefix,tokens={prefix:primary[i]})for i in owners}
   before=copy.deepcopy(engine.cache_allocation.block_tables);ledger=copy.deepcopy(engine._dynamic_tree_known_KV)
   for steps in (1,2,3):
    inp,positions,metas=metadata(engine,desc,alternate,primary,steps=steps,bucket=bucket,pad=pad)
    assert inp.tolist()==alternate+[0]*(bucket-n)and len(positions)==len(metas)==steps
    for step,(pos,meta)in enumerate(zip(positions,metas,strict=True)):
     assert pos.tolist()==[prefix+1+step]*n+[0]*(bucket-n)
     assert meta.slot_mapping.tolist()==[r['slots'][step]for r in desc['rows']]+[-1]*(bucket-n)
     assert meta.context_lens.tolist()==[prefix+2+step]*n+[1]*(bucket-n)
     assert meta.sequence_lens==tuple(meta.context_lens.tolist())
     assert meta.block_tables[:n].tolist()==desc['virtual_tables']
     assert meta.block_tables is metas[0].block_tables and meta.request_block_tables is None
     assert meta.use_fused_infer_attention is False
    assert engine.cache_allocation.block_tables==before and engine._dynamic_tree_known_KV==ledger
    results.append(dict(N=n,prefix=prefix,steps=steps,bucket=bucket))
   bad=list(primary);bad[0]+=1
   try:metadata(engine,desc,alternate,bad,steps=3,bucket=bucket,pad=pad)
   except RuntimeError:pass
   else:raise AssertionError('stale/unmaterialized primary root accepted')
 assert 'torch_npu'not in sys.modules
 result=dict(status='HOST_original_PA_padding_and_private_chain_metadata_passed',cases=len(results),rows=results,
   actual_original_pad_body_executed=True,original_owner_indices_preserved=True,
   ordinary_PA_mode_and_shared_block_table_preserved=True,all_alternate_slots_private=True,
   padded_slots_read_only_negative1=True,original_main_tables_and_ledger_unchanged=True,
   stale_primary_negative_control_rejected=True,NPU_used=False,model_executed=False,
   neural_qualified=False,production_installed=False,Goodput=None)
 with(P/'HOST_METADATA271_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
if __name__=='__main__':main()
