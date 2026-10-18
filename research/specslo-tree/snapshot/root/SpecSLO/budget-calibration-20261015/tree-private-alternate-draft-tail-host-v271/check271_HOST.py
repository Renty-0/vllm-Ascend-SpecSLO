"""Symbolic token-row proof of private alternate PA prefix/write isolation."""
import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
 sys.path[:0]=[str(B/'tree-deferred-FULL-mask-host-projection-v259'),str(B/'tree-fused-branch-KV-row-copy-v262')]
 from fixture259 import build
 env=build();engine=env['engine']
 for owner in range(500,524):
  if engine.cache_allocation.block_tables[owner][0]<0:
   engine._activate_cache_sequence(owner,[0],enable_prefix_caching=False)
 from layout271 import prepare
 cases=[]
 for n in (1,2,4,8,12):
  choices=[[p]*n for p in(0,1,126,127,128,129,1505,2555)]
  choices += [[1024+(17*i+r)%128 for i in range(n)]for r in range(128)]
  for prefixes in choices:
   owners=list(range(n))
   # The real service already prefills the complete prefix. This symbolic
   # fixture jumps between contexts, so allocate every intervening page too.
   reserve_ids=[];reserve_positions=[]
   for owner,prefix in zip(owners,prefixes,strict=True):
    for position in list(range(0,prefix+4,128))+[prefix+3]:
     reserve_ids.append(owner);reserve_positions.append(position)
   engine._ensure_cache_capacity(reserve_ids,reserve_positions)
   tables=engine.cache_allocation.block_tables;before=copy.deepcopy(tables)
   desc=prepare(tables,owners,prefixes)
   assert tables==before and desc['HOST_copy_contract']['in_place_snapshot_copy_equivalence']
   moved=dict(zip(desc['destination'],desc['source'],strict=True))
   real_pages={p for row in tables[:463]for p in row if p>=0}
   for ordinal,row in enumerate(desc['rows']):
    owner=row['owner'];p=row['prefix'];original=tables[owner];virtual=desc['virtual_tables'][ordinal]
    complete=p//128;assert virtual[:complete]==original[:complete]
    expected=[original[pos//128]*128+pos%128 for pos in range(complete*128,p+1)]
    actual_slots=[virtual[pos//128]*128+pos%128 for pos in range(complete*128,p+1)]
    assert [moved.get(slot,slot)for slot in actual_slots]==expected
    writes={}
    for step,(position,slot)in enumerate(zip(row['positions'],row['slots'],strict=True)):
     assert position==p+1+step and row['contexts'][step]==position+1
     assert slot//128 not in real_pages
     symbol=-(1+owner*3+step);writes[slot]=symbol
     expected_now=expected+[-(1+owner*3+s)for s in range(step+1)]
     slots=[virtual[pos//128]*128+pos%128 for pos in range(complete*128,position+1)]
     observed=[writes.get(s,moved.get(s,s))for s in slots]
     assert observed==expected_now
   assert desc['real_copy_rows']<=1536 and len(desc['source'])==1792
   cases.append(dict(N=n,prefixes=prefixes,real_copy_rows=desc['real_copy_rows']))
 tables=copy.deepcopy(engine.cache_allocation.block_tables)
 tables[500][0]=tables[0][0]
 try:prepare(tables,[0],[127])
 except ValueError:pass
 else:raise AssertionError('scratch-real alias accepted')
 assert 'torch_npu'not in sys.modules
 result=dict(status='HOST_private_alternate_PA_prefix_and_write_isolation_passed',
   cases=len(cases),rows=cases,all128_tail_residues=True,max_alt_rows12=True,
   same_root_plus_prefix_values_preserved=True,all_alternate_writes_private=True,
   original_tables_unchanged=True,real_alias_negative_control_rejected=True,
   NPU_used=False,model_executed=False,ledger_or_model_state_changed=False,
   neural_qualified=False,Goodput=None)
 with(P/'HOST271_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
if __name__=='__main__':main()
