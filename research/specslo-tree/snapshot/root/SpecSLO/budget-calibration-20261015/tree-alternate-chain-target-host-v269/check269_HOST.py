"""Original Native CPU protocol double for alternate chains B1..8."""
import ast,copy,inspect,json,sys,textwrap,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent

def main():
 sys.path.insert(0,str(B/'tree-deferred-FULL-mask-host-projection-v259'))
 from fixture259 import build
 env=build();engine=env['engine'];torch=env['torch'];native=env['native']
 from plan269 import make,envelope
 from layout269 import layout,prepared
 import branch_tail255,padding255
 # Clone only the original bound prepare's globals and the private q5 wrapper
 # cell. Original module/class globals and their source stay untouched.
 sys.path.append(str(B/'tree-fused-branch-KV-row-copy-v262'))
 from contract262 import checked
 state=engine._branch_tail_state;outer=state.prepare
 cells=dict(zip(outer.__code__.co_freevars,outer.__closure__))
 bound=cells['prepare'].cell_contents
 original=bound.__func__
 inner=types.FunctionType(original.__code__,dict(original.__globals__,layout=layout),
                          original.__name__,original.__defaults__,original.__closure__)
 def cell(value):return (lambda:value).__closure__[0]
 closure=tuple(cell(types.MethodType(inner,state))if name=='prepare'else cells[name]for name in outer.__code__.co_freevars)
 state.prepare=types.FunctionType(outer.__code__,outer.__globals__,outer.__name__,outer.__defaults__,closure)
 tree=ast.parse(textwrap.dedent(inspect.getsource(padding255.pad)))
 imports=[n for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)and n.module=='mapper255']
 assert len(imports)==1;imports[0].module='plan269'
 families=[n for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)and n.module=='target_layout']
 assert len(families)==1;families[0].module='plan269'
 ns=dict(vars(padding255));exec(compile(ast.fix_missing_locations(tree),str(P/'check269_HOST.py')+'::original-padding-with-declared-B8-envelope','exec'),ns)
 pad=ns['pad'];native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
 prepared()  # Complete AST inverse must pass before the first fixture.
 results=[];seen=set()
 for n in range(1,25):
  for raw in ([1]*n,[4]*n,[5]*n,[6]*n,[7]*n,[8]*n,[(i%8)+1 for i in range(n)]):
   budgets=list(raw)
   while n+sum(b>4 for b in budgets)>24:
    j=next(i for i in reversed(range(n))if budgets[i]>4);budgets[j]=4
   while sum(budgets)>96:
    j=next(i for i in reversed(range(n))if budgets[i]>1);budgets[j]-=1
   side=max(max(budgets)-4,0)
   for anchor in range(5-side)if side else(0,):
    key=(tuple(budgets),anchor)
    if key in seen:continue
    seen.add(key)
    for prefix_mode in ('zero','boundary127','boundary128','mixed-long'):
     prefixes=([0]*n if prefix_mode=='zero'else[127]*n if prefix_mode=='boundary127'else[128]*n if prefix_mode=='boundary128'else([1505,2555]+[127]*n)[:n])
     plans=[make(b,p,anchor)for b,p in zip(budgets,prefixes,strict=True)]
     rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)];roots=list(range(41,41+n))
     pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
     out=native.NativePearlEngine.target_tree_forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
     q=n+sum(budgets);physical=state.static_current['physical'];meta=physical[2]
     assert out['query_count']==120 and out['logical_query_count']==q and out['num_draft_tokens']==budgets
     assert out['target_query_token_ids'].tolist()==[x for root,row in zip(roots,rows,strict=True)for x in [root]+row]
     assert tuple(meta.actual_seq_lengths_q)==tuple(range(5,121,5))and meta.sequence_lens==(2560,)*24
     assert physical[4]['supported_primary_side_groups']==n+sum(b>4 for b in budgets)<=24
     assert out['cache_slot_mapping'].numel()==q and out['cache_slot_mapping'].unique().numel()==q
     if side:
      desc=layout(pp,ss,engine.cache_allocation.block_tables,engine._tree_graph_scratch_sequence_ids)
      proof=checked(desc['source'],desc['dest'],rows=69632)
      # Symbolic original row IDs check snapshot-copy ancestry exactly.
      moved=dict(zip(desc['dest'],desc['source'],strict=True));begin=0;offsets=[]
      for plan in plans:offsets.append(begin);begin+=plan.candidate_budget+1
      for group in desc['groups']:
       owner=group['owner']
       if owner>=n:continue
       plan=plans[owner];source_table=engine.cache_allocation.block_tables[owner]
       complete=prefixes[owner]//128
       assert group['tables'][:complete]==source_table[:complete]
       for j,oldrow in enumerate(group['original_rows']):
        local=oldrow-offsets[owner]
        raw_visible=(~plan.attention_mask[local]).nonzero().flatten().tolist()
        new_visible=(~group['masks'][j]).nonzero().flatten().tolist()
        assert len(raw_visible)==len(new_visible)==int(plan.positions[local])+1
        raw_tail=[v for v in raw_visible if v>=complete*128]
        new_tail=[v for v in new_visible if v>=complete*128]
        expected=[source_table[v//128]*128+v%128 for v in raw_tail]
        slots=[group['tables'][v//128]*128+v%128 for v in new_tail]
        assert [moved.get(v,v)for v in slots]==expected
     results.append(dict(N=n,budgets=budgets,anchor=anchor,prefix_mode=prefix_mode,
         physical_M=120,groups=n+sum(b>4 for b in budgets),real_queries=q,branch_tail_copy_slots=state.last_record.get('real_copy_slots',0)))
 assert any(8 in r['budgets']for r in results)and 'torch_npu'not in sys.modules
 result=dict(status='HOST_actual_Native_B1to8_alternate_chain_protocol_and_copy_ancestry_passed',
   cases=len(results),rows=results,original_layout_AST_inverse=True,
   original_Native_forward_executed=True,all_real_token_order_and_KV_slots_preserved=True,
   same_M120_S24_q5_KV2560=True,copy1792_contract_preserved=True,
   model='CPU arithmetic protocol double; no neural or attention computation',
   NPU_used=False,neural_qualified=False,production_installed=False,Goodput=None)
 with(P/'HOST269_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
if __name__=='__main__':main()
