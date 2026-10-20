"""Private P3/P4 source factory; P4 needs a new current-engine component issuer."""
import sys,types
from base179 import P,V159,functions,require
from scope159 import Capability,bucket,native_ready
from lease159 import guarded
_capability=Capability(frozenset({(1,1),(1,3),(32,32),(16,16)}),
 frozenset({(1,1),(3,1),(32,1),(32,32),(16,1),(16,16)}),mode='qualifying')
_factory=None
functions(V159/'live159.py',globals(),names=('fused_prelude',),changes=((
 'if any(int(p.candidate_budget)!=3 for p,row in sources.values()):return None,None',
 'if any(int(p.candidate_budget) not in (3,4) for p,row in sources.values()):return None,None',1),))

def prepare_helpers():
 global _factory
 from source159 import module,FUSION,controls_namespace
 require('fast_tree_draft_v5' not in sys.modules,'install current P4 factory before original root_seed')
 fusion=module('repair_frontier_fusion',FUSION)
 original=(V159/'fast159.generated.py').read_text();source=(P/'fast179.generated.py').read_text()
 require(source==fast_source(),'exact P4 delegate/lease/finite source with original math inverse')
 path=P/'fast179.generated.py';fast=types.ModuleType('fast_tree_draft_v5');fast.__file__=str(path);fast.__package__=''
 exec(compile(source,str(path),'exec'),fast.__dict__);sys.modules['fast_tree_draft_v5']=fast
 shape=dict(__name__='specslo_v179_shape',__file__=str(P/'shape179.generated.py'),_v159_capability=_capability,
  _v179_parent4_enabled=False,_v179_parent_budgets={})
 exec(compile((P/'shape179.generated.py').read_text(),shape['__file__'],'exec'),shape)
 _factory=dict(fast=fast,fusion=fusion,shape=shape,controls=controls_namespace(shape),P4_qualifying_only=True)
 return _factory

def activate(engine,proof,verify_actual):
 global _capability
 require(_factory is not None and callable(verify_actual) and verify_actual(engine,proof) is True,
  'same current actual P3 reps plus genuine P4 math/protocol/next evidence')
 require(engine.cache_allocation is None and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,
  'fixture cache released before production cap/warm/seal')
 _capability=Capability(frozenset((n,e) for n in range(1,33) for e in range(1,33)),
  frozenset((e,r) for e in range(1,33) for r in range(1,33)),mode='production')
 _factory['shape']['_v159_capability']=_capability;_factory['shape']['_v179_parent4_enabled']=True
 return dict(mode='production',P4_normal_steps=4,P4_normal_max16=True,P4_E_max16=True,P4_total_rows_max32=True,
  parent3_shared3_original_path_retained=True,normalB5_still_defers_all_eager=True,finite_representatives_not_all_tokens=True)

def validate_parent4(engine,native,plans,ids,sources,repairs,frontiers,steps):
 if not any(int(p.candidate_budget)==4 for p,row in sources.values()):return False
 normal=[p for p,i in zip(plans,ids) if i not in sources];N,E=len(normal),len(frontiers)
 require(steps==4 and normal and max(p.candidate_budget for p in normal)==4 and all(1<=p.candidate_budget<=4 for p in normal)
  and 1<=N<=16 and 1<=E<=16 and len(ids)==N+E<=32,'new P4 normal4/N16/E16/total32 structural domain')
 if _capability.mode=='qualifying':
  require(N==E==16 and len(repairs) in (0,1,16),'only actual finite P4 component cases before activation')
 else:
  require(_capability.mode=='production' and _factory['shape']['_v179_parent4_enabled'] is True,'own current-engine P4 issuer before live use')
  M=native._next_linear_draft_graph_bucket(len(ids),64);F=native._next_linear_draft_graph_bucket(E,64)
  U=native._next_linear_draft_graph_bucket(F+len(repairs),64) if repairs and all(len(v[2])==1 for v in repairs) else None
  require(native_ready(engine,_capability,chain_M=M,real_rows=len(ids),union_M=U),'actual current P4 Native chain/union closed entries')
  keys=[(f'draft-greedy:{engine.draft_vocab_size}|steps:4|paged|confidence',M)]
  if U is not None:keys.append(('target-hidden:0|steps:1|paged',(U,)))
  require(not any(k in engine.graph_runner.disabled_entry_keys for k in keys),'current P4 Native keys enabled')
 return True

def run_frontier(engine,native,runner_cls,frontiers,plans,ids,sources,steps,repairs):
 """Original frontier primitive, with cheap P4 source lease outside its commit."""
 import torch
 fids=[seq for seq,_,_ in frontiers];positions=[pos for _,pos,_ in frontiers];tokens=[t for _,_,t in frontiers]
 engine._ensure_cache_capacity(fids,positions)
 F=native._next_linear_draft_graph_bucket(len(frontiers),engine.config.max_num_seqs)
 def call():
  x=torch.tensor(tokens,dtype=torch.long,device=engine.device);p,m=engine._prepare_attention_metadata(fids,positions,False)
  x,p,m=runner_cls._pad_inputs(x,p,m,F)
  return engine.graph_runner.run_target_greedy([x],[p],[m],engine.draft_vocab_size)[0][:len(frontiers)].cpu().tolist()
 if not any(int(p.candidate_budget)==4 for p,row in sources.values()):return F,call()
 validate_parent4(engine,native,plans,ids,sources,repairs,frontiers,steps)
 if _capability.mode=='production':
  key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|paged',(F,));runner=engine.graph_runner;entry=runner.target_entries.get(key)
  require(entry is not None and entry.runtime_validated is True and key not in runner.disabled_entry_keys,'actual current P4 unfused frontier Native gate')
 from repair_frontier_fusion import AllocationWitness
 with guarded(engine,fids,AllocationWitness) as lease:
  lease.check();predicted=call();lease.check()
 require(all(type(v)is int and 0<=v<engine.draft_vocab_size for v in predicted),'actual finite P4 frontier token output')
 return F,predicted

def validate_parent4_output(engine,sources,rows,scores):
 if not any(int(p.candidate_budget)==4 for p,row in sources.values()):return
 import math
 require(all(len(r)==len(s)==8 and all(type(v)is int and 0<=v<engine.draft_vocab_size for v in r)
  and all(math.isfinite(float(v)) and 0<=v<=1 for v in s) for r,s in zip(rows,scores,strict=True)),
  'all already-retained actual P4 canonical8 tokens/scores finite before publication')

def fast_source():
 original=(V159/'fast159.generated.py').read_text();source=original
 old='        from live159 import fused_prelude\n'
 new='        from live179 import fused_prelude,validate_parent4,run_frontier,validate_parent4_output\n        validate_parent4(engine,native,plans,ids,sources,repairs,frontiers,steps)\n'
 require(source.count(old)==1,'one actual new P4 prelude validation anchor');source=source.replace(old,new)
 start=source.index('            if fused_plan is None:\n',source.index('        if frontiers:\n'))
 end=source.index('            for (seq,position,token),frontier',start)
 old_frontier=source[start:end]
 new_frontier='''            if fused_plan is None:
                fbucket,predicted=run_frontier(engine,native,runner_cls,frontiers,plans,ids,sources,steps,repairs)
'''
 source=source[:start]+new_frontier+source[end:]
 old_return="        return {'draft_token_ids':rows"
 new_return="        validate_parent4_output(engine,sources,rows,scores)\n"+old_return
 require(source.count(old_return)==1,'one P4 finite output anchor');source=source.replace(old_return,new_return)
 reverse=source.replace(new_return,old_return).replace(new_frontier,old_frontier).replace(new,old)
 require(reverse==original,'exact FastV5 inverse retains all original matrix/ledger/canonical operations')
 return source
