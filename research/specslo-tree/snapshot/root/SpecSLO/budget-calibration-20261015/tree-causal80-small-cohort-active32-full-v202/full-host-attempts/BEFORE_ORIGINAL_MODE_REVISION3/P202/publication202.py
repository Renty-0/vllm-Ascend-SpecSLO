"""Execute original root append/publication statements, not a new protocol."""
import ast,copy
from base202 import BASE,checked,require

def statements():
 tree=ast.parse(checked(BASE/'runtime_tree_decode.py').read_text())
 fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='admit_available')
 append=[n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.target)=='(index, token)' and ast.unparse(n.iter)=='zip(needs_prefill, tokens)']
 publish=[n for n in fn.body if isinstance(n,ast.For) and ast.unparse(n.target)=='index' and ast.unparse(n.iter)=='admitted' and any(isinstance(v,ast.Call) and ast.unparse(v.func)=='self._deliver_committed_tokens' for v in ast.walk(n))]
 require(len(append)==len(publish)==1,'exact original root append and publish loops')
 return copy.deepcopy([append[0],publish[0]])

def publish(engine,native,bank,tokens):
 import time
 from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmRuntimeState
 n=bank.n;params=[native.NativeSamplingParams(request_id='V202-'+str(i),max_tokens=256,ignore_eos=True,temperature=0.) for i in range(n)]
 draft=[native.PearlPipelineState(list(p),len(p),max_tokens=256,ignore_eos=True) for p in bank.prompts]
 target=[v.clone() for v in draft];local=draft if engine.is_draft else target
 states={i:SpecRhythmRuntimeState(request_index=i,home_batch_id=i%2,max_gamma=5,output_token_budget=256) for i in range(n)}
 callbacks=[];keys=('_token_commit_callback','_stream_delivered_counts','_stream_started','_token_commit_error')
 old={k:(k in vars(engine),vars(engine).get(k)) for k in keys}
 engine._token_commit_callback=lambda event:callbacks.append(dict(event));engine._stream_delivered_counts={};engine._stream_started=time.perf_counter();engine._token_commit_error=None
 pending=list(range(n));active=[]
 ns=dict(vars(native),self=engine,tokens=tokens,needs_prefill=list(range(n)),admitted=list(range(n)),
  draft_states=draft,target_states=target,local_states=local,request_params=params,states=states,pending_admission=pending,
  active=active,completed={},finished_decode_elapsed={},release_online_cache=lambda ids:None)
 try:
  exec(compile(ast.fix_missing_locations(ast.Module(body=statements(),type_ignores=[])),'V202::original-root-publication','exec'),ns)
  require(not pending and active==list(range(n)) and all(v.token_ids==p+[t] and v.committed_length==len(p)+1 for group in (draft,target) for v,p,t in zip(group,bank.prompts,tokens,strict=True)),'original append publishes each exact root once')
  wanted=n if engine.rank==engine.topology.target_leader_rank else 0
  require(len(callbacks)==wanted and all(row['token_ids']==[tokens[row['request_index']]] for row in callbacks) and engine._token_commit_error is None,'original first-token callback only on target leader')
  bank.contexts=[list(v.token_ids) for v in local];bank.reference_contexts=[p+[t] for p,t in zip(bank.prompts,tokens,strict=True)]
  bank.runtime_states=states;bank.publication_active=tuple(active)
  return dict(original_append_and_publish_AST=True,root_tokens=list(tokens),logical_tokens=bank.contexts,active=active,callbacks=callbacks,
   prompt_KV_not_copied_to_reference=True,original_callback=True,numerical_component_outside_formal_clocks=True)
 finally:
  for k,(had,value) in old.items():
   if had:setattr(engine,k,value)
   else:vars(engine).pop(k,None)
