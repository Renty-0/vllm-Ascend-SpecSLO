"""Genuine post-verdict next-normal B3 after actual promotion or invalidation.

The OLD history is freshly rebuilt with parent3/frontier/shared4, then only an
actually missing committed suffix is teacher-forced. This next wave is B3;
its ABI padding is never presented as the current mixed shared4 evidence.
"""
import copy
from scope151 import require,bucket
import draft151 as draft
import source151 as epoch
from finite151 import require_finite

def parameters(logical):
 ids=sorted(logical);require(ids==list(range(64)),'all64 original logical owners')
 return ids,[len(logical[i].token_ids)-1 for i in ids],[int(logical[i].token_ids[-1]) for i in ids],[list(map(int,logical[i].token_ids[-9:])) for i in ids]
def waves(e,ids,prefixes,roots,catchups,observe):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan as plan
 import observer151
 records=[];outputs={}
 for start in (0,32):
  group=ids[start:start+32];plans=[plan(2,4,prefixes[i],4096,candidate_budget=3) for i in group]
  fn=lambda:e.draft_tree_forward(plans,[roots[i] for i in group],group,committed_catchup_token_ids=[catchups[i] for i in group])
  output,seen=observer151.run(e,fn) if observe else (fn(),None);records.append(dict(ids=group,output=output,observed=seen))
  for row,i in enumerate(group):outputs[i]={k:draft.cpu_output(output)[k][row] for k in ('draft_token_ids','root_token_ids','node_confidences','draft_confidence')}
 return records,outputs

def restore_source(e,case,trace,stage):
 epoch.rebuild(e,trace['fixture']['contexts'],[case.repair]*32+[0]*32,stage,role='draft')
 def restore():
  if not e.is_draft:return None
  f=draft.fixture(e,case,trace.get('changed',False));out=draft.invoke(e,f)
  require(draft.cpu_output(out)==trace['mixed'],'fresh actual parent3/frontier/shared4 source output restored')
  return epoch.ledger(e)
 return stage('actual-whole-parent3-shared4-source-restore',restore)

def proof(e,case,trace,logical,stage):
 import torch
 import fast_tree_draft_v5 as fast
 ids,prefixes,roots,catchups=stage('logical-prefix-root-catchup',lambda:parameters(logical))
 def before():
  if not e.is_draft:return None
  saved=epoch.ledger(e);starts=[];suffixes=[]
  for i in ids:
   start,suffix=fast.missing_committed_suffix(saved[i],prefixes[i],catchups[i]);starts.append(start);suffixes.append(suffix)
  require(all(len(v)<=1 for v in suffixes),'genuine missing suffix0/1 after normal1..4/parent3/child3; no truncation')
  return saved,starts,suffixes,epoch.protect(e,[128]*64)
 before_state=stage('actual-invalidation-or-promotion-repair-inputs',before);warm=[]
 for k,change in enumerate((0,1,0)):
  restored=restore_source(e,case,trace,lambda n,f:stage('warm'+str(k)+'/'+n,f))
  def call():
   if not e.is_draft:return None
   require(restored==before_state[0],'fresh source ledger restored before changed nextroot')
   r=[(v+change)%e.draft_vocab_size for v in roots];c=[[*v[:-1],r[i]] for i,v in enumerate(catchups)]
   return waves(e,ids,prefixes,r,c,False)[1]
  warm.append(stage('warm'+str(k)+'/true-next3-draft',call))
 stage('nextroot-FTF-output-gate',lambda:require(warm[0]==warm[2],'actual nextroot F/T/F output restoration') if e.is_draft else None)
 restore_source(e,case,trace,lambda n,f:stage('actual/'+n,f))
 def actual():
  if not e.is_draft:return None
  saved,starts,suffixes,baseline=before_state;known=epoch.protect(e,starts);seen,out=waves(e,ids,prefixes,roots,catchups,True)
  require(epoch.protect(e,starts)==known and epoch.protect(e,[128]*64)==baseline,'known valid KV and originalsource protected in genuine nextdraft')
  return seen,out,epoch.logical_KV(e,[128]*64,[p-128+3 for p in prefixes]),epoch.ledger(e)
 actual_state=stage('actual-next3-waves-and-immediate-queryKV',actual)
 # Fresh prefix/model bank, no candidate tensors initialize this reference.
 epoch.rebuild(e,trace['fixture']['contexts'],[case.repair]*32+[0]*32,stage,role='draft')
 def independent():
  if not e.is_draft:return None
  saved,starts,suffixes,_=before_state;observed,actual_out,candidate_KV,candidate_ledger=actual_state
  reference=draft.independent(e,case,trace['fixture']);require(reference['ledger']==saved,'own fresh OLD parent3/frontier/shared4 ledger')
  ledger=copy.deepcopy(reference['ledger']);reference_out={};matcher=e.graph_runner._target_outputs_match;waveproof=[]
  for actual_wave in observed:
   group=actual_wave['ids'];repair=[i for i in group if suffixes[i]];repair_h=None
   for i in group:
    got,suffix=fast.missing_committed_suffix(ledger[i],prefixes[i],catchups[i]);require(got==starts[i] and suffix==suffixes[i],'independent true committed suffix start/tokens')
   if repair:
    call=draft.PA(e,repair,[starts[i] for i in repair],[suffixes[i][0] for i in repair],bucket(len(repair)));repair_h=e.model(*call).clone()
   encoded,hidden=draft.chain(e,group,[prefixes[i] for i in group],[roots[i] for i in group],3,bucket(len(group)))
   rows,scores=draft.decode(encoded,len(group),3);seen=actual_wave['observed'];out=actual_wave['output']
   require(len(seen['draft'])==1 and len(seen['hidden'])==int(bool(repair)) and not seen['greedy'] and not seen['root_head'],'actual generic nextnormal B3; not current shared4 scope')
   finite_actual=require_finite('nextnormal actual allphysical hidden',seen['draft'][0]['hidden'],[v['hidden'] for v in seen['hidden']])
   finite_reference=require_finite('nextnormal own allphysical hidden',hidden,[repair_h] if repair else [])
   if repair:require(matcher((seen['hidden'][0]['hidden'],),(repair_h,)),'own allphysical nextrepair hidden')
   require(matcher(seen['draft'][0]['hidden'],hidden) and torch.equal(seen['draft'][0]['encoded'],encoded),'own allphysical next3 hidden/encoded')
   for j,i in enumerate(group):
    reference_out[i]=dict(draft_token_ids=rows[j],root_token_ids=roots[i],node_confidences=torch.tensor(scores[j]).tolist(),draft_confidence=float(torch.tensor(sum(scores[j])/6)))
    ledger[i]=dict(safe_before=prefixes[i],tokens={prefixes[i]+k:v for k,v in enumerate([roots[i],*rows[j][:2]])})
   require(draft.cpu_output(out)['draft_token_ids']==rows and draft.cpu_output(out)['root_token_ids']==[roots[i] for i in group],'exact actual nextnormal token/root output')
   entry=e.graph_runner.draft_entries[seen['draft'][0]['key']]
   require(bool(entry.runtime_validated) and entry.validated_real_row_count>=len(group),'natural next3 Native changed/realrow qualification')
   waveproof.append(dict(owners=group,selected_candidates=96,actual_shared_steps=3,real_raw_nodes=192,returned_ABI_slots=256,
    current_shared4_claimed=False,teacher_forced_suffix_tokens={str(i):suffixes[i] for i in repair},finite_actual=finite_actual,finite_independent=finite_reference,
    allphysical_hidden_encoded=True,key=repr(seen['draft'][0]['key']),runtime_validated=True,validated_real_row_count=entry.validated_real_row_count))
  ownKV=epoch.logical_KV(e,[128]*64,[p-128+3 for p in prefixes])
  require_finite('nextnormal all28 KV',candidate_KV,ownKV)
  require(candidate_ledger==ledger and matcher(candidate_KV,ownKV) and actual_out==reference_out,'all28 next written KV/exact independent ledger/token/confidence ABI')
  return dict(passed=True,normal_waves=waveproof,all28_written_KV=True,own_old_shared4_history_rebuilt=True,candidate_KV_used_as_reference=False,
   actual_missing_suffix_lengths=[len(v) for v in suffixes],known_prefix_protected=True,real_invalidation_or_promotion_followed_by_nextdraft=True,Goodput_measured=False)
 result=stage('fresh-independent-old-shared4-and-next3-reference',independent)
 restore_source(e,case,trace,lambda n,f:stage('restore/'+n,f))
 def finish():
  if not e.is_draft:return None
  _,out=waves(e,ids,prefixes,roots,catchups,False);_,expected,KV,ledger=actual_state
  require(out==expected and epoch.ledger(e)==ledger and e.graph_runner._target_outputs_match(KV,epoch.logical_KV(e,[128]*64,[p-128+3 for p in prefixes])),
   'actual complete oldshared4+next3 history restored after own reference')
  return result
 return stage('complete-actual-nextdraft-restoration',finish)
