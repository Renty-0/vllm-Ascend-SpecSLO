"""HOST-only original cohort/common-history replay fixtures, all real calls q80."""
from pathlib import Path
import json,gzip,hashlib,datetime
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-incremental-common-KV-history-forensics-v222';assert not P.exists();P.mkdir()
a=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/packed-KV-rolefix-root-v218-full463-AR-v1';tp=a/'windows/window32/full463/rank1-full463-actual-trace.json.gz';lp=D/'ROOT_V218_ALL_OWN_AR_FIRST_DIFFERENCE_LOCATIONS.json'
def read(p):
 raw=p.read_bytes();return json.loads(gzip.decompress(raw)if p.suffix=='.gz'else raw)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
x=read(tp);locations=read(lp);sources={str(tp):sha(tp),str(lp):sha(lp)};cand=[r['completion_token_ids']for r in x['generate_results']];by_id={r['request_id']:r['completion_token_ids']for r in x['generate_results']};cand=[by_id[r['request_id']]for r in x['requests']]
normal=[];seen=set()
for ev in x['stream_events']:
 if ev['request_index']in seen:normal.append(ev)
 else:seen.add(ev['request_index'])
chunks=[];cursor=0
for call in x['normal_target_calls']:
 ev=normal[cursor:cursor+call['N']];cursor+=call['N'];assert [v['request_index']for v in ev]==call['sequence_ids'][:call['N']];chunks.append(ev)
assert cursor==len(normal)
cases=[]
for owner in(129,34):
 diff=next(r for r in locations['first_differences']if r['request_index']==owner);last=diff['location']['call_ordinal'];cohort_idx=next(i for i,c in enumerate(x['standalone_prefill_calls'])if owner in c['sequence_ids']);group=x['standalone_prefill_calls'][cohort_idx]['sequence_ids'];assert len(group)==32
 refp=a/'postfull-independent-FIA-AR'/f'rank1-cohort{cohort_idx}.json.gz';ref=read(refp);sources[str(refp)]=sha(refp);assert ref['original_owner_indices']==group
 local={o:i for i,o in enumerate(group)};prompts=[x['tokenized_inputs'][o]for o in group];outputs=[cand[o]for o in group];references=ref['reference_completions'];own=local[owner];k=diff['first_difference_zero_based'];assert outputs[own][:k]==references[own][:k]
 candidate_frames=[];offset=[1]*32
 for ordinal,(call,evs)in enumerate(zip(x['normal_target_calls'],chunks,strict=True)):
  if ordinal>last:break
  ids=call['sequence_ids'][:call['N']]
  if not set(ids)&set(group):continue
  assert set(ids)<=set(group)and call['N']==16 and call['budgets']==[4]*16 and(call['physical_M'],call['physical_S'],call['family'])==(80,16,'causal')
  selected=[local[o]for o in ids];tokens=[];positions=[];lens=[]
  for seq,context,ev in zip(selected,call['contexts'],evs,strict=True):
   assert context==len(prompts[seq])+offset[seq]-1
   q=[outputs[seq][offset[seq]-1]]+outputs[seq][offset[seq]:offset[seq]+4];assert len(q)==5
   tokens.extend(q);positions.extend(range(context,context+5));lens.append(context+5)
   assert ev['token_ids']==outputs[seq][offset[seq]:offset[seq]+len(ev['token_ids'])];offset[seq]+=len(ev['token_ids'])
  candidate_frames.append(dict(ordinal=ordinal,ids=[i for seq in selected for i in[seq]*5],tokens=tokens,positions=positions,selected=selected,sequence_lens=lens,actual_q=list(range(5,81,5)),final=ordinal==last))
 assert candidate_frames[-1]['final'];lastframe=candidate_frames[-1];cq=5*lastframe['selected'].index(own)+diff['location']['delivered_chunk_offset']
 reference_frames=[];begin=(own//16)*16
 for start in range(0,begin+1,16):
  selected=list(range(start,start+16));count=255 if start<begin else k
  for step in range(count):
   tokens=[];positions=[];lens=[]
   for seq in selected:
    pos=len(prompts[seq])+step;tokens.extend([references[seq][step],0,0,0,0]);positions.extend(range(pos,pos+5));lens.append(pos+5)
   reference_frames.append(dict(step=step,group_begin=start,ids=[i for seq in selected for i in[seq]*5],tokens=tokens,positions=positions,selected=selected,sequence_lens=lens,actual_q=list(range(5,81,5)),final=start==begin and step==count-1))
 rq=5*(own%16);rf=reference_frames[-1];assert rf['final']and(lastframe['positions'][cq],lastframe['tokens'][cq])==(rf['positions'][rq],rf['tokens'][rq])
 reserve=[max(max(f['positions'][j]for f in candidate_frames+reference_frames for j,i in enumerate(f['ids'])if i==seq)+1,len(p))for seq,p in enumerate(prompts)]
 pages=sum((n+127)//128 for n in reserve)+64;assert pages<=544
 cases.append(dict(name='request'+str(owner)+'-common-incremental-history',original_owner=owner,request_id=diff['request_id'],first_difference=k,cohort_ordinal=cohort_idx,original_owner_indices=group,prompts=prompts,original_first_roots=ref['first_roots'],affected_owner=own,common_context=prompts[own]+outputs[own][:k],candidate_frames=candidate_frames,reference_frames=reference_frames,
  candidate_query_row=cq,reference_query_row=rq,actual_candidate_token=outputs[own][k],actual_reference_token=references[own][k],reserve_lengths=reserve,page_upper=pages,
  teacher_forced_common_history_diagnostic=True,not_exact_production_unaccepted_future_inputs=True,not_an_independent_AR_qualification=True))
fixture=dict(schema='ROOT222.same-original-prompt-cohort-incremental-prefix-diagnostic.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=sources,cases=cases,NPU_used=False,performance=False,adopted=False)
(P/'FIXTURE222.json').write_text(json.dumps(fixture,indent=2)+'\n');print(json.dumps([dict(owner=c['original_owner'],candidate_calls=len(c['candidate_frames']),reference_calls=len(c['reference_frames']),pages=c['page_upper'],matching_query_token_position=True)for c in cases]))
