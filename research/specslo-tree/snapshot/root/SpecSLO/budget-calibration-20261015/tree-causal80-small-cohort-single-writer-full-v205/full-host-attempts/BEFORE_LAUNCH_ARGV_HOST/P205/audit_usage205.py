"""Recompute retained integer verdict/history/Native arithmetic, not tensors."""
import gzip,json
from pathlib import Path
from base205 import require,sha
from mapper205 import GREEDY,HIDDEN
from scope_v146 import NATIVE_COUNTERS,FAILURES

def load(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def verdict(rows,head):
 require(len(head)==5*len(rows)and all(len(v)==4 for v in rows),'actual real N B4 BONUS heads')
 out=[];paths=[]
 for i,row in enumerate(rows):
  h=head[5*i:5*i+5];a=0
  while a<4 and row[a]==h[a]:a+=1
  out.append(row[:a]+[h[a]]);paths.append(list(range(a))+[-1]*(4-a))
 return out,paths

def audit(directory):
 p=Path(directory);reports=[load(p/f'rank{r}-component.json.gz')for r in range(4)];hist=[];labels=[];cross=[];zeros=0;pins={};updates=0
 for rank,r in enumerate(reports):
  require(r['schema']=='V205.seven-real-N-current-usage.v1'and r['rank']==rank and r['completed']is r['cache_released']is r['math_independent_qualified']is True
   and r['errors']==[]and r['formal_Goodput']is r['adopted']is r['Native_flags_assigned']is r['whole_AR']is False,'complete current usage report')
  require([(t['pass_index'],t['N'])for t in r['traces']]==[(a,n)for a in range(3)for n in range(9,16)],'all seven actual scopes and two complete zero repeats')
  history=[];compatible=[]
  for t in r['traces']:
   a,n=t['pass_index'],t['N'];native=t['Native'];require(set(native['before'])==set(native['after'])==set(native['delta'])==set(NATIVE_COUNTERS)
    and native['delta']=={k:native['after'][k]-native['before'][k]for k in NATIVE_COUNTERS}and all(v>=0 for v in native['delta'].values())
    and not any(native['delta'][k]for k in FAILURES),'actual21 Native subtraction and no fallback')
   if a:require(not any(native['delta'].values()),'entire real protocol+next zero Native trace');zeros+=21
   require(t['lease']['lease']['pages']==128 and t['lease']['lease']['candidate_owners']==list(range(16))
    and t['lease']['lease']['reference_owners']==list(range(16,32))and t['lease']['lease']['dummy_owners']==list(range(463,527)),'full16+16 and all64 real scratch actual128/544 pages')
   history.append((a,n,t['logical']));trace_compatible=[]
   for cycle in range(4):
    path=p/f'rank{rank}-pass{a}-N{n}-cycle{cycle}.json.gz';row=load(path);pins[str(path)]=sha(path)
    commit=row['commit'];require(row['N']==n and row['source']['ids']==list(range(n))and len(row['rows'])==n and row['selected_normal_C']==4*n
     and row['full16_draft']is True and row['selected']==[list(range(4))]*n and len(row['draft']['variable_num_draft_tokens'])==16
     and len(commit['actual_logical_tokens'])==n and type(commit['actual_structural_failed'])is int and type(commit['actual_numerical_failed'])is int
     and commit['actual_structural_failed']==commit['actual_numerical_failed']==0 and commit['original_finish_cycle_executed']is True
     and row['independent_commit']['histories_equal']is row['independent_commit']['unselected_owners_preserved']is True
     and row['independent_commit']['candidate_KV_copied']is False,'trueN typed original verdict/commit with protected unselected owners')
    if rank==0:
     d=row['draft'];require(d['raw_encoded_shape']==[16,4,2]and d['all_physical_hidden_original_matcher']and d['all28_query_KV_original_matcher']and d['exact_full8_encoded']and d['source_KV_copied']is False,'current original full16 independent all56 draft KV and complete raw8')
    else:
     target=row['target'];ref=row['reference'];x=row['crossM'];require(target['production_greedy_M']==target['separate_hidden_counterpart_M']==80
      and target['all_physical_hidden']and target['all64_query_KV']and target['M1_head_recompute']is False
      and ref['same_M']==80 and ref['same_S']==16 and ref['independent_owners']==list(range(16,16+n))and ref['candidate_KV_copied']is False
      and ref['all_physical_hidden_original_matcher']and ref['all64_query_KV_original_matcher']and ref['all_physical_greedy_exact'],'allphysical80 hidden/head/128queryKV own reference including dummy')
     for name,key in(('greedy',GREEDY),('hidden',HIDDEN)):
      phases=target[name];require([v['phase']for v in phases]==['first','changed','restored']and all(v['key']==repr(key)and v['runtime_validated_before']is v['runtime_validated_after']is True for v in phases),'same existing natural object F/T/F, not fakecold')
      for v in phases:require(set(v['execution'])=={'mode','fallback_reason','capture_attempted','replay_executed'}and v['execution']['mode']=='replay'and v['execution']['replay_executed']is True and v['execution']['fallback_reason']is None,'actual original four-field Native execution')
     require(len(x['new_all80'])==80 and len(x['old_all128'])==128 and x['new_all80'][:5*n]==x['new_real']
      and x['old_all128'][:5*n]==x['old_real']and ref['real_query_tokens']==x['new_real'],'retained complete physical head and true realQ')
     positions=[i for i,(u,v)in enumerate(zip(x['old_real'],x['new_real'],strict=True))if u!=v];require(x['different_head_positions']==positions and x['head_equal']==(not positions),'honest crossM differences')
     actual,pathnodes=verdict(row['rows'],x['new_real']);old,oldnodes=verdict(row['rows'],x['old_real'])
     require(actual==commit['committed_tokens']and pathnodes==commit['accepted_paths']==x['new_accepted']
      and [[v for v in row if v>=0]for row in x['new_verdict_tokens']]==actual
      and [[v for v in row if v>=0]for row in x['old_verdict_tokens']]==old and x['old_accepted']==oldnodes,'integer BONUS verdict and branch path recomputation')
     equal=not positions and x['new_verdict_tokens']==x['old_verdict_tokens']and pathnodes==oldnodes
     require(x['verdict_equal']==(x['new_verdict_tokens']==x['old_verdict_tokens'])and x['accepted_equal']==(pathnodes==oldnodes),'crossM compatibility classification')
     trace_compatible.append(equal)
     seen=row['actual_length_observations'];new=[v for v in seen if v['M']==80];require(len(new)==3 and all(v['N']==n and v['S']==16 and v['host_lengths']['q']==[5]*16 and v['host_lengths']['KV'][n:]==[5]*(16-n)for v in new),'actual real/dummy same-key host signatures')
     for v in new:
      require(v['delta']=={k:v['after'][k]-v['before'][k]for k in v['before']}and all(x>=0 for x in v['delta'].values()),'original actual task update arithmetic')
      if v['previous_same_key']is not None and v['host_lengths']!=v['previous_same_key']:updates+=v['delta']['aclgraph_target_task_update_replays']
   require(t['crossM_behavior_equal']is(None if rank==0 else all(trace_compatible)),'actual trace compatibility')
   if rank:compatible.extend(trace_compatible)
   path=p/f'rank{rank}-pass{a}-N{n}-next.json.gz';nxt=load(path);pins[str(path)]=sha(path)
   require({k:nxt[k]for k in('source','selected','rows','confidences')}==t['logical']['next'],'actual current own next retained')
  for n in range(9,16):require([t['logical']for t in r['traces']if t['N']==n].count(next(t['logical']for t in r['traces']if t['N']==n))==3,'whole independent history reproduced twice')
  require(r['inventory']['zero_new_keys']is r['inventory']['objects_retained']is r['inventory']['flags_unchanged']is True
   and r['MC2']['delta']['fallback']==r['MC2']['delta']['exception']==0 and(rank==0 or r['MC2']['delta']['native_epilogue_success']>0),'actual no new keys/flags and existing MC2')
  require(r['gain_reset']['after']['observed_after']==[]and r['gain_reset']['after']['calibrated_prior_unchanged']is True and r['fresh_boundary']['known_KV_ledger_empty']is True,'no fixture policy/ledger leakage')
  events=r['boundary_events'];require([v['sequence']for v in events]==list(range(len(events)))and all(len(v['ranks'])==4 and all(q['label']==v['label']and q['sequence']==v['sequence']and q['error']is None for q in v['ranks'])for v in events),'common all4 failure-safe stages')
  hist.append(history);labels.append([(v['sequence'],v['label'])for v in events]);cross.append(compatible)
 require(all(v==hist[0]for v in hist)and all(v==labels[0]for v in labels)and cross[1]==cross[2]==cross[3],'all4 logical history/stages and target classifications')
 require(zeros==4*7*2*21 and updates>0,'all1176 zero differences and actual changed host-length task updates')
 same=all(cross[1]);return dict(schema='V205.retained-seven-use-audit.v1',passed=True,math_independent_qualified=True,N=list(range(9,16)),complete_protocol_commits=84,zero_traces=14,Native_zero_subtractions=zeros,
  changed_host_length_TTU_replays=updates,crossM_behavior_equal=same,numerical_variant_only=not same,zero_new_keys=True,actual_current_entry_objects=True,
  HOST_recomputes_unretained_hidden_or_KV=False,whole_AR=False,formal_Goodput=False,adopted=False,raw_checkpoint_sha256=pins)
