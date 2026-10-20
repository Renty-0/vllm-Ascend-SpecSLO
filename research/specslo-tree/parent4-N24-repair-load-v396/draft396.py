"""Real parent24->frontier/repair->shared4; fresh own same-M complete oracle."""
import copy
from dataclasses import replace
from base199 import V152,functions,require
from scope396 import bucket,digest
from provider396 import source_contexts
from finite151 import require_finite
import epoch199 as epoch
functions(V152/'draft152.py',globals(),names=('cpu_output','PA','top2','chain','decode','_TTU'))

def contexts(c,changed=False):
 rows=[]
 for i,row in enumerate(source_contexts(c)):
  end=c.prefix+c.repair_vector[i]+1;tokens=list(row['token_ids'][:end]);require(len(tokens)==end,'all source tokens exist through root')
  if changed:tokens[c.prefix:]=[(t+1)%151936 for t in tokens[c.prefix:]]
  rows.append(dict(row,token_ids=tokens,context_length=end))
 return rows

def parent_slots(e,p):
 import torch
 return torch.cat([epoch.slots(e,i,range(p,p+4))for i in range(24,48)])

def fixture(e,c,changed=False,*,observe_parent=False):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan as plan,pack_selected_tree_plan,tree_primary_path
 cs=contexts(c,changed)
 if not e.is_draft:return dict(contexts=cs)
 import observe199
 parents=[plan(2,4,c.prefix,4096,candidate_budget=4)for _ in range(24)];parentroots=[cs[i]['token_ids'][-1]for i in range(24,48)]
 fn=lambda:e.draft_tree_forward(parents,parentroots,list(range(24,48)))
 parent,seen=observe199.run(e,fn)if observe_parent else(fn(),None)
 parentKV=epoch.read(e,parent_slots(e,c.prefix))if observe_parent else None
 plans=[];roots=[];catchups=[];sources={};confidences={};parent_sources={}
 for row,i in enumerate(range(24,48)):
  packed=pack_selected_tree_plan(parents[row],range(4));require(tuple(tree_primary_path(packed))==(0,1,2,3),'true parent linear4')
  values=list(map(int,parent['draft_token_ids'][row][:4]));parent_sources[i]=(packed,values);confidences[i]=sum(float(v)for v in parent['node_confidences'][row][:4])/4
 for i in range(24):
  tokens=cs[i]['token_ids'];plans.append(plan(2,4,len(tokens)-1,4096,candidate_budget=4));roots.append(tokens[-1]);catchups.append(tokens[c.prefix:])
 for i in c.eager_ids:
  packed,values=parent_sources[i];child=e._spec_rhythm_tree_eager_plan(packed,type('Prefix',(),{'token_ids':cs[i]['token_ids']})(),4)
  require(child.prefix_len==c.prefix+5 and child.candidate_budget==4,'original child4 after parent4/frontier')
  plans.append(child);roots.append(cs[i]['token_ids'][-1]);catchups.append(());sources[i]=(packed,values)
 return dict(contexts=cs,plans=plans,roots=roots,ids=list(range(24))+list(c.eager_ids),sources=sources,parent_sources=parent_sources,catchups=catchups,parent_output=cpu_output(parent),parent_confidences=confidences,parent_observed=seen,parent_KV=parentKV,source_epoch=e._v151_source_epoch,source_lease=copy.deepcopy(e._v151_source_lease),parent_actual_execution=dict(actual_steps=4,actual_chain_M=24,query_count=parent['query_count'],graph_calls=parent['graph_calls'],model_calls=parent['model_calls']))

def invoke(e,f):
 require(f['source_epoch']==e._v151_source_epoch,'same actual source lease epoch')
 return e.draft_tree_forward(f['plans'],f['roots'],f['ids'],eager_parent_sources=f['sources'],committed_catchup_token_ids=f['catchups'])

def counts(c):return[4+c.repair_vector[i]if i<24 else 9 if i in c.eager_ids else 4 if i<48 else 0 for i in range(64)]

def union_call(e,c,f,parent_rows):
 """Original real PA + own dummy permutation; not candidate union builder."""
 import torch
 ids=list(c.eager_ids)+list(c.repair_owners);positions=[c.prefix+4]*c.eager+[c.prefix]*c.repair_count
 tokens=[parent_rows[i-24][3]for i in c.eager_ids]+[f['catchups'][i][0]for i in c.repair_owners]
 F=bucket(c.eager);U=c.union_M;x,p,m=PA(e,ids,positions,tokens,U)
 order=list(range(c.eager))+list(range(c.eager+c.repair_count,c.eager+c.repair_count+F-c.eager))+list(range(c.eager,c.eager+c.repair_count))+list(range(c.eager+c.repair_count+F-c.eager,U))
 require(len(order)==len(set(order))==U,'all physical real/interior/tail rows independently permuted')
 def take(v):return v.index_select(0,torch.tensor(order,dtype=torch.long,device=v.device))
 m=replace(m,slot_mapping=take(m.slot_mapping),context_lens=take(m.context_lens),block_tables=take(m.block_tables),actual_seq_lengths_q=tuple(range(1,F+c.repair_count+1))if c.repair_count else tuple(range(1,c.eager+1)),sequence_lens=tuple(m.sequence_lens[i]for i in order))
 return take(x),take(p),m

def metadata_gate(e,c,f,seen,parent_rows):
 from metadata161 import snapshot,observed_snapshot,FIELDS
 current=seen['hidden'][0]if c.repair_count else seen['greedy'][0];expected=snapshot(*union_call(e,c,f,parent_rows));actual=observed_snapshot(current['metadata'])
 require(all(actual[k]==expected[k]for k in FIELDS),'all ten actual same-lease metadata fields from independent original PA')
 real=list(range(c.eager))+list(range(bucket(c.eager),bucket(c.eager)+c.repair_count));owners=list(c.eager_ids)+list(c.repair_owners);positions=[c.prefix+4]*c.eager+[c.prefix]*c.repair_count;tables=e.cache_allocation.block_tables
 require([actual['slot_mapping'][i]for i in real]==[tables[o][p//128]*128+p%128 for o,p in zip(owners,positions,strict=True)]and all(actual['slot_mapping'][i]==-1 for i in set(range(c.union_M))-set(real)),'actual own pages and all interior/tail dummy write slots')
 return dict(fields=actual,real_indices=real,owners=owners,positions=positions,independent_original_PA=True)

def independent(e,c,f):
 from kv_ledger import record_overwrite
 p=c.prefix;baseline=epoch.protect(e,[p]*64)
 encoded_parent,parent_hidden=chain(e,list(range(24,48)),[p]*24,[f['contexts'][i]['token_ids'][-1]for i in range(24,48)],4,24)
 parent_rows,_=decode(encoded_parent,24,4);parentKV=epoch.read(e,parent_slots(e,p));ledger={i:dict(safe_before=p,tokens={})for i in range(64)}
 for row,i in enumerate(range(24,48)):
  require(parent_rows[row][:4]==f['parent_sources'][i][1],'own independent parent4 predictions')
  ledger[i]['tokens']={p+k:t for k,t in enumerate([f['contexts'][i]['token_ids'][-1],*parent_rows[row][:3]])}
 call=union_call(e,c,f,parent_rows);prelude=e.model(*call).clone();frontier=e.model.compute_greedy_tokens(prelude[:bucket(c.eager)],e.draft_vocab_size).clone();frontiers={i:int(frontier[j])for j,i in enumerate(c.eager_ids)}
 for i in c.eager_ids:record_overwrite(ledger[i],p+4,[parent_rows[i-24][3]])
 prefixes=[v.prefix_len for v in f['plans']];roots=[*f['roots'][:24],*[frontiers[i]for i in c.eager_ids]]
 encoded,hidden=chain(e,f['ids'],prefixes,roots,4,bucket(c.count));rows,scores=decode(encoded,c.count,4)
 for i,prefix,root,row in zip(f['ids'],prefixes,roots,rows,strict=True):
  written=[root,*row[:3]]
  if i<24:ledger[i]=dict(safe_before=prefix,tokens={prefix+k:t for k,t in enumerate(written)})
  else:record_overwrite(ledger[i],prefix,written)
 require(epoch.protect(e,[p]*64)==baseline,'all original64 prefixes protected by own full history')
 return dict(rows=rows,scores=scores,encoded=encoded,hidden=hidden,union_hidden=prelude,union_call=call,frontier_tokens=frontier,frontiers=frontiers,roots=roots,ledger=ledger,parent_hidden=parent_hidden,parent_encoded=encoded_parent,parent_KV=parentKV)

def prepare(e,c,stage,changed=False):
 import torch,observe199
 cs=stage('source-contexts',lambda:contexts(c,changed));epoch.rebuild(e,cs,list(c.repair_vector),stage,role='draft')
 f=stage('actual-parent24',lambda:fixture(e,c,changed,observe_parent=True)if e.is_draft else None)
 def actual():
  if not e.is_draft:return None
  baseline=epoch.protect(e,[128]*64);out,seen=observe199.run(e,lambda:invoke(e,f));parentrows=f['parent_output']['draft_token_ids']
  meta=metadata_gate(e,c,f,seen,parentrows);return dict(output=out,seen=seen,KV=epoch.logical_KV(e,[128]*64,counts(c)),ledger=epoch.ledger(e),protected=baseline,metadata=meta)
 a=stage('actual-frontier-union-shared4',actual)
 epoch.rebuild(e,cs,list(c.repair_vector),stage,role='draft')
 ref=stage('fresh-own-parent-frontier-shared4',lambda:independent(e,c,f)if e.is_draft else None)
 def compare():
  if not e.is_draft:return None
  match=e.graph_runner._target_outputs_match;o=a['output'];seen=a['seen'];own=epoch.logical_KV(e,[128]*64,counts(c));prelude=(seen['hidden']if c.repair_count else seen['greedy'])[0]
  require(len(seen['draft'])==1 and len(seen['hidden'])==int(bool(c.repair_count))and len(seen['greedy'])==int(not c.repair_count)and tuple(seen['draft'][0]['encoded'].shape)==(bucket(c.count),4,2),'actual full physical bucket complete four-step physical raw8 and exact prelude path')
  require(match((prelude['hidden'],),(ref['union_hidden'],))and match(seen['draft'][0]['hidden'],ref['hidden'])and torch.equal(seen['draft'][0]['encoded'],ref['encoded'])and len(a['KV'])==len(own)==56 and match(a['KV'],own),'allphysical prelude+fourstep hidden/encoded/all56 queryKV')
  require(o['draft_token_ids']==ref['rows']and o['root_token_ids']==ref['roots']and o['eager_frontier_tokens']==ref['frontiers']and a['ledger']==ref['ledger'],'exact real canonical8/frontier/true ledger')
  require(all(torch.equal(v,torch.tensor(s))for v,s in zip(o['node_confidences'],ref['scores'],strict=True))and torch.equal(o['draft_confidence'],torch.tensor([sum(s)/8 for s in ref['scores']])),'full8 quantized confidences and original denominator8')
  par=f['parent_observed'];require(len(par['draft'])==1 and match(par['draft'][0]['hidden'],ref['parent_hidden'])and torch.equal(par['draft'][0]['encoded'],ref['parent_encoded'])and match(f['parent_KV'],ref['parent_KV']),'independent allphysical originalM16 parent4/56KV')
  head=seen['union_head'][0]['tokens']if c.repair_count else prelude['tokens'];require(tuple(head.shape)==(bucket(c.eager),)and torch.equal(head,ref['frontier_tokens']),'complete physical F frontier head, never E-row recompute')
  require_finite('actual P4/E4',prelude['hidden'],seen['draft'][0]['hidden'],a['KV']);require_finite('own P4/E4',ref['union_hidden'],ref['hidden'],own)
  entries=[]
  for kind in('draft','hidden','greedy'):
   for row in seen[kind]:
    family=e.graph_runner.draft_entries if kind=='draft'else e.graph_runner.target_entries;entry=family[row['key']];require(entry.runtime_validated is True,'natural Native flag only')
    if kind=='draft':require(entry.validated_real_row_count>=c.count,'actual real rows verified')
    entries.append(dict(kind=kind,key=repr(row['key']),object_id=id(entry),runtime_validated=True,validated_real_row_count=getattr(entry,'validated_real_row_count',None)))
  return dict(full_physical_encoded=seen['draft'][0]['encoded'].cpu().tolist(),reference_full_physical_encoded=ref['encoded'].cpu().tolist(),candidate_hidden_hashes=[epoch.tensor_hash(prelude['hidden']),*[epoch.tensor_hash(v)for v in seen['draft'][0]['hidden']]],reference_hidden_hashes=[epoch.tensor_hash(ref['union_hidden']),*[epoch.tensor_hash(v)for v in ref['hidden']]],candidate_queryKV_hashes=[epoch.tensor_hash(v)for v in a['KV']],reference_queryKV_hashes=[epoch.tensor_hash(v)for v in own],sameM=True,parent_M=24,shared_M=bucket(c.count),F=bucket(c.eager),U=c.union_M,allphysical_hidden_encoded=True,all56_queryKV=True,canonical8_confidence_ledger=True,metadata=a['metadata'],entries=entries,raw_nodes=8*c.count,selected=96+4*c.eager,source_KV_copied=False)
 proof=stage('independent-fullphysical-encoded-ledger-KV-gate',compare)
 epoch.rebuild(e,cs,list(c.repair_vector),stage,role='draft')
 restored=stage('restore-parent24',lambda:fixture(e,c,changed)if e.is_draft else None)
 def restore():
  if not e.is_draft:return None
  out=invoke(e,restored);require(cpu_output(out)==cpu_output(a['output'])and epoch.ledger(e)==a['ledger']and e.graph_runner._target_outputs_match(a['KV'],epoch.logical_KV(e,[128]*64,counts(c)))and epoch.protect(e,[128]*64)==a['protected'],'own-reference round restores entire actual source before protocol')
  return dict(fixture=restored,mixed=cpu_output(out),numerical_proof=proof,changed=changed,source_lease=restored['source_lease'])
 trace=stage('restore-actual-source-gate',restore)
 def share():
  import torch.distributed as dist
  packet=[trace if e.is_draft else None];dist.broadcast_object_list(packet,src=e.topology.draft_leader_rank,group=e.groups.verification_coordination_group);require(packet[0]is not None,'actual draft source shared once');return packet[0]
 return stage('share-current-source',share)
