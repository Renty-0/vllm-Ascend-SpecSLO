"""Fresh multiowner eager FIA math and real root→nextdraft, no new graphs."""
import copy,gzip,json
from pathlib import Path
from base202 import require,parent_module
from dual_bank_v146 import _slots,prefix_evidence,query_KV,tensor_hash
from scope_v146 import digest,NATIVE_COUNTERS,FAILURES

def save(directory,name,row):
 path=Path(directory)/name
 with path.open('xb') as f:f.write(gzip.compress(json.dumps(row,allow_nan=False,separators=(',',':')).encode(),mtime=0))

def prompts(engine):
 from production_probe_v101 import request_rows
 from poisson_coverage_probe import tokenize_prefix
 rows=request_rows(65);values,_=tokenize_prefix(engine.tokenizer,rows)
 result=[dict(request_id=r['request_id'],token_ids=list(t),length=len(t),original_prompt=r['prompt']) for r,t in zip(rows,values,strict=True)]
 require(len(result)==65 and len({v['length']for v in result[:32]})>1 and max(v['length']for v in result)<=2301,'real mixed-length original prompt owners')
 return result

class Bank:
 def __init__(self,engine,rows,n):
  self.engine=engine;self.n=n;self.rows=rows[:n];self.ids=list(range(n));self.reference_ids=list(range(n,2*n));self.prompts=[list(r['token_ids']) for r in self.rows]
  self.incumbent=2*n;self.incumbent_tokens=list(rows[-1]['token_ids'])
  require(engine.cache_allocation is None and engine.cache_block_tables is None,'fresh genuine own history allocation')
  engine._allocate_cache(self.prompts,enable_prefix_caching=False,reserve_sequence_capacity=True)
  for i,p in enumerate(self.prompts):engine._activate_cache_sequence(n+i,p,enable_prefix_caching=False)
  engine._activate_cache_sequence(self.incumbent,self.incumbent_tokens,enable_prefix_caching=False)
  # Original allocator activates every prompt block; reserve future root/chain
  # blocks explicitly at each queried logical page, never only lastblock.
  for owner,p in [(i,p) for i,p in enumerate(self.prompts)]+[(n+i,p) for i,p in enumerate(self.prompts)]:
   for pos in range(0,len(p)+9,128):engine._ensure_cache_capacity([owner],[pos])
  self.contexts=[];self.reference_contexts=[]
  self.reference_ledger={n+i:dict(safe_before=len(p),tokens={}) for i,p in enumerate(self.prompts)}
  self.tables=copy.deepcopy(engine.cache_allocation.block_tables);self.allocation=id(engine.cache_allocation)
  if engine.is_draft:engine._dynamic_tree_known_KV={i:dict(safe_before=len(p),tokens={}) for i,p in enumerate(self.prompts)}
 def guard(self):
  pages=[v for row in self.engine.cache_allocation.block_tables for v in row if v>=0]
  require(id(self.engine.cache_allocation)==self.allocation and self.tables==self.engine.cache_allocation.block_tables and len(pages)==len(set(pages)),'own stable unique actual allocator pages')
  return dict(actual_unique_pages=len(pages),pages=pages,storage_rows=len(self.tables),candidate_owners=list(range(self.n)),reference_owners=list(range(self.n,2*self.n)),incumbent_owner=self.incumbent,
   block_tables=copy.deepcopy(self.tables),candidate_prefix_copied=False)
 def source(self):
  require(self.contexts==self.reference_contexts,'actual roots independently agree before nextdraft')
  return dict(ids=list(range(self.n)),contexts=[dict(owner=i,reference_owner=i+self.n,request_id=r['request_id'],token_ids=list(v),prefix=len(v)-1,initial_prefix=len(self.prompts[i]),output_offset=0)
   for i,(r,v) in enumerate(zip(self.rows,self.contexts,strict=True))])

def packed(bank,*,reference=False):
 n=bank.n;tokens=[];ids=[];positions=[];last=[]
 for i,p in enumerate(bank.prompts):
  tokens.extend(p);ids.extend([i+n if reference else i]*len(p));positions.extend(range(len(p)));last.append(len(tokens)-1)
 return tokens,ids,positions,last

def metadata_record(engine,inputs,pos,meta,ids):
 tables=engine.cache_allocation.block_tables;owners=[]
 for i in ids:
  if not owners or owners[-1]!=i:owners.append(i)
 return dict(inputs=inputs.cpu().tolist(),positions=pos.cpu().tolist(),owners=owners,sequence_ids=ids,
  slots=meta.slot_mapping.cpu().tolist(),contexts=meta.context_lens.cpu().tolist(),actual_seq_lengths_q=list(meta.actual_seq_lengths_q),sequence_lens=list(meta.sequence_lens),
  block_tables=meta.block_tables.cpu().tolist(),request_block_tables=meta.request_block_tables.cpu().tolist(),
  mask_shape=list(meta.attention_mask.shape),mask_dtype=str(meta.attention_mask.dtype),cached_mask_identity=meta.attention_mask is engine.model.attention_mask,
  fused=meta.use_fused_infer_attention,field_dtypes=dict(context_lens=str(meta.context_lens.dtype),slot_mapping=str(meta.slot_mapping.dtype),block_tables=str(meta.block_tables.dtype)))

def capture_begin(engine):
 model=engine.model;had='forward' in vars(model);prior=vars(model).get('forward');original=model.forward;saved=[]
 def capture(*args,**kwargs):
  value=original(*args,**kwargs)
  saved.append((args[0].clone(),args[1].clone(),args[2],value.clone()))
  return value
 model.forward=capture
 def restore():
  if had:model.forward=prior
  else:vars(model).pop('forward',None)
 return saved,restore

def run_case(engine,native,stage,rows,n,pass_index,directory):
 import torch
 lane=lambda label,fn:stage(f'multiowner202/N{n}/pass{pass_index}/'+label,fn)
 before=lane('Native-before',lambda:{k:engine.graph_metrics()[k] for k in NATIVE_COUNTERS})
 bank=lane('independent-allocation',lambda:Bank(engine,rows,n))
 hidden=lane('incumbent-original-model',lambda:engine._run_packed_hidden(bank.incumbent_tokens,[bank.incumbent]*len(bank.incumbent_tokens),list(range(len(bank.incumbent_tokens))),use_aclgraph=False,use_fused_infer_attention=True).clone())
 def protect():
  require(bool(torch.isfinite(hidden).all()),'complete real incumbent hidden finite')
  return prefix_evidence(engine,{bank.incumbent:bank.incumbent_tokens})
 protected=lane('incumbent-protected-prefix',protect)
 saved,restore=lane('capture-original-fullphysical-hidden',lambda:capture_begin(engine))
 try:
  sampled=lane('original-multiowner-eager-helper-vote-broadcast',lambda:engine._prefill_and_sample_target_batch(bank.prompts,[native.PearlPipelineState(list(p),len(p),max_tokens=256,ignore_eos=True) for p in bank.prompts],list(range(n))))
 finally:lane('restore-model-forward-exact',restore)
 def drain():
  require(len(saved)==1 and len(sampled)==n and all(type(v) is int and 0<=v<engine.draft_vocab_size for v in sampled),'one complete real multiowner eager call and original broadcast roots')
  x,pos,meta,h=saved[0];tt,ids,pp,last=packed(bank)
  require(x.cpu().tolist()==tt and pos.cpu().tolist()==pp and meta.use_fused_infer_attention is True and meta.attention_mask is engine.model.attention_mask
   and meta.attention_mask.dtype==torch.int8 and tuple(meta.attention_mask.shape)==(2048,2048),'actual original packed FIA cached int8 ABI')
  values=query_KV(engine,meta.slot_mapping.to(device=engine.device,dtype=torch.long))
  require(len(values)==(56 if engine.is_draft else 128) and all(bool(torch.isfinite(v).all()) for v in (h,*values)),'all physical hidden and every written queryKV finite')
  return h,values,metadata_record(engine,x,pos,meta,ids)
 candidate,kv,record=lane('fullphysical-all-queryKV-drain',drain)
 def reference_prepare():
  tt,ids,pp,last=packed(bank,reference=True)
  # This calls the model's original producer on its OWN allocation rows;
  # neither candidate metadata nor any candidate KV populates the oracle.
  slots=engine._cache_slot_mapping(ids,pp)
  pos,meta=engine.model.make_attention_metadata(ids,pp,engine.cache_allocation.block_tables,slots,True)
  return torch.tensor(tt,dtype=torch.long,device=engine.device),pos,meta,ids,last
 ref=lane('independent-ownmetadata-prepare',reference_prepare)
 rh=lane('independent-sameM-original-model',lambda:engine.model(ref[0],ref[1],ref[2]).clone())
 def ref_finite():
  rkv=query_KV(engine,ref[2].slot_mapping.to(device=engine.device,dtype=torch.long));matcher=engine.graph_runner._target_outputs_match
  require(len(rkv)==len(kv) and all(bool(torch.isfinite(v).all()) for v in (rh,*rkv)) and matcher((candidate,),(rh,)) and matcher(kv,rkv),'independent fullphysical hidden and every queryKV original matcher')
  require(prefix_evidence(engine,{bank.incumbent:bank.incumbent_tokens})==protected,'all incumbent actual prefix KV protected')
  return metadata_record(engine,ref[0],ref[1],ref[2],ref[3])
 refrecord=lane('independent-metadata-and-all-queryKV-match',ref_finite)
 heads=lane('independent-same-cohort-N-lasttoken-head',lambda:engine.model.compute_greedy_tokens(rh[ref[4]],engine.draft_vocab_size).clone() if not engine.is_draft else None)
 def head_match():
  if engine.is_draft:return None
  require(tuple(heads.shape)==(n,) and heads.cpu().tolist()==sampled,'original helper roots equal same N head, never M1 recompute')
  return heads.cpu().tolist()
 headrow=lane('independent-lasttoken-root-match',head_match)
 from publication202 import publish
 publication=lane('original-root-append-and-once-callback',lambda:publish(engine,native,bank,sampled))
 # Planner-generated subset only: wide prefill does not enlarge target/draft scope.
 from subset202 import prepare,guard_rest,finish
 selected=lane('original-planner-next-subset',lambda:prepare(engine,native,bank))
 protected_rest=lane('protect-all-nonselected-owner-prefixes',lambda:guard_rest(engine,bank,selected))
 work=__import__('next4_202').generate(engine,native,selected,dict(N=selected.n,budgets=[4]*selected.n,C=4*selected.n,ids=selected.ids),1000+n+pass_index,lane)
 planning=lane('original-ready-ticket-target-admission',lambda:finish(engine,native,bank,selected,work,protected_rest))

 def final():
  require(prefix_evidence(engine,{bank.incumbent:bank.incumbent_tokens})==protected,'incumbent prefix also survives true nextdraft')
  after={k:engine.graph_metrics()[k] for k in NATIVE_COUNTERS};delta={k:after[k]-before[k] for k in before}
  require(all(delta[k]==0 for k in FAILURES) and (pass_index==0 or not any(delta.values())),'current eager and subsequent true nextdraft no failure/two zero passes')
  return dict(schema='V202.current-multiowner-eager-cycle.v1',rank=engine.rank,N=n,pass_index=pass_index,
   prompt_requests=[dict(request_id=r['request_id'],length=r['length'],sha256=digest(r['token_ids'])) for r in rows[:n]],
   candidate_metadata=record,independent_metadata=refrecord,all_physical_hidden_original_matcher=True,
   all_query_KV_tensors=56 if engine.is_draft else 128,all_query_KV_original_matcher=True,incumbent_prefix_protected=True,
   independent_same_N_heads=headrow,publication=publication,next_draft=dict(rows=work['rows'],selected=work['selected'],draft_proof=work['draft_proof'],planner=planning),
   lease=bank.guard(),Native=dict(before=before,after=after,delta=delta),eager_original_multiowner=True,
   singleton201_authorization_used=False,candidate_KV_copied=False,full32_numerical_exhaustive=False)
 row=lane('actual-current-math-summary',final)
 lane('retain-actual-current-case',lambda:save(directory,f'rank{engine.rank}-N{n}-pass{pass_index}.json.gz',row))
 lane('release-independent-prefixes',engine._release_cache)
 return dict(N=n,pass_index=pass_index,Native=row['Native'],root_tokens=sampled,query_count=sum(len(p) for p in bank.prompts),all_query_KV_tensors=row['all_query_KV_tensors'])

def run(engine,native,stage,output):
 import torch.distributed as dist
 directory=Path(output)/'multiowner-eager-prefill-own-current-process'
 stage('multiowner202/create-output',lambda:directory.mkdir(parents=True,exist_ok=False) if engine.rank==engine.topology.draft_leader_rank else None)
 rows=stage('multiowner202/original-full-prompt-source',lambda:prompts(engine))
 inventory=stage('multiowner202/current-entry-objects-before',lambda:{name:dict(getattr(engine.graph_runner,name)) for name in ('entries','draft_entries','target_entries')})
 result=[]
 for p in range(3):
  for n in (2,32):result.append(run_case(engine,native,stage,rows,n,p,directory))
 def done():
  require(engine.cache_allocation is None and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,'current multiowner evidence released before production seal')
  for name,entries in inventory.items():
   now=getattr(engine.graph_runner,name)
   require(set(now)==set(entries) and all(now[k] is v and v.runtime_validated is True for k,v in entries.items()),'same actual Native dictionary/object/qualified flags '+name)
  return dict(schema='V202.current-multiowner-component.v1',rank=engine.rank,cases=result,errors=[],cache_released=True,
   representative_sizes=[2,32],passed_current_multiowner_math=True,full32_exhaustive=False,Native_flags_assigned=False,
   actual_Native_inventory={name:sorted(map(repr,entries)) for name,entries in inventory.items()},actual_entry_objects_preserved=True)
 report=stage('multiowner202/local-summary',done)
 stage('multiowner202/retain-summary',lambda:save(directory,f'rank{engine.rank}-component.json.gz',report))
 def gather():
  values=[None]*4;dist.all_gather_object(values,dict(rank=engine.rank,completed=True),group=engine.groups.verification_coordination_group)
  require(sorted(v['rank'] for v in values)==list(range(4)),'all4 retained before independent raw audit')
 stage('multiowner202/all4-retained',gather)
 value=stage('multiowner202/reopen-independent-raw-audit',lambda:audit(directory))
 report=dict(report,audit=value);engine._v202_actual_prefill=report;return report

def audit(directory):
 def read(path):
  with gzip.open(path,'rt') as f:return json.load(f)
 reports=[read(Path(directory)/f'rank{r}-component.json.gz') for r in range(4)];zeros=0;roots={}
 for rank,r in enumerate(reports):
  require(r['rank']==rank and r['passed_current_multiowner_math'] is r['cache_released'] is r['actual_entry_objects_preserved'] is True and r['Native_flags_assigned'] is False,'actual complete multiowner report')
  require([(c['pass_index'],c['N']) for c in r['cases']]==[(p,n) for p in range(3) for n in (2,32)],'actual2/32 plus two full zero passes')
  for p in range(3):
   for n in (2,32):
    row=read(Path(directory)/f'rank{rank}-N{n}-pass{p}.json.gz');a=row['candidate_metadata'];b=row['independent_metadata'];lens=[v['length'] for v in row['prompt_requests']]
    require(row['rank']==rank and row['N']==n and len(set(lens))>1 and row['all_physical_hidden_original_matcher'] is row['all_query_KV_original_matcher'] is row['incumbent_prefix_protected'] is True
     and row['all_query_KV_tensors']==(56 if rank==0 else 128),'retained true fullphysical/role queryKV proof; not HOST recomputed tensors')
    for meta,off in ((a,0),(b,n)):
     ids=[i+off for i,l in enumerate(lens) for _ in range(l)];positions=[j for l in lens for j in range(l)];ends=[];total=0
     for l in lens:total+=l;ends.append(total)
     tables=[row['lease']['block_tables'][i+off] for i in range(n)]
     slots=[tables[i][j//128]*128+j%128 for i,l in enumerate(lens) for j in range(l)]
     require(meta['sequence_ids']==ids and meta['positions']==positions and meta['contexts']==[j+1 for j in positions] and meta['slots']==slots
      and meta['actual_seq_lengths_q']==ends and meta['sequence_lens']==lens and meta['block_tables']==meta['request_block_tables']==tables
      and meta['fused'] is meta['cached_mask_identity'] is True and meta['mask_dtype']=='torch.int8' and meta['mask_shape']==[2048,2048],'independently recompute actual packed query/KV/table/slot/causal ABI')
    require(a['inputs']==b['inputs'] and len(a['inputs'])==sum(lens) and row['candidate_KV_copied'] is False and row['singleton201_authorization_used'] is False,'own sameM inputs and no singleton authorization')
    tables=row['lease']['block_tables'];pages=[v for table in tables for v in table if v>=0]
    require(len(tables)==527 and row['lease']['pages']==pages and len(pages)==len(set(pages))==row['lease']['actual_unique_pages'],'independent actual allocated page inventory')
    nextrow=row['next_draft'];pn=1 if n==2 else 16;planning=nextrow['planner'];require(len(planning['actual_next_owners'])==pn and planning['target_max16']is True and planning['N32_target_scope_added']is False and planning['untouched_owner_KV_protected']is True,'actual original home planner and protected remaining owners');require(nextrow['selected']==[list(range(4))]*pn and len(nextrow['rows'])==pn and all(len(v)==4 for v in nextrow['rows']),'actual original root consumed by genuine next B4 publication')
    if rank==0:
     d=nextrow['draft_proof'];require(d['all_physical_hidden_original_matcher'] is d['all28_query_KV_original_matcher'] is d['exact_full8_encoded'] is True
      and d['actual_draft_model_stages']==4 and d['raw_encoded_shape'][1:]==[4,2]
      and d['source_KV_copied'] is False and d['independent_owners']==planning['actual_reference_owners'],'actual sameM four-step next all56draftKV reference')
    root=row['publication']['root_tokens'];require(row['publication']['original_append_and_publish_AST'] is True and row['publication']['original_callback'] is True
     and len(root)==n and len(row['publication']['callbacks'])==(n if rank==1 else 0),'actual original publication once')
    if rank:require(row['independent_same_N_heads']==root,'same N independent head exact')
    key=(p,n)
    if key in roots:require(roots[key]==root,'actual all4 original root broadcast agreement')
    else:roots[key]=root
    d=row['Native'];require(set(d['before'])==set(d['after'])==set(d['delta'])==set(NATIVE_COUNTERS) and all(d['after'][k]-d['before'][k]==d['delta'][k]>=0 for k in NATIVE_COUNTERS),'actual21 native deltas')
    if p:require(not any(d['delta'].values()),'two complete actual zero sweeps');zeros+=21
 require(zeros==4*2*2*21,'all336 new component zero subtractions')
 return dict(schema='V202.independent-multiowner-raw-audit.v1',passed_current_multiowner_math=True,representative_sizes=[2,32],Native_zero_scalar_subtractions=zeros,
  original_helper_eager=True,current_root_publication_and_nextdraft=True,all_physical_hidden_and_role_queryKV_in_actual_process=True,
  tensors_recomputed_by_HOST=False,full32_numerical_exhaustive=False,new_Native_keys_required=False,Native_flags_assigned=False)
