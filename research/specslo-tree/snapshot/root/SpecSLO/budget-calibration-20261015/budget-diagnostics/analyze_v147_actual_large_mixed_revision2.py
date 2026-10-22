"""Independent stdlib terminal audit of frozen V147; never execute a device.

Only run after ROOT supplies the terminal receipt SHA. Frozen audit replay is
labelled structural; retained token arrays permit a separate causal/history
reconstruction. Full hidden/KV/top2 tensors were not retained for HOST replay.
"""
import argparse,ast,copy,datetime,gzip,hashlib,importlib,importlib.abc,json,math,re,struct,sys,tarfile,traceback
from contextlib import contextmanager
from pathlib import Path
sys.dont_write_bytecode=True
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-large-mixed-prompt-local-v147'
ACTION='large-mixed-prompt-local-qualification';PHYSICAL=(2,5,6,7)
FIXTURE=B/'tree-s35-production-v101/V101_QUALIFICATION_INPUTS.json';FIXTURE_SHA='a99a2120fb87ca0bb7a532eab7011d1616ce84a204a87bcc089b3229a86af6d7'
EXPECTED={str(P/'CPU_RESULT_V147.json'):'138e0b83a4c0659d2c63fa025bcc6807b0268582e39b728580281bad95d4aa85',
 str(P/'PREPARED_SOURCE_HASHES_V147.json'):'72553da2e25aa240a1c566e7d15b171c635be749b5bfe26b05a0bfcaa8564b4a',
 str(P/'PREPARED_SOURCE_ARCHIVE_V147.tar.gz'):'30aed28faa3cfbe788800aaca956c200abf579991fbb271384c6c0094fd4711a',
 str(D/'ROOT_V147_PREPARED_SOURCE_HOST_AUDIT.json'):'94ef62a9d43407206a5749f15626a538f0cb71f02900616f77cc2c8069f89416'}
REUSED=D/'analyze_v144_actual_variable_B4.py';REUSED_SHA='64520cd9a9e243c086dcb9faefcf8f18ffae1f59e0ed64af42572773cced35e1'
COUNTERS=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',*(f'aclgraph_{r}_{f}' for r in ('generic','draft','target') for f in ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
FAILURES=tuple(k for k in COUNTERS if any(v in k for v in ('failed_captures','fallback','disabled_entry','unclassified','validation_failures')))
ORIGINAL='fd53817d55fa73164fef345eba8615b4cd1ac790cdd509a82fa7d590dde6de8f';CANONICAL='ed44e73fd5ad72088aa36b41a8c94e6a1d4dccb15da909f26a060bab5a0997ae'
SCOPES=(('V147-original197-continuous',197,128,0,(65,130)),('V147-original197-final-Q1',197,128,2300,(2301,)),('V147-large645-continuous',645,512,0,(512,1024)),('V147-large645-final-Q1',645,512,2300,(2301,)))

def need(value,message):
 if value is not True:raise ValueError('V147 external audit: '+message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
 return h.hexdigest()
def document(path):return json.loads(Path(path).read_bytes())
def artifact(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def gunzip(path):return json.loads(gzip.decompress(Path(path).read_bytes()))

class NoDeviceImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers'):
   raise ImportError('V147 external HOST audit blocks backend/model import: '+fullname)

def reused_helpers():
 """Compile exact pure HOST functions; never mutate the original module globals."""
 need(sha(REUSED)==REUSED_SHA,'explicit original V144 helper source SHA')
 wanted={'check_archive','idle_record','verifier_record','causal_verdict'}
 tree=ast.parse(REUSED.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
 need({n.name for n in nodes}==wanted,'four exact reusable HOST function definitions')
 namespace=dict(need=need,sha=sha,document=document,artifact=artifact,Path=Path,hashlib=hashlib,tarfile=tarfile,re=re,json=json,P=P,B=B,PHYSICAL=PHYSICAL)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(REUSED)+'::V147-pure-HOST-functions','exec'),namespace)
 return namespace,dict(source=artifact(REUSED),functions=sorted(wanted),original_source_or_globals_modified=False)

def source_identity(helpers):
 for path,h in EXPECTED.items():need(sha(path)==h,'frozen preparation pin '+path)
 frozen=document(P/'PREPARED_SOURCE_HASHES_V147.json');cpu=document(P/'CPU_RESULT_V147.json')
 need((len(frozen['files']),len(frozen['external_files']),len(frozen['retained_attempt_files']))==(47,83,18),'exact47/83/18 frozen source closure')
 need(frozen['root_review_approved'] is True and frozen['actual_device_qualification'] is frozen['adopted'] is False,'prepared source is not device qualification')
 need(cpu['status']=='passed' and cpu['tests_run']==13 and cpu['failures']==cpu['errors']==0
  and cpu['Torch_imported'] is cpu['NPU_initialized'] is cpu['model_initialized'] is cpu['device_query'] is False,'actual13 HOST proof')
 need(cpu['input_hashes']==cpu['input_hashes_after']==frozen['files'] and cpu['external_input_hashes']==cpu['external_input_hashes_after']==frozen['external_files']
  and frozen['CPU_sha256']==sha(P/'CPU_RESULT_V147.json'),'exact current pre/post HOST inputs')
 sources={**frozen['files'],**frozen['external_files'],**frozen['retained_attempt_files'],str(P/'CPU_RESULT_V147.json'):sha(P/'CPU_RESULT_V147.json'),str(P/'PREPARED_SOURCE_HASHES_V147.json'):sha(P/'PREPARED_SOURCE_HASHES_V147.json')}
 need(len(sources)==150 and all(sha(p)==h for p,h in sources.items()),'all150 exact current frozen inputs')
 need(not [p for p in P.iterdir() if p.suffix in ('.py','.sh') and str(p) not in frozen['files']],'no unbound executable source')
 review=document(D/'ROOT_V147_PREPARED_SOURCE_HOST_AUDIT.json')
 need(review['status']=='passed' and review['archive_members']==150 and review['all_input_and_archive_bytes_verified'] is True
  and review['model_or_device_queried'] is review['V147_actual_launch_performed'] is False,'exact ROOT prelaunch source audit scope')
 prepared=helpers['check_archive'](P/'PREPARED_SOURCE_ARCHIVE_V147.tar.gz',sources)
 need(sha(FIXTURE)==FIXTURE_SHA,'independent direct provider bytes declared by frozen scope_v147')
 return sources,dict(prepared_archive=prepared,ROOT_prelaunch_review=artifact(D/'ROOT_V147_PREPARED_SOURCE_HOST_AUDIT.json'),current_input_count=150,
  supplemental_actual_provider=artifact(FIXTURE),provider_not_in_150_archive=True,provider_verified_against_frozen_scope_declared_SHA=True)

@contextmanager
def original_auditor():
 names=('audit_v147','scope_v147','local_qualification_scope','mc2_inventory_contract');old_path=list(sys.path);saved={n:sys.modules.pop(n,None) for n in names}
 try:
  sys.path.insert(0,str(P));module=importlib.import_module('audit_v147')
  need(Path(module.__file__).resolve()==P/'audit_v147.py','actual frozen structural audit origin')
  yield module
 finally:
  sys.path[:]=old_path
  for n in names:
   sys.modules.pop(n,None)
   if saved[n] is not None:sys.modules[n]=saved[n]

def fixture():
 need(sha(FIXTURE)==FIXTURE_SHA,'same exact genuine provider before reconstruction')
 data=document(FIXTURE)['S35'];result=[]
 for i,row in enumerate(data['trees']):
  prefix=min(128+i%4,len(row['token_ids'])-6)
  result.append(dict(copy.deepcopy(row),token_ids=row['token_ids'][:prefix+1],original_full_source_token_sha256=row['prompt_token_sha256'],initial_root_teacher_forced=True,initial_prefix=prefix,output_offset=0))
 need(len(result)==32 and len(data['prompt']['token_ids'])==2301,'independent real provider32 and2301prompt')
 return result,data['prompt']['token_ids']
def check_source(source,initial,contexts):
 need(source['ids']==list(range(32)) and len(source['contexts'])==32,'actual all32 normal source owners')
 for row,base,values in zip(source['contexts'],initial,contexts,strict=True):
  expected=dict(base,token_ids=values,prefix=len(values)-1,slo_class=base.get('slo_class','tight'),output_offset=len(values)-len(base['token_ids']))
  need(row==expected,'source follows independently reconstructed actual history and output offset')
def geometry(m,max_query,start,end):
 p=end-start;pad,extra=divmod(m-128-p,2);scratch=[pad+1]*extra+[pad]*(2-extra)
 return dict(M=m,S=35,Q=128,C=96,N=32,budgets=[3]*32,start=start,end=end,P=p,final=end==2301,
  mask=[35,1,max_query,4096],scratch=scratch,partition=[4]*32+[p]+scratch)
def target_key(m,max_query):
 return ('target-hidden:0|steps:1|tree-fia-full:exact-length-task-update|buffers:'+repr((((35,1,max_query,4096),(35,32)),)),(m,))
def counter_delta(raw):
 need(set(raw['before'])==set(raw['after'])==set(raw['delta'])==set(COUNTERS),'exact21 Native counters')
 values={k:raw['after'][k]-raw['before'][k] for k in COUNTERS}
 need(raw['delta']==values and all(type(v)is int and v>=0 for v in values.values()) and not any(values[k] for k in FAILURES),'independent Native subtraction/no fallback')
 return values

def tensor_summary_structure(value,hidden=False):
 shape=value['shape'];need(type(shape)is list and shape and all(type(n)is int and n>0 for n in shape)
  and shape[0]==128 and (not hidden or shape==[128,5120]),'real normal hidden/KV summary shape')
 count=value['differing_elements'];samples=value['counterexample'];maximum=value['max_abs']
 need(type(value['original_matcher'])is bool and type(value['bit_exact'])is bool
  and type(count)is int and 0<=count<=math.prod(shape) and math.isfinite(maximum) and maximum>=0
  and re.fullmatch('[a-f0-9]{64}',value['original197_sha256']) is not None
  and re.fullmatch('[a-f0-9]{64}',value['large645_sha256']) is not None,'finite bounded summary metadata')
 need(value['bit_exact']==(count==0) and len(samples)==min(8,count),'summary exactness/count/counterexample arithmetic')
 if count==0:need(maximum==0 and value['original197_sha256']==value['large645_sha256'],'equal summary hashes and zero difference')
 else:
  need(maximum>0 and value['original197_sha256']!=value['large645_sha256'],'different summary nonzero range and hashes')
  indices=[v['flat_index'] for v in samples]
  need(indices==sorted(set(indices)) and all(type(i)is int and 0<=i<math.prod(shape) for i in indices)
   and all(math.isfinite(v['original197']) and math.isfinite(v['large645']) and v['original197']!=v['large645'] for v in samples),'retained tensor counterexamples structure only')

def cross_head_observation(old,new,record):
 """Recompute only retained exact head tokens, not unretained hidden/KV arrays."""
 a,b=old['target']['normal_query_tokens'],new['target']['normal_query_tokens'];need(len(a)==len(b)==128,'real normal128 head arrays')
 need(all(type(v)is int and 0<=v<151936 for v in a+b),'actual vocabulary head tokens')
 summary=record['selected_heads'];different=[i for i,(x,y) in enumerate(zip(a,b,strict=True)) if x!=y]
 need(summary['shape']==[128] and summary['dtype']=='torch.int64' and sys.byteorder=='little','actual retained long head ABI and byte order')
 hashes=[hashlib.sha256(struct.pack('<128q',*v)).hexdigest() for v in (a,b)]
 need(summary['original197_sha256']==hashes[0] and summary['large645_sha256']==hashes[1]
  and summary['bit_exact']==(not different) and summary['differing_elements']==len(different)
  and summary['max_abs']==max(abs(x-y) for x,y in zip(a,b,strict=True)),'raw token arrays independently reproduce cross-M head hashes/diffs')
 expected=[dict(flat_index=i,original197=float(a[i]),large645=float(b[i])) for i in different[:8]]
 need(summary['counterexample']==expected and record['exact_normal_heads']==(not different),'raw token arrays independently reproduce retained head counterexamples')
 tensors=[record['hidden'],*record['query_KV']]
 need(len(tensors)==129,'normal hidden plus128 full-layer KV summaries')
 for index,value in enumerate(tensors):tensor_summary_structure(value,hidden=index==0)
 hidden_match=all(v['original_matcher'] for v in tensors)
 need(record['all_normal_hidden_KV_original_matcher']==hidden_match and record['cross_M_equivalent']==((not different) and hidden_match),'cross-M summary conclusion logic only')
 same_input=old['source']==new['source'] and old['rows']==new['rows'] and old['selected']==new['selected']
 return dict(same_real_normal_query_inputs=same_input,exact_heads_independently_recomputed=not different,
  differing_head_tokens=len(different),counterexamples=expected,selected_head_hashes_recomputed=hashes,
  recorded_hidden_KV_matcher=hidden_match,hidden_KV_numerics_HOST_recomputed=False,
  recorded_cross_M_equivalent=record['cross_M_equivalent'],full_AR_losslessness_proven=False)

def first_failure(directory,receipt):
 """Keep missing artifacts as explicit incomplete coverage, not auditor crash."""
 evidence=[];events=[]
 for rank in range(4):
  path=directory/f'rank{rank}-last-error.json'
  if path.is_file():
   try:row=document(path)
   except (ValueError,OSError):
    evidence.append(dict(rank=rank,artifact=artifact(path),checkpoint_read_error=traceback.format_exc()));continue
   if type(row)is not dict:
    evidence.append(dict(rank=rank,artifact=artifact(path),checkpoint_schema_error='last-error checkpoint is not an object'));continue
   bad=[]
   for event in row.get('boundary_events',[]) if type(row.get('boundary_events',[]))is list else []:
    if type(event)is dict and any(type(v)is dict and v.get('error') is not None for v in event.get('ranks',[])):bad.append(event)
   if bad:events.append(dict(rank=rank,first_common_failed_event=bad[0]))
   original_inner=re.search(r'(?m)^(?:[A-Za-z_]\w*\.)*BoundaryRejected: All-rank original boundary rejected ([^:\n]+):',row.get('error',''))
   evidence.append(dict(rank=rank,artifact=artifact(path),error=row.get('error'),
    first_original_prompt_boundary_failure=None if original_inner is None else original_inner.group(1),
    original_boundary_stage_source='frozen NativeBoundaries.stage exception text, not a fabricated outer event',
    last_completed_stage=(row.get('boundary_events') or [{}])[-1].get('label')))
 missing=[str(directory/f'rank{r}-component.json.gz') for r in range(4) if not (directory/f'rank{r}-component.json.gz').is_file()]
 return dict(terminal_status=receipt.get('status'),supervisor_errors=receipt.get('errors'),first_common_failed_events=events,worker_error_evidence=evidence,missing_final_reports=missing,
  failure_stage_unknown_without_reachable_checkpoint=not events and not any(v.get('first_original_prompt_boundary_failure') for v in evidence),
  component_qualification=False,MC2_645_qualified=False,adopted=False,formal_Goodput=False)

def failed_terminal_evidence(receipt,tag,sources,helpers):
 """Verify available terminal evidence without turning absence into success."""
 result={}
 for key,callback in (('devices_before',lambda:helpers['idle_record'](receipt['devices_before'],'before')),
                      ('devices_after',lambda:helpers['idle_record'](receipt['devices_after'],'after')),
                      ('pre_verifier',lambda:helpers['verifier_record'](receipt['pre_verifier'],tag,'pre')),
                      ('post_verifier',lambda:helpers['verifier_record'](receipt['post_verifier'],tag,'post'))):
  if key not in receipt:result[key]=dict(verified=False,missing=True);continue
  try:callback();result[key]=dict(verified=True)
  except (ValueError,OSError,KeyError,TypeError):result[key]=dict(verified=False,error=traceback.format_exc())
 path=P/'source-snapshots'/f'{tag}.tar.gz'
 if path.is_file() and 'archive' in receipt:
  try:
   need(receipt['archive']==dict(path=str(path),sha256=sha(path),members=150),'failed terminal actual source archive descriptor')
   result['actual_source_archive']=dict(verified=True,details=helpers['check_archive'](path,sources))
  except (ValueError,OSError,KeyError,TypeError):result['actual_source_archive']=dict(verified=False,error=traceback.format_exc())
 else:result['actual_source_archive']=dict(verified=False,missing=True)
 return result

def audit_actual(tag,receipt_sha):
 need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',tag) is not None and re.fullmatch('[a-f0-9]{64}',receipt_sha) is not None,'ROOT unique tag and terminal receipt SHA')
 helpers,reuse=reused_helpers();sources,identity=source_identity(helpers);receipt_path=P/f'receipt-{tag}.json'
 need(sha(receipt_path)==receipt_sha,'exact ROOT-provided receipt bytes');receipt=document(receipt_path);directory=P/'actual'/tag
 need(receipt['schema']=='V147.actual-component-supervisor.v1' and receipt['tag']==tag and receipt['action']==ACTION
  and receipt['status'] in ('complete','failed'),'actual original terminal V147 supervisor schema')
 if receipt['status']=='failed':return dict(schema='ROOT.V147.actual-independent-component-audit.v1',status='failed/incomplete',receipt=artifact(receipt_path),source_identity=identity,reused_helpers=reuse,available_terminal_evidence=failed_terminal_evidence(receipt,tag,sources,helpers),**first_failure(directory,receipt))
 need(receipt['errors']==[] and receipt['owned_remaining']==[] and receipt['actual_execution_complete'] is receipt['sources_unchanged'] is True and receipt['child']['exit_code']==0,'complete terminal supervisor/cleanup/source result')
 need(receipt['formal_Goodput'] is receipt['cost_measured'] is False and receipt['physical_cards']==list(PHYSICAL),'component never performance claim')
 need(receipt['execution_lock']==dict(path=str(B/'.ROOT_NPU_EXECUTION.lock'),nonblocking_exclusive=True,scope='this supervisor lifetime'),'recorded original exclusive lock contract')
 need(receipt['child']['argv']==['bash',str(P/'launch_v147.sh'),'--action',ACTION,'--tag',tag] and receipt['child']['log']==str(P/f'local-{tag}.log'),'actual launch argv/path')
 need(receipt['limits']==dict(seconds=3600,private_bytes=256*1024**2) and 0<receipt['elapsed_seconds']<3600,'actual bounded local scope')
 helpers['idle_record'](receipt['devices_before'],'before');helpers['idle_record'](receipt['devices_after'],'after')
 pre=helpers['verifier_record'](receipt['pre_verifier'],tag,'pre');post=helpers['verifier_record'](receipt['post_verifier'],tag,'post')
 archive=P/'source-snapshots'/f'{tag}.tar.gz';need(receipt['archive']==dict(path=str(archive),sha256=sha(archive),members=150),'actual150 source archive descriptor')
 actual_archive=helpers['check_archive'](archive,sources)
 expected={directory/n for n in ('prerequisites.json','config.json','completed.json','worker-exit.json')}
 need(document(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all four workers actually exited')
 need(document(directory/'completed.json')==dict(action=ACTION,replies=[['calibrated',r,dict(arms=2,prompt_phases=18,timings=0)] for r in range(4)],cost_measured=False,formal_Goodput=False),'actual original completion schema')
 need(document(directory/'prerequisites.json')==receipt['readiness'] and receipt['readiness']['CPU_sha256']==EXPECTED[str(P/'CPU_RESULT_V147.json')]
  and receipt['readiness']['freeze_sha256']==EXPECTED[str(P/'PREPARED_SOURCE_HASHES_V147.json')],'worker and supervisor same frozen readiness')
 config=document(directory/'config.json')
 need((config['max_num_seqs'],config['max_num_queued_seqs'],config['num_kvcache_blocks'],config['gamma'])==(64,66,544,8)
  and config['draft_model_path']=='/root/data/shared-models/Qwen3-0.6B' and config['target_model_path']=='/root/data/shared-models/Qwen3-32B'
  and config['enable_mc2'] is True and config['precompile_decode_graphs'] is config['precompile_serial_draft_graphs'] is False,'actual original outer configuration snapshot')
 with original_auditor() as original:strict=original.audit_reports(directory)
 need(json.loads(json.dumps(strict,allow_nan=False))==receipt['actual_component_audit'],'frozen structural audit replay equals actual supervisor result; not independent numerical replay')
 reports=[gunzip(directory/f'rank{r}-component.json.gz') for r in range(4)];initial,prompt=fixture();summaries=[];cross=[];common=None
 for rank,report in enumerate(reports):
  expected.add(directory/f'rank{rank}-component.json.gz');expected.add(directory/'startup-progress'/f'MC2-preflight-rank{rank}.json')
  need(report['rank']==rank and report['role']==('draft' if rank==0 else 'target'),'actual worker role')
  need(report['original_MC2_static']==document(directory/'startup-progress'/f'MC2-preflight-rank{rank}.json'),'raw MC2 startup report same bytes decode')
  need(report['construction']['layers']==(28 if rank==0 else 64) and report['construction']['tree_scratch']==list(range(66,130)),'actual local constructor layers/private scratch')
  if common is None:common=report['boundary_events']
  else:need(report['boundary_events']==common,'all4 exact raw fixed-stage packets equal')
  need(sum(e['label'].endswith('/complete-trace-Native-delta') for e in common)==12,'all12 Native final delta gates are common stages')
  need(report['profile_scope']['profile_M645_membership_only'] is False and report['profile_scope']['MC2_645_qualified'] is False,'original unprofiled645 scope retained')
  total={k:0 for k in COUNTERS};first_before=None;previous_after=None;stable_by_scope={};repair=set();seen_target=set()
  for index,trace in enumerate(report['traces']):
   pass_index,scope_index=divmod(index,4);name,m,max_query,start,ends=SCOPES[scope_index]
   need((trace['pass_index'],trace['scope'])==(pass_index,name),'exact original ordered scope identity')
   values=counter_delta(trace['Native']);total={k:total[k]+values[k] for k in COUNTERS}
   if first_before is None:first_before=trace['Native']['before'];need(not any(first_before.values()),'true original cold first trace')
   if previous_after is not None:need(trace['Native']['before']==previous_after,'all Native changes belong to exact contiguous local traces')
   previous_after=trace['Native']['after']
   if pass_index:need(not any(values.values()),'independently checked two complete zero Native traces')
   contexts=[list(v['token_ids']) for v in initial];phase_stable=[];delivered=0;guard_baseline=None
   lease=trace['lease'];owners=list(range(33))+[66,67];refs=list(range(33,66))+[68,69]
   need(lease['actual_candidate_ids']==owners and lease['actual_reference_ids']==refs
    and lease['source_KV_copied'] is False and lease['independent_actual_prefixes'] is True,'actual independent candidate/reference owner partition')
   hashes=lease['actual_prefix_hashes'];need(set(hashes)==set(map(str,owners+refs)) and all(len(v)==(56 if rank==0 else 128)
    and all(re.fullmatch('[a-f0-9]{64}',h) is not None for h in v) for v in hashes.values()),'all56/128 actual prefix hash structure; no raw KV enumeration')
   for cycle,end in enumerate(ends):
    path=directory/f'rank{rank}-pass{pass_index}-{name}-cycle{cycle}.json.gz';expected.add(path);record=gunzip(path)
    need(record==trace['records'][cycle] and record['geometry']==geometry(m,max_query,start,end),'exact separately retained full phase and independent M/S/P/Q arithmetic')
    check_source(record['source'],initial,contexts)
    need(record['selected']==[[0,1,2]]*32 and all(len(v)==3 for v in record['rows'])
     and all(type(v)is int and 0<=v<151936 for row in record['rows'] for v in row) and len(record['confidences'])==32
     and all(type(v)in(int,float) and math.isfinite(v) and 0<=v<=1 for v in record['confidences']),'controlled primary3 actual typed rows/confidence domain')
    leader=reports[1]['traces'][index]['records'][cycle];query=leader['target']['normal_query_tokens']
    need(len(query)==128 and all(type(v)is int and 0<=v<151936 for v in query),'retained full real Q128 target token domain')
    emitted,accepted=helpers['causal_verdict'](record['rows'],query,[3]*32)
    commit=record['compiled_normal_commit']
    need(commit['committed_tokens']==emitted and commit['accepted_paths']==accepted,'independent exact causal B3 target-verdict reconstruction')
    contexts=[old+tokens for old,tokens in zip(contexts,emitted,strict=True)]
    need(commit['actual_logical_tokens']==contexts and commit['actual_prefix_epochs']=={str(i):cycle+1 for i in range(32)}
     and commit['cycle_id']==cycle and commit['target_home']==cycle%2 and commit['original_decode_sha256']==ORIGINAL
     and commit['canonical_decode_sha256']==CANONICAL,'actual source recurrence/protocol/epoch/home rotation')
    need(record['prompt_commit']['cursor']==end and record['prompt_commit']['last_committed_cycle_id']==cycle
     and record['clock']['original_clock_block_executed'] is True and record['clock']['incumbent_only_count']==32
     and math.isfinite(record['clock']['last_cycle_ms']) and record['clock']['last_cycle_ms']>=0,'real contiguous prompt cursor and original incumbent-only diagnostic clock')
    dispatch=record['MC2'];need(dispatch['delta']=={k:dispatch['after'][k]-dispatch['before'][k] for k in dispatch['before']}
     and all(v>=0 for v in dispatch['delta'].values()) and dispatch['delta']['fallback']==dispatch['delta']['exception']==0,'independent raw MC2 counter subtraction')
    need(dispatch['MC2_645_qualified'] is False and (rank==0 or m!=197 or dispatch['delta']['native_epilogue_success']>0),'197 positive route distinct from unqualified645')
    guard=record['own_reference_commit']['lease'];need(guard['storage_rows']==130 and guard['unique_pages']==176
     and guard['candidate_ids']==owners and guard['reference_ids']==refs and guard['candidate_KV_used_as_reference'] is False
     and re.fullmatch('[a-f0-9]{64}',guard['table_sha256']) is not None
     and guard['allocation_epoch']==index+1 and guard['prompt_cursor']==start,'immutable actual176-page lease/epoch/cursor summary, not raw page reconstruction')
    immutable_guard={k:guard[k] for k in ('table_sha256','allocation_epoch','storage_rows','unique_pages','candidate_ids','reference_ids')}
    if guard_baseline is None:guard_baseline=immutable_guard
    else:need(immutable_guard==guard_baseline,'actual complete local trace keeps the same bound independent allocation table and epoch')
    if rank:
     key=target_key(m,max_query);warm=key in seen_target
     need(record['target']['normal_query_tokens']==query and record['target']['final_prompt_tokens']==leader['target']['final_prompt_tokens'],
      'all3 target real normal and final-prompt head arrays')
     for ordinal,phase in enumerate(record['target']['Native_phases']):
      expected_before=True if warm or ordinal==2 else (None if ordinal==0 else False)
      expected_after=warm or ordinal>=1
      need(phase['phase']==('first','changed','restored')[ordinal] and phase['key']==repr(key)
       and phase['entry_before'] is expected_before and phase['entry_after'] is expected_after
       and phase['tasks']==64 and phase['M']==m and phase['S']==35,'actual original cold None/False/True F/T/F lineage and exact FULL key')
      execution=phase['original_execution']
      need(set(execution)=={'mode','fallback_reason','capture_attempted','replay_executed'}
       and execution['mode'] in ('capture_replay','replay') and execution['fallback_reason'] is None
       and execution['replay_executed'] is True,'actual serialized original execution fields; no invented used_aclgraph field')
     seen_target.add(key)
    else:repair.add(record['draft']['repair_physical'])
    publication=record['publication'];final_token=None
    if end==2301:
     final_token=leader['target']['final_prompt_tokens'][0]
     need(leader['target']['final_prompt_tokens']==[final_token] and publication['first_token']==final_token
      and publication['source_prompt_sha256']==digest(prompt) and publication['logical_terminal_requests']==[32],
      'real final target first token/true2301prompt/original one-time publication')
     callbacks=publication['actual_callback_events'];need(len(callbacks)==(1 if rank==1 else 0),'only original target leader publishes once')
     if rank==1:need(callbacks[0]['request_index']==32 and callbacks[0]['token_ids']==[final_token]
      and callbacks[0]['request_id']=='V147-real2301-prompt' and callbacks[0]['finished'] is False,'actual original callback first token')
    else:need(publication is None and leader['target']['final_prompt_tokens']==[],'no intermediate prompt completion token')
    phase_stable.append(dict(source=record['source'],rows=record['rows'],selected=record['selected'],emitted=emitted,accepted=accepted,final_token=final_token))
    delivered+=sum(map(len,emitted));start=end
   nxt=trace['next_draft'];check_source(nxt['source'],initial,contexts)
   if rank==0:repair.add(nxt['draft']['repair_physical'])
   need(nxt['selected']==[[0,1,2]]*32 and len(nxt['rows'])==32 and all(len(v)==3 for v in nxt['rows']),'actual full next-draft probe from observed committed history')
   stable=dict(phases=phase_stable,next_draft={k:nxt[k] for k in ('source','rows','selected')})
   need(trace['fingerprint']==digest(stable),'independently rebuilt actual phase+nextdraft stable fingerprint')
   if pass_index:need(stable==stable_by_scope[name],'two complete fresh-history zero traces reproduce all actual outputs')
   else:stable_by_scope[name]=stable
   tracepath=directory/f'rank{rank}-pass{pass_index}-{name}-trace.json.gz';expected.add(tracepath);need(gunzip(tracepath)==trace,'full separate trace retention')
   summaries.append(dict(rank=rank,pass_index=pass_index,scope=name,M=m,S=35,Q=128,commits=len(ends),nextdraft=1,delivered_normal_tokens=delivered))
  need(total==report['final_Native'] and report['inventory_before_seal']==report['final_inventory'],'complete raw Native sum and unchanged sealed inventory')
  inventory=report['final_inventory'];need(inventory['entries']==[],'actual generic entry dictionary empty; not logical family counts')
  targets={ast.literal_eval(v) for v in inventory['target_entries']};drafts={ast.literal_eval(v) for v in inventory['draft_entries']}
  if rank:need(targets=={target_key(197,128),target_key(645,512)} and drafts==set(),'exact two target FULL hidden entries')
  else:
   expected_repairs={('target-hidden:0|steps:1|paged',(v,)) for v in repair if v>0}
   need(drafts=={('draft-greedy:151936|steps:3|paged|confidence',32)} and targets==expected_repairs,'actual shared3 draft and exact observed phase/nextdraft repair PA keys')
  need(report['final_Native']['aclgraph_captures']==len(targets)+len(drafts)==report['final_Native']['aclgraph_capture_attempts'],'actual exact entries and cold captures')
  for pass_index in range(3):
   crosspath=directory/f'rank{rank}-pass{pass_index}-cross-M.json.gz';expected.add(crosspath);observed=gunzip(crosspath)
   need(report['cross_M_real_normal'][pass_index]==dict(pass_index=pass_index,comparison=observed),'raw cross-M receipt matches summary')
   if rank:cross.append(dict(rank=rank,pass_index=pass_index,**cross_head_observation(report['traces'][pass_index*4]['records'][0],report['traces'][pass_index*4+2]['records'][0],observed)))
 need({f for f in directory.rglob('*') if f.is_file()}==expected and len(expected)==144,'exact144 actual controls/MC2/components/phase/trace/cross artifacts; nextdraft embedded not separate')
 need('torch' not in sys.modules and 'torch_npu' not in sys.modules,'external audit stayed HOST stdlib')
 return dict(schema='ROOT.V147.actual-independent-component-audit.v1',status='passed',receipt=artifact(receipt_path),source_identity=identity,actual_source_archive=actual_archive,
  reused_helpers=reuse,pre_verifier=pre,post_verifier=post,frozen_structural_audit_replayed=strict,structural_replay_is_independent_numerical_recompute=False,
  exact_raw_file_count=144,actual_prompt_and_normal_commits_per_rank=18,actual_nextdraft_probes_per_rank=12,complete_local_zero_sweeps=2,
  independent_actual_trace_summary=summaries,independent_cross_M_exact_head_observations=cross,
  actual_scope=dict(maxseq=64,maxqueued=66,storage130=True,pages544=True,lease_pages176=True,normalN32=True,normalB3=True,oldM197_full128=True,newM645_full512=True),
  Native_same_M_numerical_execution_receipts_structurally_verified=True,hidden_KV_or_full_top2_HOST_recomputed=False,
  initial_normal_root_teacher_forced_not_own_AR=True,subsequent_verdict_and_source_history_HOST_recomputed=True,
  MC2_645_qualified=False,adopted=False,full527_or_full463_qualified=False,full_AR_losslessness_proven=False,formal_Goodput=False,cost_measured=False,
  evidence_limits=['Frozen audit replay verifies retained fields, not a new NPU computation.','Full same-M hidden/KV and draft full encoded top2 tensors were not retained; their numeric match cannot be recomputed from marker/hash alone.',
   'Cross-M exact heads/diffs/hashes are independently reconstructed from128 raw target tokens; hidden/KV summaries only receive structural/logic checks.',
   'Physical block tables/actual page enumerations were not retained; lease IDs/counts/hash consistency is distinct from independent page reconstruction.',
   'Initial2300prompt KV/prefix hashes attest the actual bounded original initialization/adoption path; HOST does not re-execute those model forwards.',
   'M645 capacity permission and same-M matcher success do not qualify MC2_645, full527/full463 production or full-AR equality.'],
  Torch_imported=False,NPU_queried=False,model_called=False)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
 output=Path(args.output);need(output.parent.resolve()==D and not output.exists(),'fresh ROOT diagnostic output only')
 firewall=NoDeviceImports();sys.meta_path.insert(0,firewall)
 try:
  try:result=audit_actual(args.tag,args.expected_receipt_sha256)
  except BaseException:
   result=dict(schema='ROOT.V147.actual-independent-component-audit.v1',status='failed/incomplete',error=traceback.format_exc(),MC2_645_qualified=False,adopted=False,formal_Goodput=False)
   if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None:
    path=P/f'receipt-{args.tag}.json'
    try:
     if path.is_file() and sha(path)==args.expected_receipt_sha256:
      receipt=document(path);result['actual_failure_location']=first_failure(P/'actual'/args.tag,receipt)
    except BaseException:result['failure_location_read_error']=traceback.format_exc()
  result.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_HOST_argv=sys.argv,Torch_imported='torch' in sys.modules,NPU_queried=False)
  with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
  print(json.dumps(dict(status=result['status'],output=str(output),sha256=sha(output))))
  if result['status']!='passed':raise SystemExit(1)
 finally:sys.meta_path.remove(firewall)
if __name__=='__main__':main()
