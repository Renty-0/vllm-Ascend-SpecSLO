"""ROOT-only terminal V144 audit. Stdlib; no device query or model import.

Preparation creates no actual descriptor. Invoke only after ROOT supplies the
terminal tag and the SHA256 of its receipt. Frozen numerical gates are rerun
against retained JSON; hidden/KV tensors were not persisted for HOST replay.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime,timezone
import gzip
import hashlib
from functools import lru_cache
import importlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tarfile
import time
import traceback

sys.dont_write_bytecode=True
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-variable-B4-local-qualification-v144'
ACTION='variable-B4-local-qualification'
PHYSICAL=(2,5,6,7)
EXPECTED={
 str(P/'CPU_RESULT_V144.json'):'8308e43bca8302f626be587b75fd00ff2e0b7845ffed2cc1bf896c18ce865587',
 str(P/'PREPARED_SOURCE_HASHES_V144.json'):'4f3f18f2ab7c10f661a250fa21860a8c9164f30cb92dbf40582a2b9bb644677b',
 str(P/'PREPARED_SOURCE_ARCHIVE_V144.tar.gz'):'09eb935376586c34d0389e3f9d9e7c1d17ec210c80b75ffa7ee71e3a1afabd7a',
 str(D/'ROOT_V144_PREPARED_SOURCE_HOST_AUDIT.json'):'b1a40d79e214f613f28acaf1a17c16b9b3c9974793ee85b5d6a5cca993a3fdab',
 str(D/'V144_READONLY_PEER_REVIEW_PERFORMANCE_COST_AUDIT.json'):'d724d2a1bb69becbddb484d513e48681299e2018c2740db81e3e17be26509e93',
}
VECTORS=((4,)*19+(3,)+(2,)*5+(1,)*7,(4,)*19+(2,)*6+(1,)*7)
GEOMETRIES=((128,32,()),(165,38,(7,7,6,6,6,6)))
GENERATE_STAGES=('source','variable-work-plans','typed-controlled-grants','typed-actual-work','actual-draft',
 'independent-draft','independent-draft-original-matcher-and-exact-ABI','original-two-role-refine','original-five-item-exchange-split')
PROTO_STAGES=('V75-original-tree-cycle-preflight','V75-original-tree-host-verdict',
 'V75-original-tree-consensus-compact-finish','V75-original-tree-commit-epoch-confirmation')
ORIGINAL_DECODE='fd53817d55fa73164fef345eba8615b4cd1ac790cdd509a82fa7d590dde6de8f'
CANONICAL_DECODE='ed44e73fd5ad72088aa36b41a8c94e6a1d4dccb15da909f26a060bab5a0997ae'

def need(value,message):
 if not value:raise ValueError('V144 external audit: '+message)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
 return h.hexdigest()

def document(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def artifact(path):return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=sha(path))
@lru_cache(maxsize=256)
def gunzip(path):return json.loads(gzip.decompress(Path(path).read_bytes()))
def f32(value):return struct.unpack('!f',struct.pack('!f',value))[0]

class NoDeviceImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers'):
   raise ImportError('V144 external HOST audit blocks backend/model import: '+fullname)

def check_archive(path,expected):
 wanted={str(Path(p)).lstrip('/'):h for p,h in expected.items()}
 need(len(wanted)==len(expected),'archive path alias')
 with tarfile.open(path,'r:gz') as archive:
  members=archive.getmembers()
  need(len(members)==len(wanted) and len({m.name for m in members})==len(wanted),'archive duplicate/missing members')
  need({m.name for m in members}==set(wanted),'archive exact source coverage')
  for member in members:
   need(member.isfile() and not member.issym() and not member.islnk(),'archive must contain only exact regular inputs')
   stream=archive.extractfile(member);h=hashlib.sha256()
   for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
   need(h.hexdigest()==wanted[member.name],'archive member bytes '+member.name)
 return dict(members=len(wanted),every_member_bytes_checked=True,sha256=sha(path))

def source_identity():
 for path,h in EXPECTED.items():need(sha(path)==h,'frozen preparation pin '+path)
 frozen=document(P/'PREPARED_SOURCE_HASHES_V144.json');cpu=document(P/'CPU_RESULT_V144.json')
 need((len(frozen['files']),len(frozen['external_files']),len(frozen['retained_attempt_files']))==(38,70,7),'38/70/7 prepared closure')
 need(frozen['root_review_approved'] is True and frozen['actual_device_qualification'] is frozen['adopted'] is False,'prepared scope is source review, not device inheritance')
 need(cpu['status']=='passed' and cpu['tests_run']==13 and cpu['failures']==cpu['errors']==0 and cpu['Torch_imported'] is cpu['NPU_initialized'] is cpu['model_initialized'] is cpu['device_query'] is False,'actual13 pure HOST proof')
 need(cpu['input_hashes']==cpu['input_hashes_after']==frozen['files'] and cpu['external_input_hashes']==cpu['external_input_hashes_after']==frozen['external_files']
  and frozen['CPU_sha256']==sha(P/'CPU_RESULT_V144.json'),'actual CPU/current source before/after identity')
 sources={**frozen['files'],**frozen['external_files'],**frozen['retained_attempt_files'],
  str(P/'CPU_RESULT_V144.json'):sha(P/'CPU_RESULT_V144.json'),str(P/'PREPARED_SOURCE_HASHES_V144.json'):sha(P/'PREPARED_SOURCE_HASHES_V144.json')}
 need(len(sources)==117 and all(sha(p)==h for p,h in sources.items()),'117 exact prepared/current inputs')
 need(not [p for p in P.iterdir() if p.suffix in ('.py','.sh') and str(p) not in frozen['files']],'unbound executable source')
 review=document(D/'ROOT_V144_PREPARED_SOURCE_HOST_AUDIT.json')
 need(review['status']=='passed' and review['actual_new_HOST_tests']==13 and review['archive_members_all_bytes_checked']==117
  and review['actual_NPU_started'] is review['actual_numerical_qualification'] is False,'actual ROOT prelaunch audit scope')
 extra=document(P/'host-attempts/extra-protocol-source-capsule-attempt1.json')
 need(extra['Torch_imported'] is True and extra['NPU_initialization_state']=='not_queried_not_attested'
  and extra['explicit_model_constructor_calls']==extra['explicit_device_function_calls']==extra['explicit_device_queries']==0,
  'extra Torch source check remains separate and does not attest NPU initialization')
 archive=check_archive(P/'PREPARED_SOURCE_ARCHIVE_V144.tar.gz',sources)
 return sources,dict(CPU_sha256=sha(P/'CPU_RESULT_V144.json'),freeze_sha256=sha(P/'PREPARED_SOURCE_HASHES_V144.json'),
  local_sources=38,external_sources=70,retained_attempt_files=7,prepared_archive=archive,
  extra_Torch_source_check=artifact(P/'host-attempts/extra-protocol-source-capsule-attempt1.json'),
  extra_Torch_check_NPU_initialization_not_attested=True)

def idle_record(record,label):
 need(record['all_idle'] is record['four_910B2'] is True,label+' four-card idle guard')
 info=record['hardware'];need(info['exit_code']==0 and info['timed_out'] is False and info['argv']==['npu-smi','info'],label+' retained hardware query')
 parsed=[(int(i),name) for i,name in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',info['output'],re.MULTILINE) if not name.isdecimal()]
 need(sorted(parsed)==[(i,'910B2') for i in PHYSICAL] and record['parsed_hardware']==[[i,n] for i,n in parsed],label+' actual raw hardware rows')
 need(set(record['processes'])==set(map(str,PHYSICAL)),label+' exact physical cards')
 for card in PHYSICAL:
  p=record['processes'][str(card)]
  need(p['argv']==['npu-smi','info','-t','proc-mem','-i',str(card),'-c','0'] and p['exit_code']==0
   and p['timed_out'] is False and p['idle'] is True and re.search(r'No process in device\.|No running processes found',p['output'],re.I),label+' actual per-card idle output')

def verifier_record(record,tag,label):
 expected=['bash',str(B/'serial-v1/runtime.sh'),'python','-B','/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py','/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json']
 path=P/f'{tag}-{label}-verifier.log';need(record['argv']==expected and record['exit_code']==0 and record['path']==str(path) and record['sha256']==sha(path),'actual '+label+' verifier provenance')
 rows=[]
 for line in path.read_text().splitlines():
  try:row=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(row,dict) and 'baseline_id' in row:rows.append(row)
 need(len(rows)==1 and rows[0]['status']=='pass' and rows[0]['errors']==[],'actual '+label+' verifier raw result')
 lock=document('/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json')
 row=rows[0];need(row['baseline_id']==lock['baseline_id'] and row['checked_frozen_files']==len(lock['frozen_files_sha256'])
  and row['checked_external_files']==len(lock['external_files_sha256']) and row['checked_packages']==len(lock['package_versions']),label+' original lock scope')
 return artifact(path)

def expected_stages():
 labels=['configuration','source-origins','minimal-construction','MC2-before']
 target=['target-source-prefix-before']
 for kind in ('greedy','hidden'):
  for phase in ('first','changed','restored'):
   target.extend(f'target-{kind}/{phase}/{part}' for part in ('prepare','Native','drain'))
 target+=['target-full-physical-summary','target-reference/prepare','target-reference/original-model',
  'target-reference/query-and-finite','target-reference/original-same-M-head','target-reference/drain-original-matcher']
 for p in range(3):
  for c in range(2):
   prefix=f'pass{p}/case{c}'
   labels.append(prefix+'/fresh-dual-bank/allocate-dual-bank')
   for owner in range(64):labels.extend(prefix+f'/fresh-dual-bank/prefill-owner{owner}/'+x for x in ('original-model','finite'))
   labels.extend((prefix+'/fresh-dual-bank/independent-prefixes-ready',prefix+'/Native-before'))
   for cycle in range(4):
    base=prefix+f'/cycle{cycle}/'
    labels.extend(base+s for s in (*GENERATE_STAGES,'original-proposals-and-typed-publication',*target,*PROTO_STAGES,
     'independent-own-KV-compaction-and-history','retain-completed-cycle'))
   labels.extend(prefix+'/next-draft/'+s for s in GENERATE_STAGES)
   labels.extend(prefix+'/'+s for s in ('retain-next-draft','complete-trace','release-dual-bank'))
 labels.extend(('exact-observed-inventory','MC2-after','MC2-positive-original-route','original-unified-seal',
  'final-Native','final-exact-inventory','all4-component-required-coverage'))
 return labels

def boundary_record(events):
 labels=expected_stages();need(len(events)==len(labels),'complete fixed all4 stage count')
 for seq,(event,label) in enumerate(zip(events,labels,strict=True)):
  need(event['sequence']==seq and event['label']==label,'exact nonnested original stage order '+str(seq))
  need([v['rank'] for v in event['ranks']]==list(range(4)) and all(v['sequence']==seq and v['label']==label
   and v['error'] is None for v in event['ranks']),'every target/prefill TP lane closes all4 before the next lane')
 return len(labels)

def draft_ABI(proof,budgets):
 need(proof['raw_encoded_shape']==[32,4,2] and proof['raw_pool_nodes']==256 and proof['actual_draft_model_stages']==4
  and proof['selected_C']==sum(budgets) and proof['variable_num_draft_tokens']==list(budgets),'separate rawpool/model-stage/selected-C units')
 raw,scores,means=proof['raw_pool'],proof['node_confidences'],proof['mean_confidence']
 need(len(raw)==len(scores)==len(means)==32,'draft exact row count')
 encoded=[]
 for tokens,confidence,mean in zip(raw,scores,means,strict=True):
  need(len(tokens)==len(confidence)==8 and all(type(t) is int and 0<=t<151936 for t in tokens),'full8 original draft rows')
  need(all(type(v) is float and math.isfinite(v) and 0<=v<=1 for v in confidence),'finite float32 node confidences')
  quant=[round(v*1_000_000) for v in confidence]
  need(confidence==[f32(v/1_000_000) for v in quant] and mean==f32(sum(v/1_000_000 for v in quant)/8),'original millionth encoding and original decoded mean8 order')
  need(all(tokens[j]!=tokens[4+j] and quant[j]>=quant[4+j] for j in range(4)),'actual sorted two distinct top2 tokens')
  encoded.append([[tokens[j]*1_000_001+quant[j],tokens[4+j]*1_000_001+quant[4+j]] for j in range(4)])
 need(digest(encoded)==proof['encoded_sha256'],'full physical encoded SHA independently reconstructs from original returned fields')
 return encoded

def causal_verdict(rows,query,budgets):
 need(len(rows)==len(budgets)==32 and len(query)==32+sum(budgets),'real Q and causal selected rows')
 emitted=[];accepted=[];offset=0
 for row,b in zip(rows,budgets,strict=True):
  need(len(row)==b,'actual selected B')
  k=0
  while k<b and row[k]==query[offset+k]:k+=1
  emitted.append(list(row[:k])+[query[offset+k]])
  accepted.append(list(range(k))+[-1]*(4-k));offset+=b+1
 return emitted,accepted

def target_key(m,s,hidden=False):
 attention='fia:exact-length-task-update|buffers:'+repr((((s,32),(s,32),(2048,2048)),))
 return (('target-hidden:0' if hidden else 'target-greedy:151936')+'|steps:1|'+attention,(m,))

def inventory(record,rank,repair_buckets):
 need(record['generic']==[] and record['all_original_runtime_validated'] is True and record['qualification_flags_assigned'] is False,'original inventory flags/cold lineage')
 draft={ast.literal_eval(x) for x in record['draft']};target={ast.literal_eval(x) for x in record['target']}
 if rank==0:
  expected={('draft-greedy:151936|steps:4|paged|confidence',32)}
  if repair_buckets:expected.add(('draft-greedy:151936|steps:3|paged|confidence',32))
  need(draft==expected and target=={('target-hidden:0|steps:1|paged',(m,)) for m in repair_buckets},'only actually observed draft/repair keys, no blindly filled PA buckets')
 else:need(not draft and target=={target_key(m,s,h) for m,s,_ in GEOMETRIES for h in (False,True)},'exact four original target keys')
 return len(draft)+len(target)

def actual_config(config,rank):
 need(config['max_num_seqs']==32 and config['max_num_queued_seqs']==64 and config['kvcache_block_size']==128
  and config['num_kvcache_blocks']==544 and config['max_model_len']==4096 and config['gamma']==8
  and config['draft_tp_size']==1 and config['target_tp_size']==3,'actual Native local32/queued64/block128/pages544/TP1+3 scope')
 need(config['draft_model']=='/root/data/shared-models/Qwen3-0.6B' and config['target_model']=='/root/data/shared-models/Qwen3-32B'
  and config['enable_mc2'] is True and config['precompile_decode_graphs'] is config['precompile_serial_draft_graphs'] is config['enable_prefix_caching'] is False,
  'unchanged original public models, MC2 and minimal constructor')
 need(config['spec_rhythm_tree_width']==2 and config['spec_rhythm_tree_depth']==4 and config['spec_rhythm_max_eager_tokens']==3
  and config['spec_rhythm_cpu_verdict'] is True and config['draft_use_paged_attention'] is config['target_use_paged_attention'] is True,'actual original configured tree/PA/verdict primitives')

@contextmanager
def original_auditor():
 blocked=[name for name in sys.modules if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers')]
 need(not blocked,'run this stdlib audit in a fresh interpreter without Torch/model imports')
 guard=NoDeviceImports();sys.meta_path.insert(0,guard);old=list(sys.path);sys.path.insert(0,str(P))
 try:
  for name in ('scope_v144','audit_v144','mc2_inventory_contract'):
   if name in sys.modules:need(Path(sys.modules[name].__file__).resolve()==P/(name+'.py'),'original auditor import collision')
  module=importlib.import_module('audit_v144')
  need(Path(module.__file__).resolve()==P/'audit_v144.py','actual frozen original auditor path')
  yield module
 finally:
  sys.path[:]=old;sys.meta_path.remove(guard)

def fixtures():
 source=B/'tree-s35-qualified-counts-v119/inputs.json';rows=document(source)['contexts']
 selected=[(i,row) for i,row in enumerate(rows) if len(row['token_ids'])>136][:32]
 need(len(selected)==32,'actual V136 selection closure')
 return [dict(owner=i,reference_owner=i+32,source_index=index,request_id=row['request_id'],slo_class=row['slo_class'],
  output_offset=0,prefix=128+i%4,token_ids=row['token_ids'][:129+i%4],full_source_token_count=len(row['token_ids']),
  full_source_token_sha256=digest(row['token_ids'])) for i,(index,row) in enumerate(selected)]

def check_source(source,initial,contexts):
 need(source['ids']==list(range(32)),'actual candidate owner IDs')
 expected=[dict(o,token_ids=list(v),initial_prefix=o['prefix'],prefix=len(v)-1,output_offset=len(v)-len(o['token_ids'])) for o,v in zip(initial,contexts,strict=True)]
 need(source['contexts']==expected,'actual source/request/prefix/history binding')

def check_lease(lease,reference=None):
 need(lease['pages']==140 and lease['storage_rows']==96 and lease['candidate_owners']==list(range(32))
  and lease['reference_owners']==list(range(32,64)) and lease['dummy_owners']==list(range(64,76))
  and lease['source_KV_copied'] is False and type(lease['allocation_id']) is int
  and re.fullmatch('[0-9a-f]{64}',lease['tables_sha256']),'actual simultaneous140-page independent lease guard')
 if reference is not None:need(lease==reference,'immutable per-trace actual lease through all original commits')

def environment(report,tag):
 env=report['actual_environment'];rank=report['rank']
 need(env['SPECSLO_VISIBLE_DEVICES']=='2,5,6,7' and env['ASCEND_RT_VISIBLE_DEVICES']=='0,1,2,3'
  and env['SPECSLO_TARGET_DEVICES']=='1,2,3' and env['RANK']==env['LOCAL_RANK']==str(rank) and env['WORLD_SIZE']=='4'
  and env['SPECSLO_COST_OUTPUT']==str(P/'actual'/tag),'actual retained physical/logical/model-rank mapping')
 need(report['physical_cards']==list(PHYSICAL) and report['physical_rank_card']==PHYSICAL[rank] and report['actual_target_leader_rank']==1
  and report['actual_logical_device']=='npu','actual rank/device metadata; physical mapping comes from pinned retained ACL mapper')
 for line in (P/'launch_v144.sh').read_text().splitlines():
  if not line.startswith('export '):continue
  import shlex
  for token in shlex.split(line[7:]):
   if '=' not in token:continue
   name,value=token.split('=',1)
   if name.startswith('VLLM_ASCEND_') and '$' not in value:need(env.get(name)==value,'actual unchanged Native env '+name)
 need(env['OMP_NUM_THREADS']==env['OPENBLAS_NUM_THREADS']=='1' and env['TASK_QUEUE_ENABLE']=='1'
  and env['HCCL_OP_EXPANSION_MODE']=='AIV' and env['HCCL_DETERMINISTIC']=='true'
  and env['PYTORCH_NPU_ALLOC_CONF']=='expandable_segments:True','actual original runtime threads/allocator/collectives')

def audit_actual(tag,receipt_sha):
 need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',tag) and re.fullmatch('[0-9a-f]{64}',receipt_sha),'ROOT tag and exact expected receipt SHA required')
 sources,identity=source_identity();receipt_path=P/f'receipt-{tag}.json'
 need(sha(receipt_path)==receipt_sha,'ROOT expected terminal receipt identity')
 receipt=document(receipt_path);directory=P/'actual'/tag
 need(receipt['schema']=='V144.actual-component-supervisor.v1' and receipt['tag']==tag and receipt['action']==ACTION
  and receipt['status']=='complete' and receipt['errors']==[] and receipt['owned_remaining']==[] and receipt['actual_execution_complete'] is True
  and receipt['sources_unchanged'] is True and receipt['child']['exit_code']==0,'terminal complete actual supervisor and owned cleanup')
 need(receipt['formal_Goodput'] is receipt['cost_measured'] is False,'component has no performance claim')
 need(receipt['execution_lock']==dict(path=str(B/'.ROOT_NPU_EXECUTION.lock'),nonblocking_exclusive=True,scope='this supervisor lifetime'),'actual ROOT file-lock source/receipt contract')
 need(receipt['child']['argv']==['bash',str(P/'launch_v144.sh'),'--action',ACTION,'--tag',tag] and receipt['child']['log']==str(P/f'local-{tag}.log'),'actual launcher command/path')
 need(receipt['limits']==dict(seconds=3600,private_bytes=128*1024**2) and 0<receipt['elapsed_seconds']<3600,'actual bounded component scope')
 idle_record(receipt['devices_before'],'before');idle_record(receipt['devices_after'],'after')
 pre=verifier_record(receipt['pre_verifier'],tag,'pre');post=verifier_record(receipt['post_verifier'],tag,'post')
 snapshot=P/'source-snapshots'/f'{tag}.tar.gz'
 need(receipt['archive']['path']==str(snapshot) and receipt['archive']['sha256']==sha(snapshot) and receipt['archive']['members']==117,'actual117 source snapshot binding')
 actual_archive=check_archive(snapshot,sources)
 need(document(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all4 actual worker exits')
 completed=document(directory/'completed.json')
 need(completed==dict(action=ACTION,replies=[['calibrated',r,dict(cases=2,timings=0)] for r in range(4)],cost_measured=False,formal_Goodput=False),'actual four original completion replies')
 outer=document(directory/'config.json')
 need(outer['max_num_seqs']==32 and outer['max_num_queued_seqs']==64 and outer['draft_model_path']=='/root/data/shared-models/Qwen3-0.6B'
  and outer['target_model_path']=='/root/data/shared-models/Qwen3-32B' and outer['num_kvcache_blocks']==544,'actual outer model/config snapshot')
 prerequisites=document(directory/'prerequisites.json');scope=prerequisites['scope']
 need(prerequisites==receipt['readiness'] and prerequisites['CPU_sha256']==EXPECTED[str(P/'CPU_RESULT_V144.json')]
  and prerequisites['freeze_sha256']==EXPECTED[str(P/'PREPARED_SOURCE_HASHES_V144.json')]
  and prerequisites['actual_device_qualification'] is False,'actual main/supervisor same pending-device source readiness')
 need(scope['vectors']==[list(v) for v in VECTORS] and scope['geometries']==[[m,s,list(d)] for m,s,d in GEOMETRIES]
  and (scope['maxseq'],scope['maxqueued'],scope['storage'],scope['block_size'],scope['pages'])==(32,64,96,128,544)
  and scope['total_real_commits']==24 and scope['complete_local_protocol_zero_traces']==2 and scope['future_full_two65zero_required'] is True,
  'actual startup readiness binds this local scope, not full527')
 with original_auditor() as original:
  strict=original.audit_reports(directory)
 need(json.loads(json.dumps(strict,allow_nan=False))==receipt['actual_component_audit'],
  'retained original strict audit canonical JSON equals terminal supervisor result (no volatile directory-size comparison)')
 reports=[gunzip(directory/f'rank{r}-component.json.gz') for r in range(4)]
 initial=fixtures();repair_buckets=set();protocol_missing={0:False,1:False};final_missing={0:False,1:False};summaries=[]
 all_paths=[];expected_private={directory/'prerequisites.json',directory/'config.json',directory/'completed.json',directory/'worker-exit.json'}
 for rank,report in enumerate(reports):
  actual_config(report['actual_config'],rank);environment(report,tag)
  construction=report['construction'];need(construction['layers']==(28 if rank==0 else 64)
   and construction['tree_scratch']==list(range(64,96)) and not any(construction['initial_native'].values()),'new actual96storage and original cold constructor')
  need(report['initial_source_root_teacher_forced_not_own_AR'] and report['subsequent_draft_verdict_commit_actual']
   and report['future_full_requires_21scope26phase_and_two65zero'],'honest source-root/component/future-full scope')
  boundary_count=boundary_record(report['boundary_events'])
  if rank:need(report['boundary_events']==reports[0]['boundary_events'],'all4 actual raw common-stage packets identical')
  for name,row in report['actual_source_origins'].items():need(sources.get(row['path'])==row['sha256'],'actual imported source origin SHA '+name)
  counters=report['MC2_dispatch'];need(counters['delta']=={k:counters['after'][k]-counters['before'][k] for k in counters['before']}
   and all(v>=0 for v in counters['delta'].values()) and counters['delta']['fallback']==counters['delta']['exception']==0
   and (rank==0 or counters['delta']['native_epilogue_success']>0),'actual own-process MC2 dispatch raw arithmetic')
  state=report['MC2_static'];startup=directory/'startup-progress'/f'MC2-preflight-rank{rank}.json';expected_private.add(startup)
  need(state==document(startup) and state['native_qualification_flags_modified'] is False,'actual MC2 startup report identity')
  if rank:need(state['manifest_route_count']==128 and len(state['route_details'])==128 and state['model_chained_row_counts']==list((40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197)),'actual64layer/128route MC2 manifest')
  total_delta={k:0 for k in report['final_Native']}
  for trace in report['traces']:
   p,c=trace['pass_index'],trace['case'];budgets=VECTORS[c];m,s,dummy=GEOMETRIES[c]
   need((trace['M'],trace['S'])==(m,s),'actual variable budget maps to original physical M/S')
   bank=trace['lease'];check_lease(bank['lease']);contexts=[list(o['token_ids']) for o in initial]
   need(len(bank['prefill'])==64 and bank['initial_root_teacher_forced_not_own_AR'] is True,'all independent prefix owners and honest initial source fixture retained')
   for owner,prefill in enumerate(bank['prefill']):
    tokens=initial[owner%32]['token_ids'][:-1]
    need(prefill==dict(owner=owner,tokens=len(tokens),sha256=digest(tokens),actual_eager_PA=True,candidate_KV_copied=False),'actual separately evaluated source prefix '+str(owner))
   hashes=bank['initial_prefix'];need(hashes['tokens']==digest({i:initial[i%32]['token_ids'][:-1] for i in range(64)})
    and len(hashes['KV'])==(56 if rank==0 else 128) and all(re.fullmatch('[0-9a-f]{64}',h) for h in hashes['KV']),'all28/all64 prefix guard hash count and source tokens')
   for k,v in trace['Native']['delta'].items():total_delta[k]+=v
   local_delivered=0;repair_commits=0;raw_fingerprints=[]
   for cycle in range(4):
    path=directory/f'rank{rank}-pass{p}-case{c}-cycle{cycle}.json.gz';expected_private.add(path);all_paths.append(path);row=gunzip(path)
    check_source(row['source'],initial,contexts)
    need(row['work_plan_id']==(p*2+c)*5+cycle and row['selected_normal_candidates']==sum(budgets)
     and row['controlled_typed_fixture_grants_not_policy_allocation'] is True,'current controlled typed plan identity and selected C, not live allocator evidence')
    check_lease(row['independent_commit']['lease'],bank['lease'])
    commit=row['commit'];need(commit['cycle_id']==cycle and commit['target_home']==cycle%2
     and commit['actual_prefix_epochs']=={str(i):cycle+1 for i in range(32)} and commit['original_decode_sha256']==ORIGINAL_DECODE
     and commit['canonical_decode_sha256']==CANONICAL_DECODE and commit['extracted_structural_and_numerical_lanes_unchanged'] is True,'actual original protocol/epoch/rotation binding')
    raw_fingerprints.append(dict(source=row['source'],selected=row['selected'],rows=row['rows'],confidences=row['confidences'],
     commit={k:commit[k] for k in ('committed_tokens','accepted_paths','actual_prefix_epochs','actual_logical_tokens','actual_structural_failed','actual_numerical_failed')}))
    leader=gunzip(directory/f'rank1-pass{p}-case{c}-cycle{cycle}.json.gz');query=leader['target']['greedy'][2]['full_physical_tokens'][:32+sum(budgets)]
    emitted,accepted=causal_verdict(row['rows'],query,budgets)
    need(commit['committed_tokens']==emitted and commit['accepted_paths']==accepted,'HOST independently recomputed causal B1..4 original observed verdict')
    contexts=[old+tokens for old,tokens in zip(contexts,emitted,strict=True)]
    need(commit['actual_logical_tokens']==contexts,'actual own recurrence, never teacher-force later outputs')
    local_delivered+=sum(map(len,emitted))
    if rank==0:
     proof=row['draft'];draft_ABI(proof,budgets)
     need(proof['roots']==[v['token_ids'][-1] for v in row['source']['contexts']],'actual draft input roots')
     need(row['rows']==[raw[:b] for raw,b in zip(proof['raw_pool'],budgets,strict=True)],'original primary selected rows from full8 pool')
     selected_conf=[sum(scores[:b])/b for scores,b in zip(proof['node_confidences'],budgets,strict=True)]
     need(row['confidences']==selected_conf,'exact selected mean confidence preserved in original Float64 packet')
     if proof['natural_missing_suffix_observed']:repair_buckets.add(proof['first_PA_physical']);protocol_missing[c]=True;repair_commits+=1
    else:
     for arm in ('greedy','hidden'):
      phases=row['target'][arm];need([x['phase'] for x in phases]==['first','changed','restored'],'actual F/T/F phase order')
      warm=p>0 or cycle>0
      for ordinal,value in enumerate(phases):
       need(value['key']==repr(target_key(m,s,arm=='hidden')) and value['runtime_validated_before']==(warm if ordinal<2 else True)
        and value['runtime_validated_after']==(warm if ordinal==0 else True),'actual natural cold/changed/restored flags')
       execution=value['execution']
       # asdict retains fields, not NativeGraphExecution.used_aclgraph property.
       need(set(execution)=={'mode','fallback_reason','capture_attempted','replay_executed'}
        and execution['mode'] in ('capture_replay','replay') and execution['fallback_reason'] is None
        and execution['replay_executed'] is True,'actual Native graph execution via its original serialized mode/fields')
      if arm=='greedy':need(phases==leader['target']['greedy'],'all3TP returned full physical token/F-T-F evidence agrees')
     need(row['reference']['same_M']==m and row['reference']['same_S']==s and row['reference']['real_query_tokens']==query,'independent exact same-M reference query identity')
     need(row['independent_commit']['own_reference_query_KV_compacted'] is True,'reference commits only its own query KV')
   nextpath=directory/f'rank{rank}-pass{p}-case{c}-next-draft.json.gz';expected_private.add(nextpath);all_paths.append(nextpath);nxt=gunzip(nextpath)
   check_source(nxt['source'],initial,contexts);need(nxt['work_plan_id']==(p*2+c)*5+4,'actual final next-draft plan identity')
   need(trace['fingerprints']==raw_fingerprints and trace['next_draft']=={k:nxt[k] for k in ('source','selected','rows','confidences')},
    'zero-trace fingerprints are rebuilt from the actual retained cycle/nextdraft checkpoints')
   need(bank['current_context_sha256']==[digest(v) for v in contexts],'retained independent bank final own histories')
   if rank==0:
    draft_ABI(nxt['draft'],budgets)
    if nxt['draft']['natural_missing_suffix_observed']:repair_buckets.add(nxt['draft']['first_PA_physical']);final_missing[c]=True
    summaries.append(dict(pass_index=p,case=c,C=sum(budgets),Q=32+sum(budgets),M=m,S=s,dummy_lengths=list(dummy),
     actual_commits=4,actual_delivered_tokens=local_delivered,repaired_draft_then_committed_cycles=repair_commits,
     final_next_draft_repair=nxt['draft']['natural_missing_suffix_observed']))
  need(total_delta=={k:report['final_Native'][k]-construction['initial_native'][k] for k in total_delta},'all raw Native changes belong to the exact six traces')
  keys=inventory(report['final_inventory'],rank,repair_buckets)
  need(report['inventory_before_seal']==report['final_inventory'] and report['final_Native']['aclgraph_captures']==keys
   and report['final_Native']['aclgraph_capture_attempts']==keys,'exact observed entries/cold captures/no unrelated keys')
  finalpath=directory/f'rank{rank}-component.json.gz';expected_private.add(finalpath)
  need(not report['adopted'] and not report['full_workload_qualified'] and not report['formal_Goodput'],'local scope cannot become full527/cap4 adoption')
 need({p for p in directory.rglob('*') if p.is_file()}==expected_private,'exact retained4reports/96cycle/24next/4startup/4control artifacts')
 need(len(all_paths)==120 and len(expected_private)==132,'24commits/6nextdraft each on4roles; no extra exploration matrix')
 missing=[c for c in range(2) if not protocol_missing[c]]
 need(strict['missing_suffix_uncovered_vectors']==missing and strict['component_numerical_qualified']==(not missing),'mandatory repaired-and-committed coverage is independent of final-only draft coverage')
 private_bytes=sum(p.stat().st_size for p in P.rglob('*') if p.is_file());need(private_bytes<128*1024**2,'retained whole component private storage bound')
 need('torch' not in sys.modules and 'torch_npu' not in sys.modules,'actual external auditor stayed stdlib')
 return dict(schema='ROOT.V144.actual-independent-component-audit.v1',status='passed',receipt=artifact(receipt_path),tag=tag,
  source_identity=identity,actual_source_snapshot=actual_archive,pre_verifier=pre,post_verifier=post,
  source_current_hashes_all_checked=117,retained_attempts_preserved=True,all4_idle_before_after_raw_recomputed=True,owned_remaining=[],
  historical_file_lock=dict(recorded_contract=receipt['execution_lock'],source_implementation_pinned=True,OS_hold_not_remeasured_after_run=True),
  worker_exit=True,original_strict_audit=strict,actual_config_scope=dict(maxseq=32,maxqueued=64,storage=96,block=128,pages=544,dual_bank_lease_pages=140),
  exploration_commits=8,fixed_protocol_replay_commits=16,next_draft_probes=6,complete_local_protocol_zero_traces=2,
  all4_fixed_stage_count=boundary_count,actual_trace_summary=summaries,natural_protocol_repair_coverage=protocol_missing,
  final_next_draft_repair_coverage=final_missing,observed_draft_repair_buckets=sorted(repair_buckets),
  component_numerical_qualified=not missing,adopted=False,full527_or_full463_qualified=False,formal_Goodput=False,cost_measured=False,
  initial_source_root_teacher_forced_not_own_AR=True,subsequent_actual_recurrence_recomputed=True,
  evidence_limits=['Full hidden/KV tensors were not retained; HOST checks exact fields/hashes and the original executed matcher markers.',
   'Actual full block tables were not retained; immutable lease hashes/counts and per-cycle guard consistency are checked, not a new raw-page enumeration.',
   'Own same-M comparisons do not prove arbitrary cross-M or full workload token equality.',
   'Final-only repair does not close repair publication/target/commit coverage.',
   'Future full still requires original21scope26phase and two full65 zero traces.'],
  private_bytes_observed_after_terminal=private_bytes,Torch_imported=False,device_query=False,model_initialized=False)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True)
 parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',required=True,type=Path)
 args=parser.parse_args();need(args.output.resolve().is_relative_to(D) and not args.output.exists(),'fresh diagnostics-only output')
 started=time.perf_counter();result=None;exit_code=0
 try:result=audit_actual(args.tag,args.expected_receipt_sha256)
 except BaseException:
  result=dict(schema='ROOT.V144.actual-independent-component-audit.v1',status='failed',tag=args.tag,
   expected_receipt_sha256=args.expected_receipt_sha256,error=traceback.format_exc(),formal_Goodput=False,actual_component_qualified=False)
  exit_code=1
 result.update(auditor=artifact(__file__),audit_UTC=datetime.now(timezone.utc).isoformat(),audit_elapsed_seconds=time.perf_counter()-started,
  actual_argv=sys.argv,actual_interpreter=dict(executable=sys.executable,version=sys.version),
  actual_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TORCH_DEVICE_BACKEND_AUTOLOAD','PYTHONDONTWRITEBYTECODE')},
  retained_runtime_wrapper_claimed=False,wrapper_provenance='not inferred from inner argv/environment; use the ROOT execution log')
 with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],output=str(args.output),sha256=sha(args.output),
  component_numerical_qualified=result.get('component_numerical_qualified'),actual_device_launched_by_auditor=False)))
 if exit_code:raise SystemExit(exit_code)

if __name__=='__main__':main()
