"""Two bounded component fixtures. Importing this module performs no device work."""
from pathlib import Path
import hashlib
import json

P=Path(__file__).resolve().parent
B=P.parent
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
ACTION='variable-B4-local-qualification'
VECTORS=((4,)*19+(3,)+(2,)*5+(1,)*7,(4,)*19+(2,)*6+(1,)*7)
GEOMETRIES=((128,32,()),(165,38,(7,7,6,6,6,6)))
MAX_CYCLES=4
NATIVE_COUNTERS=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
 'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',
 *(f'aclgraph_{role}_{field}' for role in ('generic','draft','target') for field in
 ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
FAILURES=tuple(k for k in NATIVE_COUNTERS if any(s in k for s in ('failed_captures','fallback','disabled_entry','unclassified','validation_failures')))
MC2_ROWS=(40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197)

def require(condition,message):
 if not condition:raise RuntimeError('V146: '+message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def cases():
 source=B/'tree-s35-qualified-counts-v119/inputs.json'
 rows=json.loads(source.read_text())['contexts']
 selected=[(i,r) for i,r in enumerate(rows) if len(r['token_ids'])>136][:32]
 require(len(selected)==32,'first32 V136-provider genuine contexts longer than136')
 owners=[]
 for owner,(index,row) in enumerate(selected):
  tokens=row['token_ids'];p=128+owner%4
  require(all(type(t) is int and 0<=t<151936 for t in tokens),'source token type/vocabulary')
  owners.append(dict(owner=owner,reference_owner=owner+32,source_index=index,request_id=row['request_id'],
   slo_class=row['slo_class'],output_offset=0,prefix=p,token_ids=tokens[:p+1],
   full_source_token_count=len(tokens),full_source_token_sha256=digest(tokens)))
 return [dict(index=i,name='C'+str(sum(v)),N=32,C=sum(v),Q=32+sum(v),budgets=list(v),
  M=g[0],S=g[1],dummy_lengths=list(g[2]),owners=owners,source_path=str(source),source_sha256=sha(source),
  initial_root_teacher_forced_not_own_AR=True,subsequent_draft_and_verdict_actual=True)
  for i,(v,g) in enumerate(zip(VECTORS,GEOMETRIES,strict=True))]

def target_key(case,*,hidden=False):
 s=case['S'];attention='fia:exact-length-task-update|buffers:'+repr((((s,32),(s,32),(2048,2048)),))
 return (('target-hidden:0' if hidden else 'target-greedy:151936')+'|steps:1|'+attention,(case['M'],))

def validate_full_configuration(engine):
 require(engine.config.max_num_seqs==64 and engine.config.max_num_queued_seqs==463
  and engine.config.kvcache_block_size==128 and engine.prefix_cache.num_blocks==544
  and engine._cache_storage_sequence_capacity==527
  and tuple(engine._tree_graph_scratch_sequence_ids)==tuple(range(463,527))
  and not engine._mixed_target_graph_enabled and not engine._mixed_target_scratch_sequence_ids,
  'same full64/queued463/storage527/block128/pages544/flag0 actual configuration')
 require(not engine.graph_runner.graph_cache_sealed and engine.cache_allocation is None,
  'own full-process variable-B4 qualification precedes unified seal and uses a fresh cache')
 return dict(maxseq=64,maxqueued=463,storage=527,block_size=128,pages=544,
  candidate_owners=list(range(32)),reference_owners=list(range(32,64)),
  candidate_dummy_owners=list(range(463,469)),reference_dummy_owners=list(range(469,475)),
  ordinary_mixed_graph_enabled=False,tree_scratch=list(engine._tree_graph_scratch_sequence_ids))


def inventory_record(engine):
 runner=engine.graph_runner
 return {family:sorted(repr(key) for key in getattr(runner,family))
  for family in ('entries','draft_entries','target_entries')}
