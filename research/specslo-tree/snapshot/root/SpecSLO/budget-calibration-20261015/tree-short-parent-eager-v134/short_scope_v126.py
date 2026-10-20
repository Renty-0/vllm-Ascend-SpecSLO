"""Four representative source/PA/protocol cases; no full pipeline claim."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
from bootstrap_v126 import P,V119,V107,V102,V123
from target_layout import envelope
VOCAB=151936
ACTION='parent3-child3-repair01-local-protocol'
SCHEMA='V134.short-parent-FastV5-split-draft-protocol.v1'

def require(value,message):
 if not value:raise RuntimeError('V126: '+message)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@dataclass(frozen=True)
class Case:
 name:str
 normal:int
 eager:int
 repair:int
 prefix:int=128
 parent_B:int=3
 normal_B:int=3
 child_B:int=3
 @property
 def count(self):return self.normal+self.eager
 @property
 def repairs(self):return (self.repair,)*self.normal
 @property
 def selected(self):return self.normal*self.normal_B+self.eager*self.child_B
 @property
 def parent_ids(self):
  # This is a SEQUENTIAL component verification phase, after generation and
  # publication, never a simultaneous normal-draft/target live owner claim.
  return tuple(range(min(self.normal,32-self.eager)))+tuple(range(self.normal,self.count))
 @property
 def deferred_normal(self):return tuple(i for i in range(self.normal) if i not in self.parent_ids)

def cases():return (Case('N13E1-B3B3-repair0',13,1,0),Case('N13E1-B3B3-repair1',13,1,1),Case('N32E1-B3B3-repair1',32,1,1),Case('N32E32-B3B3-repair1-endpoint',32,32,1))

def target_shape(B,N):
 require(B in (1,2,3,4,5) and 1<=N<=32 and B*N<=96 and (B+1)*N<=128,'unchanged real target C96/Q128/N32')
 family='causal' if B<=4 else 'full9';M,S,scratch=envelope(N,(B+1)*N,family)
 return dict(B=B,N=N,real_Q=(B+1)*N,M=M,S=S,family=family,scratch=scratch)

def key(g,kind):
 signature=(((g['S'],32),(g['S'],32),(2048,2048)),) if g['family']=='causal' else (((g['S'],1,9,4096),(g['S'],32)),)
 attention=('fia' if g['family']=='causal' else 'tree-fia-full')+':exact-length-task-update|buffers:'+repr(signature)
 return (('target-greedy:'+str(VOCAB) if kind=='greedy' else 'target-hidden:0')+'|steps:1|'+attention,(g['M'],))

def bucket(rows):
 require(type(rows) is int and 1<=rows<=64,'actual draft64 row bound')
 return next(n for n in (1,2,4,8,12,16,20,24,28,32,36,40,44,48,64) if n>=rows)

def source_contexts(case):
 path=V102/'inputs.json';values=json.loads(path.read_text())['contexts'];cats=('coding','chat','summarization','coding')
 result=[next(v for v in values if v['category']==cats[i%4] and len(v['token_ids'])>=264) for i in range(case.count)]
 require(all(len(v['token_ids'])>case.prefix+case.repair for v in result),'genuine source has full prefix/suffix/root')
 return result

def source_contract():
 gs={key(target_shape(3,n),kind) for n in (1,2,14,32) for kind in ('greedy','hidden')}
 return dict(cases=[dict(**c.__dict__,selected_normal=c.normal*3,selected_future_eager=c.eager*3,selected_total=c.selected,
  real_work_rows=c.count,actual_shared_steps=3,actual_chain_M=bucket(c.count),generated_real_nodes=6*c.count,
  ABI_returned_slots=8*c.count,unavailable_per_row_indices=[3,7],actual_chain_physical_queries=3*bucket(c.count),
  current_target_owners=c.parent_ids,deferred_normal_owners=c.deferred_normal,
  parent_target=target_shape(3,len(c.parent_ids)),following_target_batches=[dict(phase='deferred-normal',owners=c.deferred_normal,shape=target_shape(3,len(c.deferred_normal))) for _ in (0,) if c.deferred_normal]+[dict(phase='promoted-children',owners=list(range(c.normal,c.count)),shape=target_shape(3,c.eager))]) for c in cases()],
  target_keys=sorted(map(repr,gs)),mandatory_target_graph_keys=6,possible_observed_sparse_promotion_target_keys=sorted({repr(key(target_shape(3,n),kind)) for n in range(1,33) for kind in ('greedy','hidden')}),new_MC2_exact_profiles=0,
  maximum_selected_candidates_in_declared_scope=192,selected192_endpoint_qualified=False,work64_endpoint_qualified=False,
  max_work_rows=64,max_num_seqs=64,max_num_queued_seqs=128,component_storage_rows=128,private_scratch_ids=list(range(64,102)),online_storage527_qualified=False,
  original_SLO_acceptance_floor=.4,window_abi_exploration_units_per_row=8,
  eligibility_window_need_reserve_bypassed_in_live=False,online_scheduler_installed=False,
  source_calls_controlled_sequential_not_live_round=True,teacher_reference_uses_candidate_KV=False,
  actual_device_qualified=False,full_workload_qualified=False,formal_Goodput=False)

NATIVE_COUNTERS=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
 'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',
 *(f'aclgraph_{r}_{f}' for r in ('generic','draft','target') for f in
 ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
FAILURE_COUNTERS=tuple(k for k in NATIVE_COUNTERS if any(s in k for s in ('failed','fallback','disabled','unclassified','validation_failures')))

MAX_PRIVATE_BYTES=32*1024**2
def make_config():
 from vllm_ascend.spec_decode.pearl.api import PEARLConfig
 return PEARLConfig(draft_model_path='/root/data/shared-models/Qwen3-0.6B',target_model_path='/root/data/shared-models/Qwen3-32B',
  draft_tensor_parallel_size=1,target_tensor_parallel_size=3,draft_dtype='auto',target_dtype='auto',
  draft_mode='serial_linear',gamma=8,seed=0,max_num_seqs=64,max_num_queued_seqs=128,
  max_num_batched_tokens=131072,max_model_len=4096,gpu_memory_utilization=.8,kvcache_block_size=128,
  num_kvcache_blocks=544,max_aclgraph_entries=512,enforce_eager=False,enable_prefix_caching=False,
  enable_continuous_batching=False,enable_preemptive_scheduling=False,enable_spec_rhythm=False,
  spec_rhythm_online_prefill=False,spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4,
  draft_use_paged_attention=True,target_use_paged_attention=True,precompile_decode_graphs=False,
  precompile_serial_draft_graphs=False,enable_mc2=True,
  mc2_profile='/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json',
  spec_rhythm_cpu_verdict=True,compact_worker_result_metrics=True,worker_timeout_seconds=3600)
