"""Small stdlib-only V124 arithmetic/source memo. No model/device operations."""
from pathlib import Path
import ast
import datetime
import gzip
import hashlib
import json
import math
import statistics
import sys
BASE=Path('/root/SpecSLO/budget-calibration-20261015')
D=BASE/'budget-diagnostics'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
JSON_OUT=D/'V124_CAUSAL_M384_M512_PREFILL_FEASIBILITY.json'
MEMO_OUT=D/'V124_CAUSAL_M384_M512_PREFILL_FEASIBILITY_zh.md'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def descriptor(path):return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=sha(path))
def source(path,names):
 text=path.read_text();tree=ast.parse(text)
 return dict(**descriptor(path),functions=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,
  source_sha256=hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()) for n in ast.walk(tree)
  if isinstance(n,ast.FunctionDef) and n.name in names])

def collect_trace(path):
 t=json.loads(gzip.decompress(path.read_bytes()));inputs=t['tokenized_inputs'];req=t['requests']
 assert t['role']=='full463' and t['scope_count']==len(inputs)==len(req)==463
 assert [r['request_index'] for r in req]==list(range(463)) and all(type(x) is list and x and all(type(v) is int for v in x) for x in inputs)
 assert len(t['generate_results'])==463 and all(len(r['completion_token_ids'])==256 for r in t['generate_results'])
 assert sorted(e['request_index'] for e in t['stream_events'] if e['finished'])==list(range(463))
 lengths=list(map(len,inputs));tokensha=list(map(digest,inputs));stand={}
 for call in t['standalone_prefill_calls']:
  assert len(call['sequence_ids'])==len(call['prompt_lengths'])==len(call['cached_tokens'])==len(call['prompt_sha256'])
  for owner,length,cached,h in zip(call['sequence_ids'],call['prompt_lengths'],call['cached_tokens'],call['prompt_sha256']):
   assert owner not in stand and length==lengths[owner] and cached==0 and h==tokensha[owner]
   stand[owner]=dict(prompt_length=length,prompt_sha256=h)
 reservations={}
 for e in t['logical_events']:
  if e['kind']=='reservation':
   assert e['prompt_length']==lengths[e['owner']] and e['prompt_sha256']==tokensha[e['owner']]
   reservations[e['owner']]=e
 published=[e['owner'] for e in t['logical_events'] if e['kind']=='publication']
 assert len(set(published))==len(published) and not set(published)&set(stand)
 assert set(published)|set(stand)==set(range(463)) and all(o in reservations for o in published)
 paths={o:'standalone_prefill_returned_generation_completed' for o in stand}
 paths.update({o:'mixed_terminal_publication_generation_completed' for o in published})
 case=t.get('case','V96-active64-B5-policy')
 return dict(**descriptor(path),case=case,request_count=463,prompt_tokens=sum(lengths),
  actual_prompt_lengths_known=463,unknown_request_indices=[],lengths_sha256=digest(lengths),
  tokenized_input_sha256=digest(inputs),standalone_completed_requests=len(stand),mixed_published_owners=published,
  successful_generation_all463_output256=True,prefill_argument_records_are_pre_call_snapshots=True,
  completion_evidence='Original generate_batch returned all463 outputs; original stream finished each request exactly once; disjoint complete standalone/publication first-token paths',
  allocator=t['allocator'],per_request_completion_paths=paths),inputs,req

def main():
 assert sys.version_info[:2]==(3,11)
 sources=[BASE/'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1/rank1-full463-actual-trace.json.gz']
 sources+=sorted((BASE/'tree-active-cap-controls-v103/local-output/active32-64-cap3-4-full463-v1').rglob('rank1-full463-actual-trace.json.gz'))
 assert len(sources)==4
 traces=[];baseline=None;requests=None
 for path in sources:
  trace,inputs,req=collect_trace(path);traces.append(trace)
  if baseline is None:baseline=inputs;requests=req
  else:assert inputs==baseline and [(r['request_id'],r['slo_class']) for r in req]==[(r['request_id'],r['slo_class']) for r in requests]
 lengths=list(map(len,baseline));classes={}
 for kind in ('tight','normal','loose'):
  ls=[lengths[i] for i,r in enumerate(requests) if r['slo_class']==kind]
  classes[kind]=dict(requests=len(ls),prompt_tokens=sum(ls),min=min(ls),max=max(ls),median=statistics.median(ls),
   hypothetical_chunks_by_prompt_capacity={str(cap):sum(math.ceil(n/cap) for n in ls) for cap in (65,252,380)},
   full_prompt_fits_one_chunk={str(cap):sum(n<=cap for n in ls) for cap in (65,252,380)})
 assert sum(v['prompt_tokens'] for v in classes.values())==sum(lengths)==203850
 profile=NATIVE.parents[3]/'runtime/mc2-profile.json'
 # Native source tree is nested under source/vllm-ascend-hust; use exact runtime path, never infer a sibling.
 profile=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json')
 document=json.loads(profile.read_text());measurements=document['measurements'];mrows=sorted(set(v['m'] for v in measurements))
 assert not any(v['m'] in (384,512) for v in measurements)
 capacities=[]
 for M in (197,384,512):
  P=M-128-2*2;assert P in (65,252,380)
  capacities.append(dict(M=M,S=35,N=32,Q=128,C=96,scratch_owners=2,minimum_scratch_queries_each=2,
   max_prompt_queries_without_FULL_mask_cap=P,physical_partition=[*([4]*32),P,2,2],
   current_FULL197_capacity=65,prompt_capacity_ratio_vs65=P/65,
   maximum2301_prompt_hypothetical_chunks=math.ceil(2301/P),
   all463_hypothetical_chunks=sum(math.ceil(n/P) for n in lengths),
   all463_full_prompt_fits_one_chunk=sum(n<=P for n in lengths),
   exact_MC2_profile_rows_present=any(v['m']==M for v in measurements),
   new_target_hidden_key=None if M==197 else repr(('target-hidden:0|steps:1|fia:exact-length-task-update|buffers:(((35, 32), (35, 32), (2048, 2048)),)',(M,)))))
 assert [c['all463_hypothetical_chunks'] for c in capacities]==[3354,1056,811]
 files=[source(NATIVE/'native_engine.py',{'_run_packed_hidden','_prepare_attention_metadata'}),
  source(NATIVE/'native_graph.py',{'run_tree_hidden','_run_target','_target_outputs_match'}),
  source(NATIVE/'native_model.py',{'make_attention_metadata','make_tree_attention_metadata','freeze_mc2_static_routes','_qualified_mc2_chained_flush_row_counts'}),
  source(NATIVE/'mc2.py',{'_native_epilogue_static_shape_error','qualify_static_shape','bind_mc2_static_route'}),
  source(BASE/'tree-capacity64-runtime-v96/capacity64_probe_v96.py',{'run_trace','finalize_prefill_records'}),
  source(BASE/'tree-s35-combined-controls-launchfix-v116/flexible_scope.py',{'envelope'}),
  source(BASE/'tree-s35-combined-controls-launchfix-v116/flexible_layout.py',{'build_layout','prepare_metadata'}),
  source(BASE/'tree-s35-combined-controls-launchfix-v116/live_mixed_metadata.py',{'prepare_metadata'}),
  source(BASE/'tree-linear-mixed-causal-v117/linear_mixed_metadata_v117.py',{'packed','prepare'}),
  source(BASE/'tree-linear-mixed-causal-v117/linear_mixed_actor_v117.py',{'PrefixBank','qualify','independent_reference','timed_prelude','measure_timed'})]
 negative=D/'ROOT_V117_ACTUAL_S19-S35-FULL-causal197-v1_AUDIT.json';old=json.loads(negative.read_text())
 assert old['adoption_recommended'] is False
 result=dict(schema='V124.HOST-only-causal-prefill-capacity-feasibility.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  actual_HOST_argv=sys.orig_argv,actual_outer_execution='retained serial-v1/runtime.sh --environment-only; stdlib only',
  source_script=descriptor(Path(__file__)),actual_full463_leader_traces=traces,raw_rank_scope='actual target leader rank1 only; this memo does not re-audit four-rank workload correctness',
  actual_prompt_summary=dict(requests=463,known=463,unknown=0,total_tokens=sum(lengths),min=min(lengths),max=max(lengths),median=statistics.median(lengths),by_class=classes),
  per_request_prompt_evidence=[dict(request_index=i,request_id=r['request_id'],slo_class=r['slo_class'],prompt_tokens=lengths[i],actual_token_ids_sha256=digest(baseline[i]),
   original_text_prompt_sha256=r['prompt_sha256'],completion_paths={t['case']:t['per_request_completion_paths'][i] for t in traces}) for i,r in enumerate(requests)],
  hypothetical_capacities=capacities,arithmetic_scope='All463 hypothetical independent chunks from cursor0, N32/Q128 fixed every call; not actual mixed calls, service cycles, saved runtime or end-to-end speedup. Actual online normal cohorts and remaining cursors vary.',
  MC2_profile=dict(**descriptor(profile),measurement_count=len(measurements),measurement_M_rows=mrows,
   missing_exact_shapes=[dict(M=M,K=K,N=5120,dtype='bfloat16',weight_format='ND',transposed=True,TP=3) for M in (384,512) for K in (3072,8576)],
   current_M197_attention_and_down_present=True,new_M384_M512_original_chain_rows_qualified=False,
   source_behavior='Unprofiled static M returns unqualified ticket; ordinary projection/allreduce/RMSNorm split path, no MC2 chain. Native fallback counters0 does not imply actual MC2 fused dispatch. Do not reuse M197 profile decisions for other M.'),
  source_anchors=files,V117_negative_evidence=descriptor(negative),
  feasibility=dict(Native_token_capacity512_source_allows_M384_M512=True,new_causal_S35_keys=2,
   minimal_with_fresh_M197_control_keys=3,existing_FULL197_branch_fallback_preserved=True,
   max_model_len=4096,block_size=128,block_table_columns=32,pool_blocks=544,
   metadata_requirements='actual linear ancestry/RoPE/cache positions, exact cumulativeQ and KV lists, unique page/slot ownership; no widened policy C96/Q128/target32',
   model_and_new_keys_actual_qualified=False,new_MC2_exact_shapes_actual_qualified=False,graph_cache_flags_inherited=False,
   route_M384_M512_with_original_profile='counterfactual split-route component only; no MC2-equivalent cost claim',
   independent_component_cost_possible_in_future_new_scope=True,install_in_full_allowed=False),
  known_gates=['Fresh NEW prefix bank and separate same-M independent original Native builder; no candidate KV reference initialization',
   'All384/512 physical hidden, all64 queryKV and exact real normal verdict/final prompt token, plus dummy outputs separated',
   'Native natural cold/changed/restored, MC2 actual capture/replay deltas and full exact profile/route identity',
   '2zero then original seal, no new discovery/fallback; full prefix series start/end byte hashes and each-call lease/queryslot restore',
   'Distinct all4 prelude-ready stage before forward-only timing; actual64 task updates, no skip',
   'V117 cross-kernel real-verdict disagreement remains NOTADOPT; M197/384/512 cross-M real-output disagreements must remain visible, no tolerance widening',
   'Actual HBM/pages/leases/cleanup/source closure; old measured page headroom is not a new-shape admission guarantee'],
  model_initialized=False,NPU_initialized=False,whole_workload_qualified=False,independent_AR_qualified=False,performance_gain_proved=False,formal_Goodput=False)
 # Preserve the compact summary without repeating all463 path maps in each trace.
 for t in result['actual_full463_leader_traces']:t.pop('per_request_completion_paths')
 with JSON_OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status='HOST-arithmetic-complete',output=str(JSON_OUT),sha256=sha(JSON_OUT),actual_prompt_tokens=sum(lengths),by_class=classes,new_capacity=[(x['M'],x['max_prompt_queries_without_FULL_mask_cap']) for x in capacities],NPU_initialized=False)))
if __name__=='__main__':main()
