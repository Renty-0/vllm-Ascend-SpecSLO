"""Genuine M80 against its independent ordinary M80 computation, and M120.

The old cross-M120 discrepancy remains an observation, not a passing test.
This one-frame diagnostic grants neither serving nor own-AR qualification.
"""
from dataclasses import replace
from pathlib import Path
import hashlib,json,math,statistics
from whole455 import unfused_forward

def measure(engine,payloads,output):
 import torch
 import torch.distributed as dist
 state=engine._branch_tail_state;current=state.static_current
 if engine.is_draft or state.enabled or current['real_N']!=24 or current['real_queries']!=120:
  raise RuntimeError('ROOT457 original real24 B4 diagnostic bank')
 if engine.config.max_num_seqs!=64:raise RuntimeError('ROOT457 unchanged global active ceiling64')
 if 80 not in engine.model._mc2_chained_row_counts:
  raise RuntimeError('ROOT457 constructor must naturally qualify the existing exact80 route')
 group=engine.model.layers[0].context.group
 if dist.get_world_size(group=group)!=3:raise RuntimeError('ROOT457 actual target TP3')
 output=Path(output)
 def save(name,value):
  with(output/f'rank{engine.rank}-{name}.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 def digest(value):
  return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 pointers=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]
 before={n:digest(v)for n,v in engine.model.named_parameters()};save('whole457-weights-before',before)
 manifest=engine.mc2_static_route_manifest;manifest_digest=manifest.digest
 x,p,m,_,_=current['physical'];xx=x[:80].contiguous();pp=p[:80].contiguous()
 bt=m.request_block_tables[:16].contiguous()
 mm=replace(m,slot_mapping=m.slot_mapping[:80].contiguous(),context_lens=m.context_lens[:80].contiguous(),
  block_tables=bt,request_block_tables=bt,actual_seq_lengths_q=tuple(range(5,81,5)),
  sequence_lens=(2560,)*16,tree_attention_mask=m.tree_attention_mask[:16].contiguous())
 if tuple(mm.tree_attention_mask.shape)!=(16,1,10,4096)or len(set(mm.slot_mapping.cpu().tolist()))!=80:
  raise RuntimeError('ROOT457 true80 rows and16 disjoint real owners')
 qslots={80:mm.slot_mapping,120:m.slot_mapping}
 def kv(count):
  index=qslots[count].long()
  return[cache.flatten(0,1).index_select(0,index).detach().cpu().contiguous()
   for layer in engine.model.layers for cache in(layer.self_attn.key_cache,layer.self_attn.value_cache)]
 def ids(h):return engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).detach().cpu().clone()
 def run_control80():
  with state.active(False):return unfused_forward(engine.model,xx,pp,mm).clone()
 def run_qualified80():
  with state.active(False):return engine.model(xx,pp,mm).clone()
 def run_current120():
  with state.active(False):return engine.model(x,p,m).clone()
 operations=dict(control80=run_control80,qualified80=run_qualified80,current120=run_current120)
 gold={};eager={}
 for name,operation in operations.items():
  h=operation();torch.npu.synchronize();cpu=h.detach().cpu().clone();caches=kv(120 if name=='current120'else 80);tokens=ids(h)
  gold[name]=(cpu,caches,tokens)
 h80,k80,t80=gold['qualified80'];hr,kr,tr=gold['control80'];h120,k120,t120=gold['current120']
 eager=dict(hidden_exact=bool(torch.equal(h80,hr)),tokens_exact=bool(torch.equal(t80,tr)),
  all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(k80,kr,strict=True)),
  hidden_max_abs=float((h80.float()-hr.float()).abs().max()),
  query_KV=[dict(index=i,equal=bool(torch.equal(a,b)),actual_sha256=digest(a),reference_sha256=digest(b),max_abs=float((a.float()-b.float()).abs().max()))for i,(a,b)in enumerate(zip(k80,kr,strict=True))])
 cross=dict(scope='same first16 request inputs; different physical M80 versus M120; observation only',
  hidden_exact=bool(torch.equal(h80,h120[:80])),hidden_max_abs=float((h80.float()-h120[:80].float()).abs().max()),
  greedy_equal=bool(torch.equal(t80,t120[:80])),different_greedy_rows=(t80!=t120[:80]).nonzero().flatten().tolist(),
  all128_query_KV_exact=all(torch.equal(a,b[:80])for a,b in zip(k80,k120,strict=True)))
 save('whole457-eager-before-capture',dict(own_M80=eager,cross_M120_observation=cross))
 gold['qualified80']=gold['control80']
 for operation in operations.values():
  for _ in range(3):operation()
 torch.npu.synchronize();graphs={};outputs={}
 for name,operation in operations.items():
  dist.barrier(group=group);graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph):h=operation()
  graphs[name]=graph;outputs[name]=h
  for _ in range(3):graph.replay()
  torch.npu.synchronize()
 engine._root457_whole_resources=(graphs,outputs,(x,p,m,xx,pp,mm),gold)
 keys=('hidden_exact','tokens_exact','all128_query_KV_exact')
 def check(name,*,replay=True):
  if replay:graphs[name].replay()
  torch.npu.synchronize();cpu=outputs[name].detach().cpu().clone();actual_kv=kv(120 if name=='current120'else 80)
  actual_ids=ids(outputs[name]);wanted,wkv,wids=gold[name]
  return dict(name=name,hidden_exact=bool(torch.equal(cpu,wanted)),tokens_exact=bool(torch.equal(actual_ids,wids)),
   all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(actual_kv,wkv,strict=True)),
   hidden_max_abs=float((cpu.float()-wanted.float()).abs().max()),actual_hidden_sha256=digest(cpu),reference_hidden_sha256=digest(wanted))
 initial={name:check(name)for name in operations}
 local_gate=dict(rank=engine.rank,exact=all(eager[k]is True for k in keys)and all(all(v[k]is True for k in keys)for v in initial.values()))
 gates=[None]*3;dist.all_gather_object(gates,local_gate,group=group)
 timings={name:[]for name in operations};post=[]
 if all(v['exact']is True for v in gates):
  for repeat in range(9):
   names=list(operations);rotation=repeat%3;names=names[rotation:]+names[:rotation]
   if repeat%2:names.reverse()
   for name in names:
    dist.barrier(group=group);begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);begin.record()
    for _ in range(20):graphs[name].replay()
    end.record();end.synchronize();elapsed=float(begin.elapsed_time(end))/20
    if not math.isfinite(elapsed)or elapsed<=0:raise RuntimeError('ROOT457 positive actual model event')
    timings[name].append(elapsed);post.append(dict(repeat=repeat,**check(name,replay=False)))
 final={name:check(name)for name in operations}
 after={n:digest(v)for n,v in engine.model.named_parameters()};save('whole457-weights-after',after)
 if before!=after or pointers!=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]or manifest.digest!=manifest_digest:
  raise RuntimeError('ROOT457 unchanged parameter bytes, addresses and original MC2 routes')
 local=dict(rank=engine.rank,eager=eager,cross_M120_observation=cross,initial_graph_checks=initial,actual_all3_pre_timing_gate=gates,
  timings_ms=timings,timing_skipped=None if all(v['exact']for v in gates)else'actual M80 same-geometry ordinary reference mismatch',
  post_timing_checks=post,final_graph_checks=final,
  all_graph_checks_exact=all(all(v[k]is True for k in keys)for v in[*initial.values(),*post,*final.values()]),
  physical_M80=80,physical_S80=16,real80_queries=80,real80_owners=list(range(16)),global_active_ceiling=64,
  actual_chained_rows=sorted(engine.model._mc2_chained_row_counts),actual_deferred_rows=sorted(engine.model._mc2_deferred_row_counts),actual_profile_sha256=manifest.profile_sha256,
  original_unfused_reference_uses_real_modules_and_weights=True,model_configuration_flags_modified=False,
  model_parameter_SHA_unchanged=True,mc2_manifest_unchanged_during_epoch=True,ordinary_prompt_prefill_unchanged=True,
  Native_flags_assigned=False,production_permission=False,own_AR_qualified=False,full463=False,Raw=None,Goodput=None)
 gathered=[None]*3;dist.all_gather_object(gathered,local,group=group);costs=None
 if all(v['timing_skipped']is None for v in gathered):
  samples={name:[max(v['timings_ms'][name][i]for v in gathered)for i in range(9)]for name in operations}
  median={name:statistics.median(values)for name,values in samples.items()}
  costs=dict(paired_max_rank_samples_ms=samples,p50_ms=median,
   qualified80_over_unfused80_speedup=median['control80']/median['qualified80'],
   current120_over_qualified80_frame_speedup=median['current120']/median['qualified80'],
   fully_occupied_request_rate80_over120=(16/median['qualified80'])/(24/median['current120']))
 report=dict(schema='ROOT457.own-geometry80-whole-model-reference-and-cost.v1',rank=engine.rank,by_target_rank=gathered,
  all3_eager_and_graph_math_exact=all(all(v['eager'][k]is True for k in keys)and v['all_graph_checks_exact']is True for v in gathered),
  actual_whole_model_costs=costs,original_cross_M120_discrepancy_not_erased=True,production_permission=False,
  whole_workload_AR_qualified=False,Native_flags_assigned=False,Raw=None,Goodput=None,adopted=False)
 save('whole457-final',report)
 return dict(report=str(output/f'rank{engine.rank}-whole457-final.json'),all_parameter_storages_unchanged=True,
  original_profile_unchanged=True,full463=False,Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
