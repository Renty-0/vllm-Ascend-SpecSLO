"""Small readonly completed-V143 opportunity analysis, no suites or device imports."""
import collections,gzip,hashlib,importlib.abc,json,math,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
sys.dont_write_bytecode=True
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers'):raise ImportError('HOST opportunity analysis rejects '+fullname)
sys.meta_path.insert(0,Guard());P=B/'tree-high-normal-cap4-native-families-v148';sys.path.insert(0,str(P))
import target_layout

def desc(path):
 p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

def main():
 cost_path=D/'ROOT_V143_COMPLETED_FULL463_COST_SUMMARY.json';cost=json.loads(cost_path.read_text())
 artifact=cost['inputs']['artifacts'][1];raw=Path(artifact['path']);assert desc(raw)['sha256']==artifact['sha256']
 t=json.loads(gzip.decompress(raw.read_bytes()));calls=t['normal_target_calls'];wall=t['elapsed_seconds'];arrival_end=max(t['absolute_arrivals'])-t['origin']
 f=cost['draft_queries']['FastV5_counts'];p=cost['normal_budget']['policy_counts']
 ordinary=f['chain_calls']-f['root_seed_calls'];assert ordinary>=f['repair_calls'] and ordinary>=f['eager_frontier_calls']
 assert f.get('repair_teacher_forced_chains',0)==0
 cooccur_min=max(0,f['repair_PA_calls']+f['eager_frontier_calls']-ordinary);cooccur_max=min(f['repair_PA_calls'],f['eager_frontier_calls'])
 source_calls=f['chain_calls']+f['repair_PA_calls']+f['eager_frontier_calls']+f['root_seed_calls']
 scenario=collections.Counter();oldQ=newQ=0
 for c in calls:
  n,q=c['N'],c['real_query_count']
  if n>8 and c['family']=='causal' and q>=2*n+4:
   m,s,_=target_layout.envelope(n,q,'full9');scenario[(c['physical_M'],m,c['physical_S'],s)]+=1;oldQ+=c['physical_M'];newQ+=m
 uniform=[]
 for factor in [1.,1.1,1.2,1.25,1.5,1.8,2.]:
  attained=collections.Counter()
  for request,result in zip(t['requests'],t['generate_results'],strict=True):
   if result['paper_tpot_ms']/factor<=request['slo_tpot_ms']:attained[request['slo_class']]+=1
  uniform.append(dict(hypothetical_all_TPOT_speed_factor=factor,attained_by_class=dict(attained),total_attained=sum(attained.values())))
 gain={}
 for row in cost['observed_class_B_gain']:gain[row['slo_class'],row['B']]=row['g']
 old_gain=[dict(slo_class=c,g3=gain[c,3],g5=gain[c,5],observed_g5_over_g3=gain[c,5]/gain[c,3]) for c in ('tight','normal','loose')]
 r=dict(schema='HOST.V143-completed-next-optimization-opportunities.v1',status='completed-readonly-analysis',
 inputs=dict(cost=desc(cost_path),completed_leader_trace=desc(raw),frozen_V148_mapper=desc(P/'target_layout.py'),analysis=desc(__file__)),
 observed=dict(wall_seconds=wall,arrival_end_seconds=arrival_end,after_last_arrival_tail_seconds=wall-arrival_end,
 standalone_refill_host_seconds=cost['standalone_refill_seconds'],standalone_refill_fraction_of_wall=cost['standalone_refill_seconds']/wall,
 target_leader_host_seconds=cost['phase_seconds_by_rank'][1]['phase_seconds']['target'],draft_rank0_host_seconds=cost['phase_seconds_by_rank'][0]['phase_seconds']['draft'],
 target_intervals_not_additive_or_pure_device=True,per_family_target_wall_missing=True,
 highN_target_calls=sum(c['N']>8 for c in calls),highN_FULL_calls=sum(c['N']>8 and c['family']=='full9' for c in calls),
 actual_raw=cost['clocks']['raw_tokens_per_second'],actual_Native_Goodput=cost['clocks']['Native_paper']['Goodput'],actual_stream_Goodput=cost['clocks']['host_stream_paper']['Goodput']),
 new_fusion_information=dict(ordinary_FastV5_batches=ordinary,repair_batches=f['repair_PA_calls'],frontier_batches=f['eager_frontier_calls'],
 no_multitoken_repair_measured=f.get('repair_teacher_forced_chains',0)==0,
 intersection_lower_bound=cooccur_min,intersection_upper_bound=cooccur_max,
 proof='Both repair and frontier are at most one call in each of1440 ordinary FastV5 batches; root_seed121 is a separate path. Complete Counter snapshots/deltas have zero teacher-forced-multitoken repair increments. Inclusion-exclusion yields1376+1376-1440=1312.',
 current_model_and_graph_calls=source_calls,potential_graph_call_reduction_fraction=[cooccur_min/source_calls,cooccur_max/source_calls],
 attention_task_updates_potentially_removed=[28*cooccur_min,28*cooccur_max],
 observed_all_rank0_attention_tasks=cost['task_update_counts_by_rank'][0]['delta']['aclgraph_draft_task_update_tasks']+cost['task_update_counts_by_rank'][0]['delta']['aclgraph_target_task_update_tasks'],
 remaining_unknown='Exact per-batch R/E/U and per-family wall were not retained; this is a reachable-call-count bound, not saved milliseconds. Union can increase padding and moves frontier head outside its old graph.'),
 cap5_geometry_sensitivity=dict(condition='Keep old highN N/Q fixed and change only family causal→full9 where legal; not a replay of the cap5 allocator.',
 calls=sum(scenario.values()),old_physical_queries=oldQ,new_physical_queries=newQ,ratio=newQ/oldQ,
 shape_histogram=[dict(oldM=k[0],newM=k[1],oldS=k[2],newS=k[3],calls=v) for k,v in scenario.items()],
 exact_N32_Q128_causal=target_layout.envelope(32,128,'causal')[:2],exact_N32_Q128_FULL9=target_layout.envelope(32,128,'full9')[:2],
 interpretation='A single genuine B5 turns the whole target batch into FULL9, but N32/Q128 does not inherently require M165; this frozen mapper has FULL9 M128/S32. Dense masked attention and four-step draft costs still need measurement.'),
 adaptive_gain_descriptive_only=old_gain,uniform_TPOT_sensitivity_not_a_prediction=uniform,
 sensitivity_limit='The uniform table simply rescales retained request TPOT to show threshold distances; it changes neither real workload nor observed performance and is not a causal estimate of any optimization.',
 ranked_recommendations=[
 dict(rank=1,name='把同批single-token repair PA与frontier PA合成一次hidden模型调用',
 why_first='现有完整计数已证明至少1312批同时存在两种工作，避免了同M cache方案的可达性缺口；已有CPU帧设计，既不改变selectedB、g/need、到达、SLO，也不额外给所有行加第4步。',
 expected_opportunity='可减少1312–1376个原图/模型提交，占4434调用的29.6%–31.0%，以及36736–38528个draft attention task更新；这些是代码/计数上界，不是时延收益。',
 cost_risk='union U可能比旧Rbucket+Fbucket多padding，model M改变可能产生BF16输出差异；frontier原M=F的greedy head保持F但移到图外，有额外host dispatch。',
 HOST_boundary='现成帧原型与本次计数交集推导即可；无必要重跑旧套件。',
 NPU_next='ROOT空闲后只做小/端点两组新帧真实同M原matcher、精确原frontier real tokens/ABI/KV及原协议consume；完整draft含head/readback旧新交替计时。若输出资格或净成本不赢就停该候选，不盲跑full。'),
 dict(rank=2,name='大chunk mixed prefill以减少独占refill屏障，但先局部测量645形状',
 why='V143独占whole-prompt refill397次，真实host总50.657s（wall29.5%）；旧V140 mixed397?实际799次小mixed且rank0 join326.9s，说明小chunk的反复公共等待本身可能灾难，不能直接恢复旧mixed。',
 intended='已有大chunk设计将N32/C96/Q128+prompt512+scratch5放在M645/S35；同一2301 prompt理想5块代替65上限的36块，仅是组块算术，实际并发/合批数会变。',
 expected_opportunity='有机会同时隐藏target prefill、减少每chunk的world/Gloo/hash/epoch边界。50.657s不是可直接从wall减掉的可用收益；到达延续119.864s且末到达后还拖51.761s，必须看请求TPOT和排队。',
 blocker='M645全attention mask/真实query分段/多页dummy lease尚需本机数值资格；原MC2 profile未含645，不能把enable_mc2 flag当645真实融合成功。先明确合法常规route与真正645 profile的成本，不能伪造profile或复用197资格。',
 HOST_boundary='可读现成645源码和原prompt分布；不改当前V148/no-mixed控制。',
 NPU_next='先原197/65与新645/512在同真实prompt的首两块/最后P1做有限数值和整交易局部cost，保留all4公共边界但让测量边界区分device与诊断hash。若645平台不支持或净耗时不赢，保持standalone。'),
 dict(rank=3,name='等待cap4完整结果后再单变量放宽高N cap5，真正测试分叉收益',
 why_later='高N1413个target调用中仅4个FULL，现B5 g样本主要是低N/低负载，不足以预测高N。cap5会把更多整个batch切FULL9并使draft shared4，且原parent3-only/E-defer门继续损失E；优先获得cap4信息。',
 expected_opportunity='旧自适应样本g5/g3分别tight1.249/normal1.279/loose1.142，只作线索。V143 tight中位TPOT76.83ms对40ms，仅25%理想advance提升仍不够；单靠cap5不能据此预期达到644.77 Goodput。',
 no_fake_cost='保持同N/Q切FULL9的物理query总量只增加0.20%，所以不能凭M膨胀断言必慢；也不能忽略FULL9 attention代价。真实cap5会重新分配C96，更多B5意味着其他请求B1/B2/zero，与全体owner都变B5完全不同。',
 HOST_boundary='枚举真实C96/Q128变量B5向量及原mapper即可，禁止把旧B5 g当因果估计。',
 NPU_next='在明确N32/C96/Q128 genuine B5变量向量下比较causal cap4与FULL9 cap5的整draft/target成本和实际有效输出；保持原MC2/numerics/target预算、arrival/SLO。只有单位时间有效advance与真实SLO方向都更好再跑完整463。')],
 forbidden_backend_modules_loaded=[name for name in sys.modules if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend')],
 current_V148_raw_or_receipt_or_fusion_read=False,candidate_changed=False,existing_suite_rerun=False,NPU_or_model_queried=False,results_are_not_new_NPU_measurements=True)
 # Correct the textual old mixed count directly from its retained audited summary.
 r['ranked_recommendations'][1]['why']=r['ranked_recommendations'][1]['why'].replace('旧V140 mixed397?实际799次','旧V140 mixed实际799次')
 out=D/'NEXT_OPTIMIZATION_PRIORITIES_FROM_COMPLETED_V143_HOST.json'
 with out.open('x') as stream:json.dump(r,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(output=str(out),sha256=desc(out)['sha256'],fusion_cooccur=[cooccur_min,cooccur_max],forced_full_query_ratio=newQ/oldQ,forbidden_loaded=r['forbidden_backend_modules_loaded'])))
if __name__=='__main__':main()
