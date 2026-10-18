"""Independently reconstruct synthetic paired costs and verify frozen inputs."""
import hashlib,json,math,statistics
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def common(p,num,tag,name):
 root=p/'actual'/tag;receipt=json.loads((root/'receipt.json').read_text())
 assert receipt['status']=='complete' and receipt['errors']==[] and receipt['exit_code']==0
 assert receipt['owned_remaining']==[] and receipt['sources_unchanged']is True
 assert receipt['devices_before']['all_idle']and receipt['devices_after']['all_idle']
 assert receipt['freeze_sha256']==sha(p/f'FREEZE{num}.json')
 assert receipt['archive_sha256']==sha(p/f'SOURCE{num}.tar.gz')
 for path,wanted in json.loads((p/f'FREEZE{num}.json').read_text())['files'].items():
  assert sha(path)==wanted,path
 for phase in ('pre_verifier','post_verifier'):
  r=receipt[phase];assert r['sha256']==sha(r['path'])
  v=r['actual_result'];assert v['status']=='pass'and v['errors']==[]
  assert (v['checked_frozen_files'],v['checked_external_files'],v['checked_packages'])==(10496,15,212)
 path=root/name;assert receipt['paired_cost_sha256']==sha(path)
 m=json.loads(path.read_text());assert m['status']=='complete' and m['models_loaded']is False
 assert m['Goodput']is None and m['neural_qualified']is False and m['production_installed']is False
 return m,dict(receipt=str(root/'receipt.json'),receipt_sha256=sha(root/'receipt.json'),
              measured_source=str(path),measured_sha256=sha(path),
              actual_guard_complete=True,all_frozen_sources_unchanged=True,
              original10496_15_212_pre_and_post_passed=True,all_four_idle=True,
              neural_qualified=False,Goodput=None)
def main():
 p=B/'tree-FIA-fixed-q5-mask-capacity-cost-v264'
 m,result=common(p,264,'fixed-q5-FIA-root-v264-mask-capacity-cost-v1','paired-FIA.json')
 caps=[5,8,9,10,11,12,16];assert len(m['rows'])==6
 summaries=[]
 for row in m['rows']:
  assert row['physical_M']==120 and row['actual_q_partition']==list(range(5,121,5))
  assert row['sequence_lens']==[2560]*24 and row['all64_outputs_each_cap_equal_eager']is True
  assert len(row['pairs'])==80 and all(row['bitwise_equal_to_cap10'].values())
  for j,pair in enumerate(row['pairs']):
   assert pair['order']==caps[j%7:]+caps[:j%7]
   assert set(pair['milliseconds_for64_FIA_calls'])==set(map(str,caps))
  for cap in caps:
   v=[p['milliseconds_for64_FIA_calls'][str(cap)]for p in row['pairs']]
   assert all(math.isfinite(t)and t>0 for t in v)
   assert statistics.median(v)==row['median_graph_ms'][str(cap)]
   assert statistics.median(v)/64==row['median_per_FIA_ms'][str(cap)]
  summaries.append(dict(prefix=row['prefix'],layout=row['layout'],
      per_FIA_ms=row['median_per_FIA_ms'],all_caps_bitwise_equal=True))
 result.update(status='independent_synthetic_FIA_cost_audit_passed',rows=summaries,
   arms=42,paired_rounds_per_configuration=80,
   conclusion='No consistent large cap10/cap11 penalty; preserve original model graph capacities')
 with(D/'ROOT264_INDEPENDENT_SYNTHETIC_FIA_COST_AUDIT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 p=B/'tree-native-rank3-KV-copy-cost-v265'
 m,result=common(p,265,'native-rank3-KV-root-v265-three-arm-copy-v1','paired-copy.json')
 assert m['view2d_alias_without_copy']is True and len(m['map_cases'])==344
 assert len({(r['width'],r['case'])for r in m['map_cases']})==344
 for row in m['map_cases']:
  assert row['actual_original_native_map']and row['all_full_cache_values_equal']and row['actual_ACLGraph_replayed']
  assert row['HOST_contract']['pairs']==1792 and row['HOST_contract']['rows']==69632
  assert row['HOST_contract']['in_place_snapshot_copy_equivalence']is True
 assert len(m['raw_bit_replays'])==6 and len(m['eager_controls'])==2
 for row in m['raw_bit_replays']:
  assert all(row[k]is True for k in ('all65536_raw16_bit_patterns','complete_K_and_V_bitwise_equal',
                                   'changed_and_restored_same_buffers','actual_ACLGraph_replayed'))
 arms=['original','view2d','fused'];summaries=[]
 assert len(m['timings'])==10
 for row in m['timings']:
  assert row['paired_calls']==80 and len(row['pairs'])==80
  for j,pair in enumerate(row['pairs']):assert pair['order']==arms[j%3:]+arms[:j%3]
  for arm in arms:
   v=[p['milliseconds'][arm]for p in row['pairs']]
   assert all(math.isfinite(t)and t>0 for t in v)
   assert statistics.median(v)==row['median_ms'][arm]
  assert row['median_ms']['fused']<row['median_ms']['original']
  summaries.append({k:v for k,v in row.items()if k!='pairs'})
 result.update(status='independent_actual_rank3_three_arm_KV_copy_audit_passed',
  complete_cache_comparisons=344,raw_bit_changed_restored_replays=6,
  paired_rounds_per_case=80,timings=summaries,
  conclusion='Fused copy wins these synthetic paired cases; full same-instance math and all463 ownAR still required')
 with(D/'ROOT265_INDEPENDENT_RANK3_THREE_ARM_COPY_AUDIT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print('ROOT264 and ROOT265 independent audits passed')
if __name__=='__main__':main()
