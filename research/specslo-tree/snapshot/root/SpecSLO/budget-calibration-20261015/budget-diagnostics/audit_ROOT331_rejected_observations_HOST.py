"""Describe completed331 measurements without accepting its rejected run."""
import hashlib,json,math,statistics
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M160-FIA160-MC2projection165-root-v331-B1to8-532-math-cost-v1';A=O/'actual'/TAG
r=json.loads((O/f'receipt-{TAG}.json').read_text())
assert r['status']=='failed'and r['child']['exit_code']==-15 and r['owned_remaining']==[]
assert len(r['errors'])==2 and all('aggregate private bound'in e for e in r['errors'])
assert r['sources_unchanged']is True and r['post_verifier']['exit_code']==0 and r['devices_after']['all_idle']is True
reports=[json.loads((A/f'rank{rank}-numeric.json').read_text())for rank in range(4)]
costs=[];tokens=[]
for rank,report in enumerate(reports[1:],1):
 rows=report['math']['cases'];assert len(rows)==532
 assert all(all(row[k]is True for k in('hidden_bitwise_equal','KV_bitwise_equal','exact_greedy','graph_eager_KV_bitwise_equal'))for row in rows)
 tokens.append([row['candidate_tokens']for row in rows])
 assert report['math']['all_independent_prefixes_unchanged']is report['all_prefixes_still_exact_after_costs']is True
 assert len(report['fresh160_raw_operator']['checks'])==144
 panel=report['paired_target_cost'];assert len(panel['rows'])==16 and panel['no_new_capture_or_validation_in_costs']is True
 for row in panel['rows']:
  names=tuple(range(4,9))if row['branch_owners']else(4,)
  assert row['rounds']==len(row['pairs'])==20
  for i,pair in enumerate(row['pairs']):
   assert tuple(pair['order'])==names[i%len(names):]+names[:i%len(names)]
   assert all(type(v)in(float,int)and math.isfinite(v)and v>0 for v in pair['milliseconds'].values())
  costs.append(dict(rank=rank,N=row['N'],branch_owners=row['branch_owners'],
   median_ms={str(b):statistics.median(pair['milliseconds'][str(b)]for pair in row['pairs'])for b in names}))
assert tokens[0]==tokens[1]==tokens[2]
summary=[]
for row in costs[:16]:
 selected=[v for v in costs if(v['N'],v['branch_owners'])==(row['N'],row['branch_owners'])]
 summary.append(dict(N=row['N'],branch_owners=row['branch_owners'],
  median_across_target_rank_medians_ms={b:statistics.median(v['median_ms'][b]for v in selected)for b in row['median_ms']}))
result=dict(status='RETAINED_MEASUREMENTS_ORIGINAL_RUN_REJECTED_STORAGE_LIMIT',
 original_receipt=str(O/f'receipt-{TAG}.json'),original_receipt_sha256=hashlib.sha256((O/f'receipt-{TAG}.json').read_bytes()).hexdigest(),
 original_run_status_unchanged='failed',original_child_exit=-15,normal_worker_exit_not_qualified=True,
 source_prepost_verified=True,owned_cleanup_complete=True,all4_idle=True,
 completed_recorded_math_cases_each_target=532,all_recorded_hidden_KV_greedy_graph_eager_bitwise=True,
 recorded_target_panels_each_rank=16,rotations=20,cost_summary=summary,target_costs=costs,
 current_N32_B4_ms=summary[-1]['median_across_target_rank_medians_ms']['4'],
 source_qualified_shape165_with_legacy_nonchained_ABI=True,standalone_chained_flush_not_used=True,
 no_performance_adoption=True,no_external_calibration_input_to333=True,
 full463_own_AR_qualified=False,Goodput=None,adopted=False)
path=D/'ROOT331_REJECTED_RUN_RETAINED_COMPONENT_OBSERVATIONS.json'
with path.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in('target_costs','cost_summary')}))
