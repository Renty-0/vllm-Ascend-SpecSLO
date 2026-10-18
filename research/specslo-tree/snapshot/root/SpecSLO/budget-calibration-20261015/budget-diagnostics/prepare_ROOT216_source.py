"""Extend215 budget-cost repair only to current-qualified real N9..16."""
import ast,difflib,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
DONOR=B/'tree-cost-aware-mixed16-with-own-FIA-AR-full-v215'
P=B/'tree-cost-aware-N9to16-with-own-FIA-AR-full-v216'
assert not P.exists();P.mkdir()
records=[]
for old in sorted(DONOR.iterdir()):
    if old.suffix not in('.py','.sh'):continue
    new=P/old.name.replace('215','216')
    new.write_text(old.read_text().replace(str(DONOR),str(P)).replace('215','216'))
    records.append(dict(source=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new=str(new)))


def change(name,old,new):
    p=P/name;s=p.read_text();assert s.count(old)==1,(name,old);p.write_text(s.replace(old,new))


change('cost216.py','def repair_plan(policy, plan, *, context_len, counts):', '''_COST_TABLE=None


def original_cost_table():
    global _COST_TABLE
    if _COST_TABLE is None:
        from mapper173 import expected_geometry
        values={}
        for n in range(9,17):
            for q in range(4*n+1,5*n):
                m,s,_=expected_geometry(n,q,'causal')
                if m not in(96,128) or m<=80:
                    raise RuntimeError('ROOT216 exact original mixed3/4 geometry changed')
                values[n,q]=(m,s)
        _COST_TABLE=values
    return _COST_TABLE


def repair_plan(policy, plan, *, context_len, counts):''')
change('cost216.py',"    if (len(plan.requested_normal) != 16 or len(normal) != 16\n", "    n=len(plan.requested_normal)\n    if (not 9<=n<=16 or len(normal) != n\n")
change('cost216.py',"    counts['eligible_mixed16'] += 1", "    counts['eligible_mixed_N9to16'] += 1\n    old_m,old_s=original_cost_table()[n,n+sum(normal.values())]")
change('cost216.py','and 64 <= min(plan.work_limit.normal_candidates, plan.target_limit.candidates)',
       'and 4*n <= min(plan.work_limit.normal_candidates, plan.target_limit.candidates)')
change('cost216.py','and 16 <= plan.target_limit.rows and 80 <= plan.target_limit.queries',
       'and n <= plan.target_limit.rows and 5*n <= plan.target_limit.queries')
change('cost216.py',"    counts['proposed_target_physical_queries_saved'] += 48",
       "    counts['proposed_target_physical_queries_saved'] += old_m-80\n    counts[f'repaired_old_M{old_m}_calls'] += 1\n    counts[f'repaired_N{n}_calls'] += 1")
change('cost216.py',"    state = dict(private_shape_bindings=0, by_generation={})", "    table=original_cost_table()  # Exact original mapper geometry prepared once outside model work.\n    state = dict(private_shape_bindings=0, by_generation={},original_cost_table=[dict(N=n,Q=q,M=m,S=s)for(n,q),(m,s)in sorted(table.items())])")
change('cost216.py',"row.get('eligible_mixed16', 0)","row.get('eligible_mixed_N9to16', 0)")
change('cost216.py',"48 * row.get('repaired_calls', 0)","16 * row.get('repaired_old_M96_calls', 0)+48 * row.get('repaired_old_M128_calls', 0)")
change('cost216.py',"    for row in state['by_generation'].values():", "    expected=[dict(N=n,Q=q,M=m,S=s)for(n,q),(m,s)in sorted(original_cost_table().items())]\n    if state['original_cost_table']!=expected:\n        raise RuntimeError('ROOT216 actual precomputed original cost table differs')\n    for row in state['by_generation'].values():")
change('cost216.py',"or row.get('added_selected_candidates', 0) != row.get('upgraded_owners', 0)",
       "or row.get('added_selected_candidates', 0) != row.get('upgraded_owners', 0)\n                or row.get('repaired_calls',0)!=sum(row.get('repaired_N'+str(n)+'_calls',0)for n in range(9,17))\n                or row.get('repaired_calls',0)!=row.get('repaired_old_M96_calls',0)+row.get('repaired_old_M128_calls',0)")

host=P/'host216.py';s=host.read_text()
s=s.replace("counts['eligible_mixed16']", "counts['eligible_mixed_N9to16']")
s=s.replace("([3]*16,[4]*16,[3,4]*7,[3,4]*8+[3], [2]+[4]*15)","([3]*16,[4]*16,[3,4]*4,[3,4]*8+[3], [2]+[4]*15)")
s=s.replace("eligible_mixed16=4,repaired_calls=3,blocked_by_original_limits=1,upgraded_owners=8,added_selected_candidates=8,proposed_target_physical_queries_saved=144",
            "eligible_mixed_N9to16=4,repaired_calls=3,repaired_N16_calls=3,repaired_old_M128_calls=3,blocked_by_original_limits=1,upgraded_owners=8,added_selected_candidates=8,proposed_target_physical_queries_saved=144")
s=s.replace("from cost216 import audit_snapshot", "from cost216 import audit_snapshot,original_cost_table")
s=s.replace("        self.assertTrue(audit_snapshot(cache,5))", "        cache['cost_aware_mixed16_budget']['original_cost_table']=[dict(N=n,Q=q,M=m,S=ss)for(n,q),(m,ss)in sorted(original_cost_table().items())]\n        self.assertTrue(audit_snapshot(cache,5))")
s=s.replace("private_shape_bindings=1,by_generation={'5':dict(shape_calls=1)}", "private_shape_bindings=1,by_generation={'5':dict(shape_calls=1)},original_cost_table=[dict(N=n,Q=q,M=m,S=ss)for(n,q),(m,ss)in sorted(__import__('cost216').original_cost_table().items())])")
anchor='    def test_actual_mapper_cost_discontinuity(self):';assert s.count(anchor)==1
s=s.replace(anchor,'''    def test_all_current_qualified_sizes_and_both_real_cost_cliffs(self):
        from collections import Counter
        from cost216 import repair_plan,original_cost_table
        for n in range(9,17):
            for fours in range(1,n):
                budgets=dict(enumerate([4]*fours+[3]*(n-fours)))
                policy,plan=self.plan(budgets);counts=Counter()
                new=repair_plan(policy,plan,context_len=4092,counts=counts)
                self.assertEqual(list(new.normal_budgets.values()),[4]*n)
                old_m,_=original_cost_table()[n,4*n+fours]
                self.assertEqual(counts['proposed_target_physical_queries_saved'],old_m-80)
                self.assertEqual(policy.counts['grant_B3'],0);self.assertEqual(policy.counts['grant_B4'],n)
        self.assertEqual(original_cost_table()[9,37][0],96)
        self.assertEqual(original_cost_table()[9,44][0],128)

'''+anchor)
s=s.replace("sorted(__import__('cost216').original_cost_table().items())]))),","sorted(__import__('cost216').original_cost_table().items())])),")
host.write_text(s)
for name in ['extension216.py','guard216.py','host216.py']:
    p=P/name;s=p.read_text().replace('cached-parent-SLO-admission1-cost-aware-mixed16','cached-parent-SLO-admission1-cost-aware-N9to16');p.write_text(s)
change('guard216.py','mixed16 B3/B4 to allB4','N9to16 mixed B3/B4 to allB4')
for item in records:
    file=Path(item['new']);file.write_text(file.read_text().replace('cost_aware_mixed16_budget','cost_aware_N9to16_budget'))
for r in records:
    old=Path(r['source']).read_text().replace(str(DONOR),str(P)).replace('215','216');new=Path(r['new']).read_text()
    r['new_sha256']=hashlib.sha256(new.encode()).hexdigest();r['patch_after_namespace_projection']=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
    if Path(r['new']).suffix=='.py':ast.parse(new)
(P/'PROJECTION216.json').write_text(json.dumps(dict(schema='ROOT216.cost-aware-N9to16-source-projection.v1',
    source_donor215_freeze_sha256=hashlib.sha256((DONOR/'FREEZE215.json').read_bytes()).hexdigest(),
    projection=records,production_delta='extend existing mixed3/4 repair from16 to own-qualified9..16; original mapper cost table built once before model work',
    postfull_AR_unchanged=True,new_Native_graphs=0,new_math_keys=0,NPU_started=False,
    source_peer_review=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='source_prepared_unfrozen',directory=str(P),NPU_started=False)))
