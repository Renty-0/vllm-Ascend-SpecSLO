"""Independent terminal evidence/cost audit for ordinary160 and MC2 bridge165."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
RUNS={
    '329':('tree-M160-C128-capacity-binding-v329','guard329',329,
           'M160-S32-C128-root-v329-B1to8-532-math-cost-v1'),
    '331':('tree-FIA160-projection165-MC2-audit-adapter-v332','guard332',332,
           'M160-FIA160-MC2projection165-root-v331-B1to8-532-math-cost-v1'),
}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',choices=RUNS,required=True)
    args=parser.parse_args()
    folder,module,version,tag=RUNS[args.run]
    p=B/folder;a=O/'actual'/tag;receipt_path=O/f'receipt-{tag}.json'
    receipt=read(receipt_path)
    assert receipt['status']=='complete' and not receipt['errors']
    assert receipt['sources_unchanged']is True and receipt['owned_remaining']==[]
    assert receipt['devices_after']['all_idle']is True and receipt['child']['exit_code']==0
    sys.path.insert(0,str(p));guard=importlib.import_module(module)
    guard.PIN=(p/f'PIN{version}.txt').read_text().strip();prepared=guard.prepared()
    actual,verdict,flags=guard.actual_audit(a)
    assert actual==receipt['actual_component_audit'] and verdict=='diagnostic_only'
    reports=[read(a/f'rank{r}-numeric.json')for r in range(4)]
    expected=read(B/'tree-M160-S32-N32-target-math-cost-v326/DECLARED326_CASES.json')['cases']
    assert len(expected)==532
    tokens=[];costs=[]
    configurations=[]
    for n in(4,8,12,16,20,24,28,31):
        for forks in sorted({1,min(n,32-n,(128-4*n)//4,12)}):configurations.append((n,forks))
    configurations.append((32,0));assert len(configurations)==16
    for rank,report in enumerate(reports[1:],1):
        rows=report['math']['cases']
        assert [r['case']for r in rows]==expected
        assert all(r['hidden_bitwise_equal']and r['KV_bitwise_equal']and r['exact_greedy']
                   and r['graph_eager_KV_bitwise_equal']and r['all128_KV_original_matcher_passed']
                   and r['candidate_KV_copied']is False for r in rows)
        tokens.append([r['candidate_tokens']for r in rows])
        assert report['math']['all_independent_prefixes_unchanged']is True
        assert report['all_prefixes_still_exact_after_costs']is True
        raw=report['fresh160_raw_operator']
        assert len(raw['checks'])==144 and raw['raw16_bit_coverage']=={'256':65536,'384':65536}
        assert {r['slots_dtype']for r in raw['checks']}=={'torch.int32','torch.int64'}
        assert all(r['reference_slots_dtype']=='torch.int32'for r in raw['checks'])
        panel=report['paired_target_cost']
        assert panel['no_new_capture_or_validation_in_costs']is True
        assert [(r['N'],r['branch_owners'])for r in panel['rows']]==configurations
        for row in panel['rows']:
            names=tuple(range(4,9))if row['branch_owners']else(4,)
            assert row['rounds']==20 and len(row['pairs'])==20
            values={b:[]for b in names}
            for i,pair in enumerate(row['pairs']):
                assert tuple(pair['order'])==names[i%len(names):]+names[:i%len(names)]
                assert set(pair['milliseconds'])=={str(b)for b in names}
                for budget in names:
                    value=pair['milliseconds'][str(budget)]
                    assert type(value)in(float,int)and math.isfinite(value)and value>0
                    values[budget].append(value)
            costs.append(dict(rank=rank,N=row['N'],branch_owners=row['branch_owners'],
                rounds=20,median_ms={str(b):statistics.median(v)for b,v in values.items()},
                within_pair_ratio_to_B4={str(b):statistics.median(pair['milliseconds'][str(b)]/pair['milliseconds']['4']for pair in row['pairs'])for b in names if b!=4}))
    assert tokens[0]==tokens[1]==tokens[2]
    for name in('pre_verifier','post_verifier'):
        value=receipt[name];assert value['exit_code']==0 and sha(value['path'])==value['sha256']
        check=value['actual_result']
        assert check['status']=='pass' and check['errors']==[]
        assert(check['checked_frozen_files'],check['checked_external_files'],check['checked_packages'])==(10496,15,212)
    summary=[]
    for n,forks in configurations:
        data=[row for row in costs if(row['N'],row['branch_owners'])==(n,forks)]
        budgets=data[0]['median_ms']
        summary.append(dict(N=n,branch_owners=forks,
            median_across_target_rank_medians_ms={b:statistics.median(row['median_ms'][b]for row in data)for b in budgets}))
    result=dict(status=f'PASS_actual{args.run}_target532_all3_bitwise_rawKV_hidden_greedy_rotated_cost',
        receipt=str(receipt_path),receipt_sha256=sha(receipt_path),
        freeze_sha256=guard.PIN,target_cases_per_rank=532,target_ranks=[1,2,3],
        all_original408_and_new124_executed=True,all_target_greedy_outputs_equal=True,
        all_hidden_all128KV_and_graph_eager_bitwise=True,
        raw_operator_checks_per_target=144,both_candidate_slot_dtypes=True,
        independent_committed_prefixes96_unchanged=True,
        paired_cost_configurations_per_rank=16,paired_rotations=20,
        target_costs=costs,cost_summary=summary,
        physical_FIA_rows160=True,physical_FIA_groups32=True,
        actual_projection_MC2_rows=165 if args.run=='331'else None,
        profile_M160_MC2_qualified=False,
        original_environment_prepost_10496_15_212_verified=True,
        owned_cleanup_complete=True,all4_idle_at_terminal=True,
        full463_own_AR_qualified=False,real_draft_cost_measured=False,
        end_to_end_performance_qualified=False,Goodput=None,adopted=False)
    output=D/f'ROOT{args.run}_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json'
    with output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('target_costs','cost_summary')},ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
