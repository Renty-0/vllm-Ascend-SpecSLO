"""HOST geometry, source inverses and actual original231 logical superset."""
import copy
import json
from pathlib import Path
import sys
import host_paths383

P=Path(__file__).parent;B=P.parent


def main():
    from geometry383 import envelope,valid_record
    from physical377 import plan
    from bucket377 import extent
    from oracle377 import cases as new_cases
    from oracle257 import cases as old_cases
    new=new_cases();old=old_cases()
    key=lambda c:(c['N'],tuple(c['budgets']),c['anchor'])
    assert len(old)==227 and len(new)==231 and all(any(c['name']==o['name']and key(c)==key(o)for c in new)for o in old)
    fixture=json.loads((B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257/FIXTURE257.json').read_text())['prefixes']
    count={80:0,120:0}
    for case in new:
        budgets=case['budgets'];n=len(budgets);q=n+sum(budgets);branches=budgets.count(5)
        m,s,padding=envelope(n,q,'full9'if branches else'causal',budgets)
        sizes=[]
        for b in budgets:sizes.extend([5,1]if b==5 else[b+1])
        sizes.extend(padding);cursor=0;ends=[]
        for size in sizes:cursor+=size;ends.append(cursor)
        physical=plan(ends,q,m)
        assert sorted(physical['physical_to_original'])==list(range(m))
        assert sorted(t['old_row']for t in physical['real_moves'])==list(range(q))
        kv=extent(tuple(len(fixture[i])+min(b,4)+1 for i,b in enumerate(budgets)))
        row=dict(N=n,C=sum(budgets),branches=branches,M=m,attention_segments=m//5,
            primary_side_groups=n+branches,actual_q_partition=list(range(5,m+1,5)),fixed_KV_extent=[kv]*(m//5),
            uniform_KV_bucket377=dict(enabled=True,extent=kv,original_sequence_lens=[len(fixture[i])+min(b,4)+1 for i,b in enumerate(budgets)]))
        assert valid_record(row)
        damaged=copy.deepcopy(row);damaged['M']=120 if m==80 else 80
        assert valid_record(damaged)is False
        damaged=copy.deepcopy(row);damaged['fixed_KV_extent']=[512]*(m//5)
        assert valid_record(damaged)is False
        count[m]+=1
    import extension383,audit383,qualify383,runtime383
    extension383.projected_qualifier();audit383.projected_original_audit();qualify383.qualifier()
    projected,changes=runtime383.project_source((B/'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text())
    compile(projected,'HOST383.actual-runtime-source','exec')
    assert "report['ROOT383_uniform_KV_math']"in projected
    assert 'from scope383 import issue'in projected and 'from audit383 import audit'in projected
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    v=dict(status='PASS_HOST383_GEOMETRY_BIJECTIONS_SOURCE_INVERSES_AND_LOGICAL_SUPERSET',
        NPU_used=False,cases=231,old227_logical_subset_retained=True,physical_families=count,
        actual_reversible_worker_changes=len(changes),neural_qualification_inherited=False,production_qualified=False)
    (P/'HOST383_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))


if __name__=='__main__':main()
