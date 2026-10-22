"""Offline physical-shape counterfactual using actual V96 call geometries.

No alternate token predictions or speedups are inferred. Reducing a budget
would require a new device run; this only executes the unchanged HOST mapper.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

BASE=Path('/root/SpecSLO/budget-calibration-20261015')
SOURCE=BASE/'tree-capacity64-runtime-v96/target_layout.py'
TRACE=BASE/'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1/rank1-full463-actual-trace.json.gz'
OUT=BASE/'budget-diagnostics/V96_ACTUAL_PHYSICAL_BUDGET_CLIFFS.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('unchanged_V96_actual_layout',SOURCE)
    layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
    trace=json.loads(gzip.decompress(TRACE.read_bytes()))
    groups=Counter();examples=[];improved=Counter();total=Counter()
    for ordinal,call in enumerate(trace['normal_target_calls']):
        n,q,family=call['N'],call['real_query_count'],call['family']
        original=layout.envelope(n,q,family)
        assert original[0]==call['physical_M'] and original[1]==call['physical_S']
        assert sum(call['budgets'])+n==q and len(call['budgets'])==n
        groups[n,q,family,original[0],original[1]]+=1
        total['calls']+=1;total['physical_rows']+=original[0];total['real_rows']+=q
        # Keep at least one genuine B5 branch for a full9 call. Feasible B1..5
        # sums are contiguous, so a lower q in this range has an allocation;
        # the actual request to trim and its g loss remain unmeasured.
        floor=2*n+(4 if family=='full9' else 0)
        alternative=None
        for newq in range(q-1,floor-1,-1):
            new=layout.envelope(n,newq,family)
            if new[0]<original[0]:
                alternative=(newq,new);break
        if alternative:
            newq,new=alternative
            improved['calls']+=1;improved['real_candidates_removed']+=q-newq
            improved['physical_rows_removed']+=original[0]-new[0]
            improved[f'candidates_removed_{q-newq}']+=1
            if len(examples)<20:
                examples.append(dict(actual_call=ordinal,N=n,family=family,actual_budgets=call['budgets'],
                    actual_real_Q=q,actual_M=original[0],actual_S=original[1],
                    hypothetical_real_Q=newq,hypothetical_M=new[0],hypothetical_S=new[1],
                    candidates_removed=q-newq,genuine_B5_branch_retained=family=='full9'))
    result=dict(schema='ROOT.V96.actual-geometry-budget-cliff-counterfactual.v1',
        UTC=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in
            (Path(__file__),SOURCE,SOURCE.parent/'BOUNDED_BUCKET_WHITELIST.json',TRACE)},
        original_mapper_matches_every_actual_call=True,actual=total,
        potential_lower_row_geometries=improved,examples=examples,
        common_actual_shapes=[dict(N=n,Q=q,family=f,M=m,S=s,calls=count)
            for (n,q,f,m,s),count in groups.most_common(20)],
        alternate_device_execution=False,counterfactual_g_or_token_loss_estimated=False,
        causal_performance_or_Goodput_gain_proved=False,production_policy_modified=False,
        model_or_NPU_initialized=False)
    with OUT.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('actual','potential_lower_row_geometries','common_actual_shapes')}))


if __name__=='__main__':main()
