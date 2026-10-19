"""Exact inherited cases and independent M165/N32 reference coverage; no NPU."""
import ast
import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"]="0"
import json
import sys
from pathlib import Path
P=Path(__file__).resolve().parent
B=P.parent

def main():
    for folder in ['tree-M165-single-prepare-host-v313','tree-uniform-q5-M165-S33-C128-host-v285',
                   'tree-single-validated-prefix-descriptors-host-v304','tree-single-prepare-target-math-cost-v311',
                   'tree-alternate-chain-target-host-v269','tree-static-q5-M120-node-math-v255',
                   'tree-canonical-contiguous-full463-own-AR-v235','tree-contiguous-branch-tail-final-metadata-math-v234']:
        sys.path.append(str(B/folder))
    import check313_HOST
    from fixture313_HOST import build
    env=build()
    from oracle317 import cases,prepare,configurations
    from oracle311 import cases as oldcases
    from padding285 import pad
    from reference_schedule317 import schedule
    fixture=json.loads((P/'FIXTURE317.json').read_text())
    old_fixture=json.loads((B/'tree-single-prepare-target-math-cost-v311/FIXTURE311.json').read_text())
    prefixes=fixture['prefixes']
    assert prefixes[:24]==old_fixture['prefixes'] and prefixes[24:]==prefixes[:8] and len(prefixes)==32
    declared=cases()
    assert len({c['name']for c in declared})==len(declared)
    for old,new in zip(oldcases(),declared[:408],strict=True):
        assert all(new[k]==v for k,v in old.items() if k not in ('M','S','padding'))
    nodes=frames_total=0
    for case in declared:
        n=case['N'];budgets=case['budgets']
        assert n<=32 and sum(budgets)<=128 and n+sum(b>4 for b in budgets)<=33 and sum(b>4 for b in budgets)<=12
        plans,rows=prepare(case,prefixes)
        roots=list(range(41,41+n))
        pp,rr,dd,ss,real=pad(env['engine'],plans,roots,rows,list(range(n)),[],[])
        assert real==n and len(pp)==case['S'] and sum(x.candidate_budget+1 for x in pp)==165
        assert pp[:n]==plans and rr[:n]==roots and dd[:n]==rows and ss[:n]==list(range(n))
        frames=schedule(plans,roots,rows)
        labels=[v['candidate_query']for f in frames for v in f['real_queries']if v['candidate_query'] is not None]
        expected={(i,q)for i,b in enumerate(budgets)for q in range(b+1)}
        assert len(labels)==len(set(labels)) and set(labels)==expected
        for frame in frames:
            assert len(frame['tokens'])==len(frame['ids'])==len(frame['positions'])==165
            assert len(frame['selected'])==len(set(frame['selected']))==33
            assert frame['actual_seq_lengths_q']==list(range(5,166,5))
            assert all(32<=v['reference_owner']<96 for v in frame['real_queries'])
            for v in frame['real_queries']:
                assert frame['ids'][v['head_row']]==v['reference_owner']
        assert all(int(plan.positions.max())<2560 for plan in plans)
        nodes+=len(labels);frames_total+=len(frames)
    assert len(configurations())==15
    assert any(c['N']==32 and c['budgets']==[4]*32 for c in declared)
    assert 'torch_npu' not in sys.modules
    for file in P.glob('*.py'):ast.parse(file.read_text(),filename=str(file))
    (P/'DECLARED317_CASES.json').write_text(json.dumps(dict(cases=declared),indent=2)+'\n')
    result=dict(passed=True,cases=len(declared),unchanged_legacy408_cases=True,real_nodes=nodes,
                independent_reference_frames=frames_total,independent_owner_offsets=[32,64],
                physical_reference_groups=33,physical_reference_rows=165,prefixes=32,
                independently_prefilled_owners_planned=96,paired_target_cost_configurations=16,
                NPU_used=False,neural_qualified=False,Goodput=None)
    with (P/'HOST317_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
