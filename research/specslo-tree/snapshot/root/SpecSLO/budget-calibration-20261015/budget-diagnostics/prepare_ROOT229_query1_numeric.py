"""Complete q1 observations, isolate M/Q effects eagerly, time only existing q1 graph."""
import ast,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-query1-FULL-vs-FULL9-numeric-cost-diagnostic-v228';P=B/'tree-query1-FULL-numeric-isolation-diagnostic-v229'
P.mkdir(exist_ok=False)
for p in OLD.iterdir():
    if p.suffix in('.py','.sh'):(P/p.name.replace('228','229')).write_text(p.read_text().replace(str(OLD),str(P)).replace('228','229'))
(P/'FIXTURE229.json').write_bytes((OLD/'FIXTURE228.json').read_bytes())
def sub(s,a,b):
    if s.count(a)!=1:raise RuntimeError('single source anchor '+a[:90])
    return s.replace(a,b)
p=P/'fresh229.py';s=p.read_text().replace('def prepare_frame(engine,case,frame):','def prepare_frame(engine,case,frame,*,fixed_M=True,split_queries=True):')
s=sub(s,'    frame=dict(frame,M=80,S=16) # same physical body M for all bounded fresh variants','    if fixed_M:frame=dict(frame,M=80,S=16) # default main diagnostic uses canonical M80')
s=sub(s,'    meta=canonical(meta)','    if split_queries:meta=canonical(meta)')
s=sub(s,"len(meta.actual_seq_lengths_q)==frame['M']","len(meta.actual_seq_lengths_q)==(frame['M']if split_queries else frame['S'])")
p.write_text(s)
p=P/'probes229.py';s=p.read_text();s=sub(s,'    history=run_history',"    from ablation229 import run as run_ablation\n    ablations=run_ablation(engine,native,stage,output,fixture['fresh'])\n    save(output,f'rank{engine.rank}-ablations-complete.json',ablations)\n    history=run_history")
s=sub(s,"cases=[dict(kind='fresh',**c)","ablations=ablations,cases=[dict(kind='fresh',**c)")
p.write_text(s)
p=P/'guard229.py';s=p.read_text()
s=sub(s,"len(cost['samples'])==12","len(cost['samples'])==6 and cost['zero_new_body_graph_captures']is True")
s=sub(s,"            tokens.append([[r['diagnostic_refined_token']for r in rows]for rows in[fresh,*hist]])", "            ablations=report['ablations']\n            require(len(ablations)==2 and all(len(a['variants'])==6 for a in ablations),'two original contexts with six eager-only M/Q controls')\n            tokens.append(dict(main=[[r['diagnostic_refined_token']for r in rows]for rows in[fresh,*hist]],ablations=[[v['token']for v in a['variants']]for a in ablations]))")
s=s.replace('isolated_dynamic_body_cost_only=True','isolated_existing_q1_body_cost_only=True').replace('dynamic body replay ABBA cost','existing q1 body replay cost and eager M/Q ablations')
p.write_text(s)
(P/'ROOT_SOURCE_REVIEW229.json').write_text(json.dumps(dict(status='source_only_unqualified',main_four_contexts_unchanged=True,
    ablation='six eager-only variants separate model rows M40/M80 from query partition q5/q1; no new q5 graph captured',
    cost='absolute existing q1 graph only; prebuilt adjacent contexts; require zero additional captures and unchanged Native gates',
    source228_failed_cost_preserved=True,original_sampler_unchanged=True,production_installed=False,performance_qualified=False,adopted=False),indent=2)+'\n')
for p in P.glob('*.py'):ast.parse(p.read_text())
print(json.dumps(dict(directory=str(P),prepared=True,NPU_started=False)))
