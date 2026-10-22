"""Additive same444 diagnostic, retaining graph input frames explicitly."""
from pathlib import Path
import re,json
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-M120-padding-prefill-math-diagnostic-v444'
P=B/'tree-M120-owned-padding-prefill-math-diagnostic-v445'
assert not P.exists();P.mkdir()
def rename(t):
    return re.sub(r'[a-f0-9]{64}|(?<!\d)444(?!\d)',lambda m:'445'if m.group()=='444'else m.group(),t).replace('tree-M120-padding-prefill-math-diagnostic-v445',P.name)
t=rename((OLD/'host_env444.py').read_text())
t=t.replace("str(Path(__file__).resolve().parent)+':'+", "str(Path(__file__).resolve().parent)+':'+str(B/'tree-M120-padding-prefill-math-diagnostic-v444')+':'+")
(P/'host_env445.py').write_text(t)
t=rename((OLD/'check444.py').read_text()).replace('check385,check441','check385,check444').replace("check441.ready(result['current441_freeze'])","check444.ready(result['current444_freeze'])")
(P/'check445.py').write_text(t)
t=rename((OLD/'guard444.py').read_text()).replace('import guard441 as current','import guard444 as current').replace("'SOURCE441_RECEIPT.json'","'SOURCE444_RECEIPT.json'")
t=t.replace('M120-empty-q5-groups-prefill-math-and-cost','M120-owned-q5-prefill-frames-math-and-cost')
(P/'guard445.py').write_text(t)
t=rename((OLD/'startup444.sh').read_text()).replace('M120-empty-q5-groups-prefill-math-and-cost','M120-owned-q5-prefill-frames-math-and-cost')
t=t.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
(P/'startup445.sh').write_text(t)
t=rename((OLD/'runtime444.py').read_text())
t=t.replace('import runtime441 as original','import runtime444 as original')
t=t.replace('from check441 import ready','from check444 import ready').replace('from probe441 import run','from probe444 import run')
t=t.replace("('M120-all64-gateup-NZ-whole-target-cost',ACTION,2)","('M120-empty-q5-groups-prefill-math-and-cost',ACTION,2)")
t=t.replace("ACTION='M120-empty-q5-groups-prefill-math-and-cost'","ACTION='M120-owned-q5-prefill-frames-math-and-cost'")
(P/'runtime445.py').write_text(t)
for name in ('probe','layout','audit'):
    (P/f'{name}445.py').write_text(rename((OLD/f'{name}444.py').read_text()))
t=rename((OLD/'forward444.py').read_text())
edits=(
    ("graphs={'baseline':graph0};hidden_outputs={'baseline':hidden0};cases=[]",
     "graphs={'baseline':graph0};hidden_outputs={'baseline':hidden0};frames={'baseline':base};cases=[]"),
    ("g,h=capture(f);name='chunk'+str(length);graphs[name]=g;hidden_outputs[name]=h",
     "g,h=capture(f);name='chunk'+str(length);graphs[name]=g;hidden_outputs[name]=h;frames[name]=f"),
    ("    times={name:[]for name in graphs};names=list(graphs)",
     "    if set(frames)!=set(graphs)or len(frames)!=6:raise RuntimeError('ROOT445 every exact graph frame remains owned')\n"
     "    with (Path(output)/f'rank{engine.rank}-padding-prefill445-before-timing.json').open('x')as stream:\n"
     "        json.dump(dict(rank=engine.rank,cases=cases,actual_owned_graph_frames=list(frames),Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None),stream,indent=2);stream.write('\\n')\n"
     "    times={name:[]for name in graphs};names=list(graphs)"),
    ("all_parameter_storages_unchanged=True,\n        MC2_manifest_unchanged=True,physical_M=120",
     "all_parameter_storages_unchanged=True,actual_owned_graph_frames=list(frames),\n        MC2_manifest_unchanged=True,physical_M=120"),
)
for old,new in edits:
    assert t.count(old)==1,old;t=t.replace(old,new)
(P/'forward445.py').write_text(t)
plan=json.loads((OLD/'PLAN444.json').read_text());plan.update(schema='ROOT445.owned-padding-prefill-diagnostic.v1',same_actual444_math_and_cost_scope=True,graph_external_frames_explicitly_owned_through_all_replays=True,numerical_evidence_saved_before_timing=True,no_old_NN_result_inherited=True)
(P/'PLAN445.json').write_text(json.dumps(plan,indent=2)+'\n')
t=rename((OLD/'check444_HOST.py').read_text())
t=t.replace("for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')", "body=inspect.getsource(forward445.measure)\nassert \"frames={'baseline':base}\"in body and 'frames[name]=f'in body\nassert 'before-timing.json'in body\nfor p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')")
(P/'check445_HOST.py').write_text(t)
print(P)
