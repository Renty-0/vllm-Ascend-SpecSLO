from pathlib import Path
import json,hashlib,difflib
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';A=B/'tree-empty-active-cohort-causal80-active32-full-v197';P=B/'tree-causal80-small-cohort-active32-full-v202';G=D/'v202-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.loads((P/'INPUTS202.json').read_text());v['files'][str(Path(__file__))]=sha(__file__)
(P/'INPUTS202.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def record(parent,target):
 raw=parent.read_text().replace('197','202');out=target.read_text();ops=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(a=raw,b=out,autojunk=False).get_opcodes():
  if tag!='equal':ops.append(dict(start=a,end=b,old=raw[a:b],new=out[c:d]))
 return dict(file=target.name,parent=str(parent),parent_sha256=sha(parent),namespace=['197','202'],deltas=ops)
projection=[record(p,P/p.name.replace('197','202'))for p in sorted(A.glob('*.py'))if p.name!='test_full197.py']
guards=[record(p,G/p.name.replace('197','202'))for p in sorted((D/'v197-root-execution').iterdir())if p.is_file()]
(P/'PROJECTIONS202.json').write_text(json.dumps(dict(projection=projection,guard_projection=guards,new_usage_only=list(range(9,16)),all_parent_math_executed_current=True,no_parent_globals_mutated=True),indent=2,sort_keys=True)+'\n')
print('selected',len(v['files']),'whole projections',len(projection),'guard projections',len(guards))
