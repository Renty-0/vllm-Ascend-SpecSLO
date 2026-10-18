"""Close small205 source projection from immutable202; source only."""
from pathlib import Path
import json,hashlib,difflib
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
A=B/'tree-causal80-small-cohort-active32-full-v202';P=B/'tree-causal80-small-cohort-single-writer-full-v205'
G=D/'v205-root-execution';AG=D/'v202-root-execution'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
v=json.loads((P/'INPUTS205.json').read_text());v['files'][str(Path(__file__))]=sha(__file__)
(P/'INPUTS205.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def record(parent,target):
 text=parent.read_text().replace('202','205');out=target.read_text();ops=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(a=text,b=out,autojunk=False).get_opcodes():
  if tag!='equal':ops.append(dict(start=a,end=b,old=text[a:b],new=out[c:d]))
 return dict(file=target.name,parent=str(parent),parent_sha256=sha(parent),namespace=['202','205'],deltas=ops)
projection=[record(p,P/p.name.replace('202','205'))for p in sorted(A.glob('*.py'))]
guards=[record(p,G/p.name.replace('202','205'))for p in sorted(AG.iterdir())if p.is_file()]
(P/'PROJECTIONS205.json').write_text(json.dumps(dict(projection=projection,guard_projection=guards,source_donor202_immutable=True,
 sole_production_parent197=True,only_model_actor_delta='shared output folder mkdir draft-leader-only inside original common stage',
 HOST_7th_check='real4process actual actor lambda sharedFS and common failure'),indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(readonly=len(v['files']),whole_modules=len(projection),guards=len(guards))))
