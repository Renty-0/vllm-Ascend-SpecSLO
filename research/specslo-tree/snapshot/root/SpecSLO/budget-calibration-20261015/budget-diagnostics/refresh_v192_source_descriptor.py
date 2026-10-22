"""Finalize only the fresh P192 explicit source projection and static inventory."""
from pathlib import Path
import difflib,hashlib,json
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-live-small-mixed197-cap64-active32-full-v192'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=json.loads((P/'PROJECTION192.json').read_text());outputs=set()
for row in record['projection']:
 current=P/row['output'];old=Path(row['parent']).read_text().replace('172','192');new=current.read_text();deltas=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
  if tag!='equal':deltas.append(dict(start=a,end=b,old=old[a:b],new=new[c:d]))
 row.update(edits=deltas,current_sha256=sha(current));outputs.add(current.name)
new_names={p.name for p in P.glob('*.py')}-outputs
record.update(changes_not_finalized=False,new_files={n:sha(P/n) for n in sorted(new_names)},
 change_scope=['actual197/cap64/128mask producer current scope and four full transactions','active32 exact final delegate AST cap',
 'original serialized four-field NativeGraphExecution audit','fresh source/HOST/guard identity; no old device qualification transfer'])
(P/'PROJECTION192.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
names=sorted([p.name for p in P.glob('*.py')]+['SOURCE_INPUTS192.json','STATIC_INPUTS192.json','PROJECTION192.json'])
(P/'STATIC_INPUTS192.json').write_text(json.dumps(dict(files=names),indent=2)+'\n')
