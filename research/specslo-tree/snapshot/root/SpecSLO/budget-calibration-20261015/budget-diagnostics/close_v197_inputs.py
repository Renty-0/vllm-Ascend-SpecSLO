from pathlib import Path
import difflib,hashlib,json
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-empty-active-cohort-causal80-active32-full-v197';M=B/'tree-vacancy-cohort-prefill-cap4-active32-full-v190';S=B/'tree-causal80-real16-bankfix-full-v173'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((M/'INPUTS190.json').read_text());base=json.loads((S/'INPUTS173.json').read_text())
files=dict(base['files']);f=json.loads((S/'FROZEN_SOURCE173.json').read_text())
for category in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):files.update(f[category])
files.update({str(S/'CPU_RESULT173.json'):sha(S/'CPU_RESULT173.json'),str(S/'FROZEN_SOURCE173.json'):sha(S/'FROZEN_SOURCE173.json'),f['root_review']['path']:f['root_review']['sha256']})
for n in ('prefill190.py','publication190.py','next4_190.py','check190.py','base190.py','source190.py','progress190.py','runtime190.py','trace190.py','audit190.py','scope190.py','component190.py','extension190.py'):
 files[str(M/n)]=sha(M/n)
for k,v in old['files'].items():
 if '/budget-diagnostics/v154-root-execution/' in k or '/budget-diagnostics/v177-root-safe-storage-overlay/'in k:files[k]=v
for n in ('guard190.py','launch190.sh','startup190.sh','GUARD_INPUTS190.json'):
 p=D/'v190-root-execution'/n;files[str(p)]=sha(p)
for p,h in files.items():assert sha(p)==h,p
inputs={k:v for k,v in old.items()if k!='files'};inputs['files']=files
# Write final input map after all explicit projection parents are included below.
# Whole source diffs against explicit immutable producer, regenerated before HOST/final close.
rows=[]
for p in sorted(P.glob('*.py')):
 oldp=M/(p.name.replace('197','190'))
 if not oldp.is_file():continue
 files[str(oldp)]=sha(oldp)
 original=oldp.read_text().replace('190','197');new=p.read_text();ops=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(None,original,new,autojunk=False).get_opcodes():
  if tag!='equal':ops.append(dict(start=a,end=b,old=original[a:b],new=new[c:d]))
 rows.append(dict(file=p.name,parent=str(oldp),parent_sha256=sha(oldp),namespace=['190','197'],deltas=ops))
(P/'PROJECTIONS197.json').write_text(json.dumps(dict(schema='V197.thin-private-source-projection.v1',production_parent=str(S),readonly_multiowner_math_source=str(M),projection=rows),indent=2,sort_keys=True)+'\n')
inputs['files']=files
(P/'INPUTS197.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
print(len(files),len(rows))
