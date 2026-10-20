from pathlib import Path
import json,hashlib,difflib
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199';A=B/'tree-empty-active-cohort-causal80-active32-full-v197';OLD=B/'tree-linear-parent4-child3-bank-binding-full-v179'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
path=P/'INPUTS199.json';v=json.loads(path.read_text())
for p in [D/'build_v199_private_source.py',Path(__file__),A/'test_full197.py']:
 v['files'][str(p)]=sha(p)
# Explicit independent reference dependency, not a prior receipt.
for p in (B/'tree-union-shared4-protocol-local-v152').glob('*.py'):v['files'][str(p)]=sha(p)
for p in [B/'tree-normal4-parent3-shared4-eager-v151/PARENT_FILES151.json',B/'tree-live-shared4-union-v159/INPUTS159.json']:
 v['files'][str(p)]=sha(p)
path.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
projection=[]
for target,parent,old,new in [(name+'199.py',A/(name+'197.py'),'197','199')for name in('base','source','progress','config','check','host_full','runtime')]+[('provider199.py',OLD/'provider179.py','179','199'),('observe199.py',OLD/'observer179.py','179','199')]:
 raw=parent.read_text().replace(old,new);out=(P/target).read_text();ops=[]
 for tag,a,b,c,d in difflib.SequenceMatcher(a=raw,b=out,autojunk=False).get_opcodes():
  if tag!='equal':ops.append(dict(start=a,end=b,old=raw[a:b],new=out[c:d]))
 projection.append(dict(file=target,parent=str(parent),parent_sha256=sha(parent),namespace=[old,new],deltas=ops))
(P/'PROJECTIONS199.json').write_text(json.dumps(dict(projection=projection,definition_reuse=['epoch199:original source151 all per-owner prepare/model/finite lanes; only allocator scratch/capacity/lease binding','target199:original target151 numerical/history/FTF with actual cold3/warm1 and actual173 pad','next199:original next151 true nextB3 oracle with repair vector and own draft/epoch/observer names','component199:full component173 fresh namespace, sole actor_v146 origin import delta extension199; current original B4 and M80 unchanged','protocol199:exact original179 check_commit/commit algorithms, pre-captured object + original commit_states identity, explicit home and integer observation gate'],no_parent_globals_mutated=True),indent=2,sort_keys=True)+'\n')
# Pre-suite preparation failure is retained independently of the four-test run.
for p in (P/'full-host-attempts').glob('*/RESULT.json'):
 d=json.loads(p.read_text())
 if d.get('status')=='failed_preparation':
  target=P/('HOST_ATTEMPT199_preparation_'+p.parent.name+'.json')
  if not target.exists():target.write_bytes(p.read_bytes())
print('selected',len(v['files']),'whole projections',len(projection))
