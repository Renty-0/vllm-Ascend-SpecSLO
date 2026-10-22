"""Private source projections only; never freeze, run models or alter parents."""
import hashlib,json,pathlib
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
JOBS=((167,173,'tree-causal80-active32-full-v167','tree-causal80-real16-bankfix-full-v173',7,108),
      (170,174,'tree-normal-cap5-active32-full-v170','tree-normal-cap5-real16-bankfix-full-v174',6,218))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,s):
 with p.open('x') as f:f.write(s)
for old,new,oldname,newname,tests,pages in JOBS:
 O=B/oldname;P=B/newname;P.mkdir(exist_ok=False);OG=D/f'v{old}-root-execution';G=D/f'v{new}-root-execution';G.mkdir(exist_ok=False)
 def port(t):return t.replace(oldname,newname).replace(str(old),str(new))
 frozen=json.loads((O/f'FROZEN_SOURCE{old}.json').read_text());original=json.loads((O/f'INPUTS{old}.json').read_text())
 for p in sorted(O.glob('*.py')):
  if p.name in (f'host{old}.py',f'test_host{old}.py',f'check{old}.py'):continue
  text=port(p.read_text())
  if p.name==f'bank{old}.py':
   anchor="source=source.replace('range(64)','range(32)').replace('32+i','16+i').replace('i+32','i+16').replace('owner%32','owner%16')"
   assert text.count(anchor)==1
   text=text.replace(anchor,anchor+".replace('i%32','i%16')")
  write(P/p.name.replace(str(old),str(new)),text)
 for name in (f'METRICS_FIXTURE{old}.json',f'RUNNER_DERIVATION{old}.json',f'SCOPE{old}.txt'):
  if (O/name).exists():write(P/name.replace(str(old),str(new)),port((O/name).read_text()))
 values=dict(original['files']);values.update(frozen['files']);values.update(frozen['retained_HOST_evidence']);values.update(frozen['execution_guard_inputs']);values.update(frozen['peer_reviews'])
 for p in (O/f'CPU_RESULT{old}.json',O/f'FROZEN_SOURCE{old}.json',pathlib.Path(frozen['root_review']['path'])):values[str(p)]=sha(p)
 original['files']=values
 original['bankfix_parent']=dict(path=str(O),freeze_sha256=sha(O/f'FROZEN_SOURCE{old}.json'),CPU_sha256=sha(O/f'CPU_RESULT{old}.json'),old_tests=tests,all_previous_HOST_retained=True,device_qualification_inherited=False)
 write(P/f'INPUTS{new}.json',json.dumps(original,indent=2,sort_keys=True)+'\n')
 for p in OG.iterdir():
  if not p.is_file():continue
  text=port(p.read_text())
  if p.name==f'guard{old}.py':text=text.replace(f"c['tests_run']=={tests}","c['tests_run']==1").replace(f'actual own{tests} HOST','actual own1 HOST')
  write(G/p.name.replace(str(old),str(new)),text)
 check='''"""One new true bank regression plus unchanged frozen parent source proof."""
from baseE import P,read,sha,require,checked,parent_ready
from pathlib import Path

def local_files():
 return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('INPUTSE.json','SCOPEE.txt','METRICS_FIXTUREE.json','RUNNER_DERIVATIONE.json','SOURCE_PORT_E.json'))}
def parent_frozen():
 pin=read(P/'INPUTSE.json')['bankfix_parent'];old=Path(pin['path']);fpath=old/'FROZEN_SOURCEO.json';cpu=old/'CPU_RESULTO.json'
 require(sha(fpath)==pin['freeze_sha256'] and sha(cpu)==pin['CPU_sha256'],'immutable failed parent freeze and all previous HOST bytes')
 f=read(fpath);c=read(cpu)
 require(f['ROOT_source_reviewed'] is True and f['CPU_sha256']==sha(cpu) and c['status']=='passed' and c['tests_run']==pin['old_tests'] and c['failures']==c['errors']==0 and c['local_before']==c['local_after']==f['files'],'unchanged original parent HOST source proof only')
 for p,h in f['files'].items():require(sha(p)==h,'unchanged frozen parent source '+p)
 return dict(parent=str(old),freeze_sha256=sha(fpath),CPU_sha256=sha(cpu),old_HOST_not_rerun=True,device_qualification_inherited=False)
def external_files():
 values=read(P/'INPUTSE.json')['files']
 for path in values:checked(path)
 parent_frozen();return values
def require_ready(expected_identity=None):
 cpu=P/'CPU_RESULTE.json';freeze=P/'FROZEN_SOURCEE.json'
 c=read(cpu);f=read(freeze)
 require(c['status']=='passed' and c['tests_run']==1 and c['failures']==c['errors']==0 and c['Torch_imported'] is c['NPU_used'] is False and c['local_before']==c['local_after']==f['files']==local_files() and c['external_before']==c['external_after']==f['external_files']==external_files() and f['CPU_sha256']==sha(cpu) and f['ROOT_source_reviewed'] is True,'new own bank HOST/current source/ROOT identity')
 identity=dict(CPU_sha256=sha(cpu),freeze_sha256=sha(freeze));require(expected_identity is None or identity==expected_identity,'own spawned bankfix epoch identity')
 return dict(identity=identity,parent=parent_ready(),frozen_bank_parent=parent_frozen(),device_qualification_inherited=False,formal_Goodput=False)
'''.replace('E',str(new)).replace('O',str(old))
 # Replace explicit placeholder tokens only; uppercase identifiers elsewhere stay untouched.
 check=check.replace('CPU_R'+str(new)+'SULT','CPU_RESULT').replace('FROZ'+str(new)+'N_S'+str(old)+'URC'+str(new),'FROZEN_SOURCE') if False else check
 # The template uses E/O only in known names; restore ordinary capitalized words.
 for damaged,correct in ((f'FR{new}Z{old}N_S{old}URC{new}','FROZEN_SOURCE'),):pass
 # Recreate safely from a tokenized template if an identifier was affected.
 check=check.replace(f'FROZ{new}N_S{old}URC{new}','FROZEN_SOURCE').replace(f'CPU_R{new}SULT','CPU_RESULT')
 write(P/f'check{new}.py',check)
 print(new,len(list(P.iterdir())),len(values),len(list(G.iterdir())))
