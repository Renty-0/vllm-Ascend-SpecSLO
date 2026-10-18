"""One true bank regression plus unchanged frozen parent source proof."""
from base173 import P,read,sha,require,checked,parent_ready
from pathlib import Path

def local_files():
 return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('INPUTS173.json','SCOPE173.txt','METRICS_FIXTURE173.json','RUNNER_DERIVATION173.json','SOURCE_PORT_173.json'))}
def parent_frozen():
 pin=read(P/'INPUTS173.json')['bankfix_parent'];old=Path(pin['path']);fpath=old/'FROZEN_SOURCE167.json';cpu=old/'CPU_RESULT167.json'
 require(sha(fpath)==pin['freeze_sha256'] and sha(cpu)==pin['CPU_sha256'],'immutable failed parent freeze and all previous HOST bytes')
 f=read(fpath);c=read(cpu)
 require(f['ROOT_source_reviewed'] is True and f['CPU_sha256']==sha(cpu) and c['status']=='passed' and c['tests_run']==pin['old_tests'] and c['failures']==c['errors']==0 and c['local_before']==c['local_after']==f['files'],'unchanged original parent HOST source proof only')
 for p,h in f['files'].items():require(sha(p)==h,'unchanged frozen parent source '+p)
 return dict(parent=str(old),freeze_sha256=sha(fpath),CPU_sha256=sha(cpu),old_HOST_not_rerun=True,device_qualification_inherited=False)
def external_files():
 values=read(P/'INPUTS173.json')['files']
 for path in values:checked(path)
 parent_frozen();return values
def require_ready(expected_identity=None):
 cpu=P/'CPU_RESULT173.json';freeze=P/'FROZEN_SOURCE173.json'
 c=read(cpu);f=read(freeze)
 require(c['status']=='passed' and c['tests_run']==1 and c['failures']==c['errors']==0 and c['Torch_imported'] is c['NPU_used'] is False and c['local_before']==c['local_after']==f['files']==local_files() and c['external_before']==c['external_after']==f['external_files']==external_files() and f['CPU_sha256']==sha(cpu) and f['ROOT_source_reviewed'] is True,'new own bank HOST/current source/ROOT identity')
 identity=dict(CPU_sha256=sha(cpu),freeze_sha256=sha(freeze));require(expected_identity is None or identity==expected_identity,'own spawned bankfix epoch identity')
 return dict(identity=identity,parent=parent_ready(),frozen_bank_parent=parent_frozen(),device_qualification_inherited=False,formal_Goodput=False)
