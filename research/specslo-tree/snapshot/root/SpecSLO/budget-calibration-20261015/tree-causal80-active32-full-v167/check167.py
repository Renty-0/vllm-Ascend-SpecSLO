"""Current source/CPU/ROOT freeze, plus immutable P148 launch prerequisites."""
import json
from base167 import P,read,sha,require,checked,parent_ready
def local_files():
 return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and
         (p.suffix=='.py' or p.name in ('INPUTS167.json','SCOPE167.txt','METRICS_FIXTURE167.json','RUNNER_DERIVATION167.json'))}
def external_files():
 values=read(P/'INPUTS167.json')['files']
 for path in values:checked(path)
 return values
def require_ready(expected_identity=None):
 cpu=P/'CPU_RESULT167.json';freeze=P/'FROZEN_SOURCE167.json'
 require(cpu.is_file() and freeze.is_file(),'own actual HOST and explicit ROOT freeze')
 c=read(cpu);f=read(freeze)
 require(c['status']=='passed' and c['tests_run']==7 and c['failures']==c['errors']==0
  and c['Torch_imported'] is c['NPU_used'] is False
  and c['local_before']==c['local_after']==f['files']==local_files()
  and c['external_before']==c['external_after']==f['external_files']==external_files()
  and f['CPU_sha256']==sha(cpu) and f['ROOT_source_reviewed'] is True,'own exact CPU/source/ROOT identity')
 identity=dict(CPU_sha256=sha(cpu),freeze_sha256=sha(freeze))
 require(expected_identity is None or identity==expected_identity,'spawned current epoch identity')
 return dict(identity=identity,parent=parent_ready(),device_qualification_inherited=False,formal_Goodput=False)
if __name__=='__main__':print(json.dumps(require_ready()))
