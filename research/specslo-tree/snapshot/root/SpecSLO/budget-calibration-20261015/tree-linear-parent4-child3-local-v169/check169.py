"""New local parent4 source proof; it never borrows numerical qualification."""
from base169 import P,read,sha,require,parent_ready

def local_files():
 return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('INPUTS169.json','FIXTURE_INPUT169.json','SCOPE169.json'))}
def external_files():
 files=read(P/'INPUTS169.json')['files']
 for p,h in files.items():require(sha(p)==h,'immutable selected input '+p)
 return files

def require_ready(expected_identity=None):
 c=read(P/'CPU_RESULT169.json');f=read(P/'FROZEN_SOURCE169.json')
 require(c['status']=='passed' and c['tests_run']==4 and c['failures']==c['errors']==0 and c['Torch_imported'] is c['NPU_used'] is False
  and c['local_before']==c['local_after']==f['files']==local_files()
  and c['external_before']==c['external_after']==f['external_files']==external_files()
  and f['CPU_sha256']==sha(P/'CPU_RESULT169.json') and f['ROOT_source_reviewed'] is True,'new own4HOST/source/ROOT identity')
 identity=dict(CPU_sha256=sha(P/'CPU_RESULT169.json'),freeze_sha256=sha(P/'FROZEN_SOURCE169.json'))
 require(expected_identity is None or identity==expected_identity,'own spawned current epoch identity')
 return dict(identity=identity,parent=parent_ready(),device_qualification_inherited=False,full463_qualified=False,formal_Goodput=False)
if __name__=='__main__':
 import json
 print(json.dumps(require_ready()))
