"""Own source identity; parent169 source proof confers no device qualification."""
from pathlib import Path
from base179 import P,read,sha,require,parent_ready
def local_files():
 return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('INPUTS179.json','FIXTURE_INPUT179.json'))}
def external_files():
 files=read(P/'INPUTS179.json')['files']
 for p,h in files.items():require(sha(p)==h,'immutable selected input '+p)
 return files
def verify_parent_projection(*,archive=False):
 pin=read(P/'INPUTS179.json')['parent_P169'];root=Path(pin['path'])
 require(sha(root/'FROZEN_SOURCE169.json')==pin['freeze_sha256'] and sha(root/'CPU_RESULT169.json')==pin['CPU_sha256'],'actual parent source/HOST identity')
 c=read(root/'CPU_RESULT169.json');f=read(root/'FROZEN_SOURCE169.json')
 require(c['status']=='passed' and c['tests_run']==4 and c['failures']==c['errors']==0
  and c['local_before']==c['local_after']==f['files'] and c['external_before']==c['external_after']==f['external_files'],'old source-only HOST evidence retained')
 for path,row in pin['projection'].items():
  require(sha(path)==row['parent_sha256'],'original projection bytes '+path);source=Path(path).read_text()
  if row['namespace_replacement']:source=source.replace('169','179')
  for change in row['deltas']:
   require(source.count(change['old'])==1,'unique minimal source delta');source=source.replace(change['old'],change['new'])
  require(Path(row['path']).read_text()==source,'whole-byte reversible definition '+path)
 if archive:require(sha(pin['archive']['path'])==pin['archive']['sha256'],'old169 archive in place')
 return dict(parent=str(root),CPU_sha256=pin['CPU_sha256'],freeze_sha256=pin['freeze_sha256'],archive=pin['archive'],
  archive_verified_in_place=archive,projected_definitions=len(pin['projection']),old4HOST_rerun=False,old_binary_copied=False,parent_failure_confers_qualification=False)
def require_ready(expected_identity=None):
 c=read(P/'CPU_RESULT179.json');f=read(P/'FROZEN_SOURCE179.json')
 require(c['status']=='passed' and c['tests_run']==3 and c['failures']==c['errors']==0 and c['Torch_imported'] is c['NPU_used'] is False
  and c['local_before']==c['local_after']==f['files']==local_files() and c['external_before']==c['external_after']==f['external_files']==external_files()
  and f['CPU_sha256']==sha(P/'CPU_RESULT179.json') and f['ROOT_source_reviewed'] is True,'own3HOST/source/ROOT identity')
 verify_parent_projection();identity=dict(CPU_sha256=sha(P/'CPU_RESULT179.json'),freeze_sha256=sha(P/'FROZEN_SOURCE179.json'))
 require(expected_identity is None or identity==expected_identity,'own spawned current epoch identity')
 return dict(identity=identity,parent=parent_ready(),device_qualification_inherited=False,full463_qualified=False,formal_Goodput=False)
if __name__=='__main__':
 import json
 print(json.dumps(require_ready()))
