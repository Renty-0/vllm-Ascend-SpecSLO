"""New source/CPU identity; no source or prior result grants model flags."""
import hashlib,tarfile
from pathlib import Path
from base199 import P,P173,read,sha,require,parent_ready
def local_files():
 return {str(p):sha(p)for p in sorted(P.iterdir())if p.is_file()and(p.suffix=='.py'or p.name in('INPUTS199.json','PROJECTIONS199.json','SCOPE199.txt','FIXTURE_INPUT199.json'))}
def external_files():
 values=read(P/'INPUTS199.json')['files']
 for p,h in values.items():require(sha(p)==h,'exact selected source '+p)
 return values
def require_parent(*,archive=False):
 pin=read(P/'INPUTS199.json')['P173'];f=read(P173/'FROZEN_SOURCE173.json');c=read(P173/'CPU_RESULT173.json')
 require(sha(P173/'FROZEN_SOURCE173.json')==pin['freeze_sha256']and sha(P173/'CPU_RESULT173.json')==pin['CPU_sha256']
  and c['status']=='passed'and c['tests_run']==1 and c['failures']==c['errors']==0
  and c['local_before']==c['local_after']==f['files']and c['external_before']==c['external_after']==f['external_files'],
  'original173 bank source HOST retained,never rerun or transferred as model evidence')
 expected={**f['files'],**f['external_files'],**f['retained_HOST_evidence'],**f['execution_guard_inputs'],**f['peer_reviews'],
  str(P173/'CPU_RESULT173.json'):pin['CPU_sha256'],str(P173/'FROZEN_SOURCE173.json'):pin['freeze_sha256'],f['root_review']['path']:f['root_review']['sha256']}
 for p,h in expected.items():require(sha(p)==h,'immutable selected parent173 source '+p)
 for row in read(P/'PROJECTIONS199.json')['projection']:
  parent=Path(row['parent']);require(sha(parent)==row['parent_sha256'],'exact projection parent')
  text=parent.read_text().replace(*row['namespace']);parts=[];offset=0
  for op in row['deltas']:
   require(op['start']>=offset and text[op['start']:op['end']]==op['old'],'exact source delta')
   parts.extend((text[offset:op['start']],op['new']));offset=op['end']
  result=''.join(parts)+text[offset:]+row.get('append_text','')
  require(result==(P/row['file']).read_text(),'whole source projection '+row['file'])
 if archive:
  require(sha(pin['archive']['path'])==pin['archive']['sha256'],'parent173 archive in place')
  with tarfile.open(pin['archive']['path'],'r:gz')as saved:
   members=saved.getmembers();require(len(members)==len(expected)and len({m.name for m in members})==len(members),'exact parent173 regular member inventory')
   for m in members:
    require(m.isfile()and '/'+m.name in expected and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==expected['/'+m.name],
     'exact parent173 archived member '+m.name)
 return dict(production_parent=str(P173),original_parent_suites_rerun=False,device_qualification_inherited=False,archive_verified_in_place=archive)
def require_ready(expected_identity=None):
 c=read(P/'CPU_RESULT199.json');f=read(P/'FROZEN_SOURCE199.json')
 require(c['status']=='passed'and c['tests_run']==4 and c['errors']==c['failures']==0
  and c['Torch_imported']is c['NPU_used']is False and c['local_before']==c['local_after']==f['files']==local_files()
  and c['external_before']==c['external_after']==f['external_files']==external_files()
  and f['CPU_sha256']==sha(P/'CPU_RESULT199.json')and f['ROOT_source_reviewed']is True,'current199 source/HOST/ROOT freeze')
 require_parent();identity=dict(CPU_sha256=sha(P/'CPU_RESULT199.json'),freeze_sha256=sha(P/'FROZEN_SOURCE199.json'))
 require(expected_identity is None or identity==expected_identity,'same spawned source epoch')
 return dict(identity=identity,parent=parent_ready(),device_qualification_inherited=False,formal_Goodput=False)
if __name__=='__main__':
 import json
 print(json.dumps(require_ready(),sort_keys=True))
