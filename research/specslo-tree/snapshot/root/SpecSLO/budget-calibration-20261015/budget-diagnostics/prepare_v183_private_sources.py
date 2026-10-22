"""Generate only small P183 source descriptors; never freeze or run a model."""
import difflib,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-uniform-b3-persistent-metadata-full-v183'
PARENT=B/'tree-uniform-b3-three-bucket-full-v180'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def main():
 parent=read(PARENT/'FROZEN_SOURCE180.json');old=read(PARENT/'CPU_RESULT180.json')
 assert old['status']=='passed'and old['tests_run']==5
 assert old['local_before']==old['local_after']==parent['files']
 assert old['external_before']==old['external_after']==parent['external_files']
 # Source/archive prerequisites only. Parent device results and current raw
 # output are neither opened nor included in this new candidate.
 inherited={**parent['files'],**parent['external_files'],**parent['retained_HOST_evidence'],
  **parent['execution_guard_inputs'],**parent['peer_reviews'],
  str(PARENT/'CPU_RESULT180.json'):sha(PARENT/'CPU_RESULT180.json'),
  str(PARENT/'FROZEN_SOURCE180.json'):sha(PARENT/'FROZEN_SOURCE180.json'),
  parent['root_review']['path']:parent['root_review']['sha256'],str(Path(__file__)):sha(__file__)}
 assert not any(path.endswith('.tar.gz')for path in inherited)
 for path,h in inherited.items():assert sha(path)==h,path
 inputs=read(PARENT/'INPUTS180.json');inputs['files']=inherited
 inputs['P180']=dict(path=str(PARENT),freeze_sha256=sha(PARENT/'FROZEN_SOURCE180.json'),
  CPU_sha256=sha(PARENT/'CPU_RESULT180.json'),archive=dict(path=str(PARENT/'PREPARED_SOURCE_ARCHIVE180.tar.gz'),
   sha256=sha(PARENT/'PREPARED_SOURCE_ARCHIVE180.tar.gz')))
 inputs['source_only_parent_results_not_device_qualification']=True
 write(P/'INPUTS183.json',inputs)
 rows=[]
 for oldpath in sorted(parent['files']):
  oldpath=Path(oldpath)
  if oldpath.suffix!='.py':continue
  name=oldpath.name.replace('180','183');current=P/name
  if not current.is_file():continue
  text=oldpath.read_text().replace('180','183');new=current.read_text();ops=[]
  for tag,a,b,c,d in difflib.SequenceMatcher(a=text,b=new,autojunk=False).get_opcodes():
   if tag!='equal':ops.append(dict(start=a,end=b,old=text[a:b],new=new[c:d]))
  rows.append(dict(file=name,parent=str(oldpath),parent_sha256=sha(oldpath),namespace='180->183',
   deltas=ops,sha256=sha(current)))
 write(P/'SOURCE_DELTAS183.json',dict(schema='V183.explicit-small-source-projection.v1',
  projection=rows,new_usage_files=['staging183.py','lease183.py','usage183.py','audit_usage183.py','test_staging183.py'],
  no_parent_file_changed=True,parent_device_qualification_inherited=False))
 print(json.dumps(dict(local_source_projection_count=len(rows),selected_external_count=len(inherited),
  parent_five_HOST_not_rerun=True,source_only=True,freeze_created=False,NPU_used=False)))
if __name__=='__main__':main()
