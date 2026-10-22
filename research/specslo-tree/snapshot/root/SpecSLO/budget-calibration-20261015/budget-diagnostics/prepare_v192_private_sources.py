"""Create the reviewed P172 transaction projection in a fresh P192 only.

This is source preparation. It imports no backend and never prepares a freeze.
"""
from pathlib import Path
import ast, hashlib, json

B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-live-large-mixed-mask-int8-full-v172'
P=B/'tree-live-small-mixed197-cap64-active32-full-v192'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(name,text):
 p=P/name
 if p.exists():raise RuntimeError('Keep prior source bytes: '+str(p))
 p.write_text(text)

def main():
 assert not list(P.iterdir()), 'fresh private source directory'
 projection=[]
 for parent in sorted(OLD.glob('*.py')):
  if parent.name in ('host172.py','check172.py'):continue
  name=parent.name.replace('172','192');text=parent.read_text().replace('172','192')
  put(name,text)
  projection.append(dict(output=name,parent=str(parent),parent_sha256=sha(parent),namespace_projection_sha256=hashlib.sha256(text.encode()).hexdigest()))
 original=json.loads((OLD/'SOURCE_INPUTS172.json').read_text())
 files=dict(original['files'])
 frozen=json.loads((OLD/'FROZEN_SOURCE172.json').read_text())
 for field in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
  for path,h in frozen[field].items():
   assert path not in files or files[path]==h
   files[path]=h
 files[str(OLD/'CPU_RESULT172.json')]=sha(OLD/'CPU_RESULT172.json')
 files[str(OLD/'FROZEN_SOURCE172.json')]=sha(OLD/'FROZEN_SOURCE172.json')
 files[frozen['root_review']['path']]=frozen['root_review']['sha256']
 extra=[B/'tree-active-window32-16-local-v160/window160.py',
        B/'tree-active-window32-16-local-v160/PARENT_INPUTS160.json',
        B/'budget-diagnostics/v177-root-safe-storage-overlay/safe_storage177.py',
        B/'budget-diagnostics/v177-root-safe-storage-overlay/FROZEN_OVERLAY177.json',
        B/'budget-diagnostics/v177-root-safe-storage-overlay/CPU_RESULT177.json',Path(__file__)]
 for path in extra:files[str(path)]=sha(path)
 for path,h in files.items():assert sha(path)==h,path
 original.update(files=files,original_P172=dict(path=str(OLD),freeze_sha256=sha(OLD/'FROZEN_SOURCE172.json'),
  CPU_sha256=sha(OLD/'CPU_RESULT172.json'),archive_path=str(OLD/'PREPARED_SOURCE_ARCHIVE172.tar.gz'),
  archive_sha256=sha(OLD/'PREPARED_SOURCE_ARCHIVE172.tar.gz'),device_qualification_inherited=False))
 put('SOURCE_INPUTS192.json',json.dumps(original,sort_keys=True,indent=2)+'\n')
 put('PROJECTION192.json',json.dumps(dict(schema='V192.P172-whole-source-projection-with-explicit-deltas.v1',projection=projection,
  parent_source_immutable=True,parent_device_qualification_inherited=False,changes_not_finalized=True),indent=2)+'\n')

if __name__=='__main__':main()
