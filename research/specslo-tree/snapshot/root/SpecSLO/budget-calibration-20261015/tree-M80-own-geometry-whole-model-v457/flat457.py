"""Recheck every byte of the exact original455 closure in one traversal."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent.parent
EXPORT=B/'budget-diagnostics/ROOT456_ORIGINAL455_SOURCE_CLOSURE_HOST.json'
EXPORT_SHA='f55502f11b1636893ab6cc47b2442aaebc34d9d66f2d2795418120464a83a435'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verified():
 if sha(EXPORT)!=EXPORT_SHA:raise RuntimeError('ROOT457 exact full original455 prepared export')
 value=json.loads(EXPORT.read_text())
 if value['status']!='PASS_HOST456_ORIGINAL455_PREPARED_COMPLETE_SOURCE_EXPORT'or value['NPU_used']is not False or len(value['files'])!=5025:
  raise RuntimeError('ROOT457 actual original prepared proof')
 if value['original455_freeze']!='ad040025cafaf5d2f6b1e22d5a7952154cd9bab579b8bfa64896ff82c280809f'or value['original455_archive']!='64a6236daac945c331ac9275cf12f4e382e88ff21f186bdc1c8821349ff974f5':
  raise RuntimeError('ROOT457 exact455 source epoch')
 files=dict(value['files'])
 for path,h in value['extra_archive_and_receipt_hashes'].items():
  if path in files and files[path]!=h:raise RuntimeError('ROOT457 consistent archive bytes')
  files[path]=h
 files[str(EXPORT)]=EXPORT_SHA
 for path,h in files.items():
  if sha(path)!=h:raise RuntimeError('ROOT457 unchanged complete parent source/archive '+path)
 return files
