"""Materialize the complete original455 prepared closure, without a device."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys,time
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M160-qualified-profile-whole-model-v455'
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST456 source export no model')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env455 import setup
setup()
import guard455
source=json.loads((P/'SOURCE455_RECEIPT.json').read_text());guard455.EXPECTED_FREEZE=source['freeze_sha256'];guard455.EXPECTED_ARCHIVE=source['archive_sha256']
start=time.monotonic();_,closure,readiness=guard455.prepared();elapsed=time.monotonic()-start
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert len(closure)==5025
archives={}
for folder in sorted({Path(p).parent for p in closure}):
 for f in folder.glob('SOURCE*.tar.gz'):
  if f.is_file():archives[str(f)]=sha(f)
receipt=P/'SOURCE455_RECEIPT.json'
extra=dict(archives);extra[str(receipt)]=sha(receipt)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST456_ORIGINAL455_PREPARED_COMPLETE_SOURCE_EXPORT',
 original455_freeze=source['freeze_sha256'],original455_archive=source['archive_sha256'],
 original_prepared_elapsed_seconds=elapsed,files=closure,extra_archive_and_receipt_hashes=extra,
 original_prepared_readiness=readiness,NPU_used=False,model_constructors=0,NN_permission_inherited=False)
destination=D/'ROOT456_ORIGINAL455_SOURCE_CLOSURE_HOST.json'
with destination.open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(status=out['status'],destination=str(destination),sha256=sha(destination),files=len(closure),additional_archives_and_receipts=len(extra),original_prepared_seconds=elapsed,NPU_used=False)))
