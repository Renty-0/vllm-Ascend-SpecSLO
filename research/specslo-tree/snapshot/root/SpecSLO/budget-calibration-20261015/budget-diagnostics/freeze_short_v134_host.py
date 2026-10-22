"""One HOST freeze/archive operation after the actual22-test source epoch."""
import hashlib,json,sys,tarfile
from pathlib import Path
P=Path('/root/SpecSLO/budget-calibration-20261015/tree-short-parent-eager-v134');sys.path.insert(0,str(P))
from check_v134 import candidate_files,external_files,source_contract,CPU,FREEZE
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
local,external=candidate_files(),external_files();c=json.loads(CPU.read_text())
assert c['status']=='passed' and c['tests_run']==22 and c['failures']==c['errors']==0 and c['NPU_initialized'] is c['model_initialized'] is False
assert c['input_hashes']==c['input_hashes_after']==local and c['external_input_hashes']==c['external_input_hashes_after']==external
record=dict(schema='V134.exclusive-prepared-HOST-source-freeze.v1',files=local,external_files=external,CPU_sha256=sha(CPU),
 source_contract=source_contract(),actual_HOST_argv=sys.orig_argv,model_initialized=False,NPU_initialized=False,actual_device_qualified=False,formal_Goodput=False)
with FREEZE.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
allfiles=local|external|{str(CPU):sha(CPU),str(FREEZE):sha(FREEZE)}
archive=P/'PREPARED_SOURCE_ARCHIVE_V134.tar.gz'
with tarfile.open(archive,'x:gz') as t:
 for name,wanted in sorted(allfiles.items()):
  p=Path(name);assert sha(p)==wanted;t.add(p,arcname='inputs/'+str(p).lstrip('/'),recursive=False)
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers();assert len(members)==len(allfiles) and len({v.name for v in members})==len(members)
 for member in members:
  name='/'+member.name.removeprefix('inputs/');assert member.isfile() and hashlib.sha256(t.extractfile(member).read()).hexdigest()==allfiles[name]
assert candidate_files()==local and external_files()==external
print(json.dumps(dict(local=len(local),external=len(external),CPU=str(CPU),CPU_sha256=sha(CPU),freeze=str(FREEZE),freeze_sha256=sha(FREEZE),archive=str(archive),archive_sha256=sha(archive),archive_members=len(allfiles),archive_bytes=archive.stat().st_size,actual_device_qualified=False)))
