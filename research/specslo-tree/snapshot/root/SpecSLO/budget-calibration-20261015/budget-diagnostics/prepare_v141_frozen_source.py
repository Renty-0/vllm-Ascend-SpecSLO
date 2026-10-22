"""Freeze the already successful three-test V141 CPU evidence; no test/model rerun."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tarfile
P=Path('/root/SpecSLO/budget-calibration-20261015/tree-single-point-config-snapshot-fix-v141')
D=P.parent/'budget-diagnostics'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 if sys.version_info[:2]!=(3,11) or not sys.dont_write_bytecode:raise RuntimeError('retained3.11 -B required')
 sys.path.insert(0,str(P))
 from check_v139 import candidate_files,external_files,source_contract,require_ready,CPU,FREEZE
 cpu=json.loads(CPU.read_text());local,external=candidate_files(),external_files()
 assert cpu['status']=='passed' and cpu['tests_run']==3 and cpu['errors']==cpu['failures']==0
 assert cpu['input_hashes']==cpu['input_hashes_after']==local
 assert cpu['external_input_hashes']==cpu['external_input_hashes_after']==external
 assert cpu['parent5_HOST_proof_reused'] is True and cpu['parent5_suite_rerun'] is False
 assert cpu['Torch_imported'] is True and not cpu['forbidden_import_attempts'] and not cpu['forbidden_runtime_modules_loaded']
 assert cpu['NPU_initialized'] is cpu['NPU_or_ACL_queried'] is cpu['model_initialized'] is False
 scope=source_contract()
 attempts={str(path):sha(path) for path in sorted((P/'host-attempts').glob('attempt*.json'))}
 freeze=dict(schema='V141.metadata-only-one-point-prepared-source.v1',preparation_epoch='V141',files=local,external_files=external,
  CPU_sha256=sha(CPU),retained_attempt_files=attempts,parent5_actual_HOST_proof_reused_not_rerun=True,
  actual_device_qualification=False,extra4_adoption=False,
  actual_actor_model_fixture_math_and_MC2_document_unchanged=True)
 with FREEZE.open('x') as f:json.dump(freeze,f,indent=2,allow_nan=False);f.write('\n')
 paths={**local,str(CPU):sha(CPU),str(FREEZE):sha(FREEZE),**attempts}
 archive=P/'PREPARED_SOURCE_ARCHIVE_V141.tar.gz'
 with tarfile.open(archive,'x:gz') as tar:
  for name,wanted in sorted(paths.items()):
   path=Path(name);assert path.is_relative_to(P) and sha(path)==wanted
   tar.add(path,arcname=str(path.relative_to(P)),recursive=False)
 with tarfile.open(archive,'r:gz') as tar:
  members=tar.getmembers()
  assert len(members)==len(paths)==len({m.name for m in members}) and all(m.isfile() for m in members)
  assert {str(P/m.name):hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==paths
 ready=require_ready()
 argv=[sys.executable,'-B',str(P/'check_v139.py')]
 checked=subprocess.run(argv,capture_output=True,text=True,env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0'})
 assert checked.returncode==0,checked.stdout+checked.stderr
 child_ready=json.loads(checked.stdout);assert child_ready==ready
 assert candidate_files()==local and external_files()==external
 result=dict(schema='V141.frozen-HOST-readiness.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='prepared',
  preparation_epoch='V141',directory=str(P),actual_prepare_argv=sys.orig_argv,prepare_script_sha256=sha(__file__),
  tests_run=3,parent5_proof_reused_by_SHA_not_rerun=True,new_tests_rerun_by_prepare=False,
  CPU=dict(path=str(CPU),sha256=sha(CPU),actual_HOST_argv=cpu['actual_HOST_argv'],elapsed_seconds=cpu['elapsed_seconds']),
  freeze=dict(path=str(FREEZE),sha256=sha(FREEZE)),source_count=len(local),external_input_count=len(external),
  archive=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,members=len(paths),all_member_bytes_checked=True),
  readiness_argv=argv,readiness_exit_code=checked.returncode,readiness=ready,
  backend_or_model_imported_by_prepare=False,NPU_or_ACL_or_model_or_query_executed=False,
  actual_device_qualification=False,extra4_adoption=False,formal_Goodput=False)
 output=D/'V141_PREPARED_READINESS.json'
 with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status='prepared',path=str(output),sha256=sha(output),CPU_sha256=sha(CPU),freeze_sha256=sha(FREEZE),archive_sha256=sha(archive),archive_members=len(paths),readiness_exit_code=checked.returncode)))
if __name__=='__main__':main()
