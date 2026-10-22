"""Root's independent actual HOST/archive/source check; no device execution."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

BASE=Path('/root/SpecSLO/budget-calibration-20261015')
P=BASE/'tree-causal80-cost-v91'
OUT=BASE/'budget-diagnostics/ROOT_V91_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert sys.version_info[:2]==(3,11) and sys.dont_write_bytecode
    assert not OUT.exists()
    sys.path.insert(0,str(P))
    from causal80_cost_readiness_v91 import require
    from causal80_cost_snapshot_v91 import collect,collect_external
    ready=require()
    cpu=json.loads((P/'CPU_RESULT_V91.json').read_text())
    freeze=json.loads((P/'CAUSAL80_COST_SOURCE_MANIFEST_V91.json').read_text())
    local=collect(include_evidence=False);external=collect_external()
    checks={
      'actual37_HOST_tests':cpu['status']=='passed' and cpu['tests_run']==37 and len(cpu['modules'])==7 and cpu['failures']==cpu['errors']==[],
      'actual_HOST_model_and_NPU_absent':cpu['NPU_initialized'] is False and cpu['NPU_run'] is False and cpu['model_initialization_attempts']==cpu['NPU_initialization_attempts']==[],
      'all_local_before_after_exact':cpu['input_hashes']==cpu['input_hashes_after']==local and freeze['files']==local|{'CPU_RESULT_V91.json':sha(P/'CPU_RESULT_V91.json')},
      'all_external_before_after_exact':cpu['external_input_hashes']==cpu['external_input_hashes_after']==external==freeze['external_files'],
      'new_scope_no_inherited_NPU':ready['NPU_qualified'] is False and ready['policy_modified'] is False and ready['independent_own_AR_qualified'] is False and ready['formal_Goodput'] is False,
    }
    archive=P/'host-attempts/final-37-tests-source-evidence.tar.gz'
    with tarfile.open(archive,'r:gz') as t:
      members=t.getmembers();names={m.name:m for m in members}
      expected=freeze['files']|{'CAUSAL80_COST_SOURCE_MANIFEST_V91.json':sha(P/'CAUSAL80_COST_SOURCE_MANIFEST_V91.json')}
      checks['actual_frozen_source_CPU_archive']=len(names)==len(members) and set(expected)<=set(names) and all(names[k].isfile() and hashlib.sha256(t.extractfile(names[k]).read()).hexdigest()==v for k,v in expected.items())
    original=BASE/'tree-causal80-cost-v90/CAUSAL80_COST_SOURCE_MANIFEST_V90.json'
    checks['V90_failed_collector_source_still_frozen']=sha(original)=='cd05412b0e19bc27a0aa61e825f1827b75de70b1dfd6940ba13b99ad70b59b47'
    checks['actual_sourced_CANN_shells_in_external']=all(k in external for k in ('/root/data/specslo-cann-runtime/cann-9.0.0/set_env.sh','/root/data/specslo-cann-runtime/nnal/atb/9.0.0/atb/set_env.sh'))
    checks['root_NPU_not_initialized']=not bool(getattr(sys.modules.get('torch_npu.npu'),'_initialized',False))
    assert all(checks.values()),checks
    result={'schema':'ROOT.V91.actual-pre-NPU-source-and-HOST-audit.v1','created_UTC':datetime.now(timezone.utc).isoformat(),'all_checks_passed':True,'checks':checks,'HOST_tests':37,'CPU_sha256':sha(P/'CPU_RESULT_V91.json'),'freeze_sha256':sha(P/'CAUSAL80_COST_SOURCE_MANIFEST_V91.json'),'local_sources':len(local),'external_inputs':len(external),'source_archive':str(archive),'source_archive_sha256':sha(archive),'readiness':ready,'NPU_run':False,'formal_Goodput':False,'root_auditor_sha256':sha(__file__)}
    fd=os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
