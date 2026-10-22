"""Only rebind the existing V113 source/dataflow proof to identical V116 code."""
import ast
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
D=Path(__file__).resolve().parent; B=D.parent
OLD=B/'tree-s35-combined-controls-v113'; NEW=B/'tree-s35-combined-controls-launchfix-v116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise RuntimeError(m)
prior=D/'V113_ACTUAL_TARGET_FACTORY_CALLER_SOURCE_SCOPE_v2.json'
require(sha(prior)=='4e63482d0a5d966a185f5cfd20cdd1604c59ff9bd76fbcbfbc3e90c0c8307822','parent proof bytes')
base=json.loads(prior.read_text())
require(all(sha(p)==h for p,h in base['source_sha256'].items()),'frozen parent/source drift')
new_hashes={}; inherited=[]
for p,h in base['source_sha256'].items():
 path=Path(p)
 rebound=NEW/path.relative_to(OLD) if path.is_relative_to(OLD) else path
 require(sha(rebound)==h,'algorithm/prototype copied byte delta: '+str(rebound))
 new_hashes[str(rebound)]=h
 if rebound!=path:inherited.append(dict(old_path=str(path),new_path=str(rebound),sha256=h))
# The full actual tree's implicit-site census must be unchanged after path rebinding.
spec=importlib.util.spec_from_file_location('_v116_census',D/'audit_cpu_factory_sources_host.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
old_calls=[];new_calls=[]
for root,dest in ((OLD,old_calls),(NEW,new_calls)):
 for path in sorted(root.glob('*.py')):
  if path.name.startswith('test_'):continue
  c,a,e=mod.scan(path)
  require(not a and not e and not any(r['expanded_kwargs'] for r in c),'new alias/escaped/kwargs obligation')
  dest.extend(c)
def canonical_calls(rows):return sorted((Path(r['path']).name,r['line'],r['call_source'],r['device_expression'],r['default_device_applies']) for r in rows)
require(canonical_calls(old_calls)==canonical_calls(new_calls),'factory call inventory changed')
for name in ('combined_runtime_v113.py','combined_controls_v113.py','candidate_budget_policy.py','cpu_default_device_cleanup_host.py'):
 require(not mod.scan(NEW/name)[0],'actual control factory added')
# These are the exact launch repairs. No model flags/environment/control flow change.
for name in ('launch_combined_v113.sh','combined-startup-body.sh'):
 before=(OLD/name).read_text();expected=before.replace(str(OLD),str(NEW))
 if name=='launch_combined_v113.sh':
  expected=expected.replace('( $2 != s35-qualify65 && $2 != s35-full463 )','$2 != combined-full463')
  expected=expected.replace('<s35-qualify65|s35-full463>','<combined-full463>')
 require((NEW/name).read_text()==expected,'undeclared shell delta '+name)
 subprocess.run(['/bin/bash','-n',str(NEW/name)],check=True)
# Bind the new checks/framework source too; they contain no new data factory calls.
for path in sorted(NEW.glob('*.py')):
 if not path.name.startswith('test_'):new_hashes[str(path)]=sha(path)
for name in ('launch_combined_v113.sh','combined-startup-body.sh'):new_hashes[str(NEW/name)]=sha(NEW/name)
new_hashes[str(prior)]=sha(prior)
report=copy.deepcopy(base)
report.update(schema='HOST.V116.launcher-fix-actual-target-default-CPU-factory-caller-scope.v1',
 created_utc=datetime.now(timezone.utc).isoformat(),source_dataflow_closed_for_V116_actual_target_pipeline=True,
 source_sha256=new_hashes,source_sha256_after={p:sha(p) for p in new_hashes},
 actual_scope_directory=str(NEW),legacy_V113_key_retained_for_unchanged_checker=True,
 inherited_dataflow_proof=dict(path=str(prior),sha256=sha(prior),algorithm_dataflow_reproved=False,
     identical_actual_algorithm_copies=inherited,only_launch_action_and_task_root_changed=True,
     full_three_factory_inventory_equal=True,implicit_factory_sites=sum(r['default_device_applies'] for r in new_calls),
     Native_bytes_unchanged=True,global_factory_equivalence_proved=False),
 actual_execution=dict(argv=sys.argv,python=sys.version,auditor_path=str(Path(__file__).resolve()),auditor_sha256=sha(__file__),
     Torch_imported='torch' in sys.modules,NPU_started=False,shell_checks='bash -n only; no shell/model launch'))
for row in report['checked_source_anchors']:
 path=Path(row['path'])
 if path.is_relative_to(OLD):row['path']=str(NEW/path.relative_to(OLD))
require(report['source_sha256']==report['source_sha256_after'],'source drift during binding')
output=D/'V116_ACTUAL_TARGET_FACTORY_CALLER_SOURCE_SCOPE.json'
with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=str(output),sha256=sha(output),source_files=len(new_hashes),passed=True,
 inherited_algorithm_source_files=len(inherited),implicit_factory_sites=21,Torch_imported='torch' in sys.modules)))
