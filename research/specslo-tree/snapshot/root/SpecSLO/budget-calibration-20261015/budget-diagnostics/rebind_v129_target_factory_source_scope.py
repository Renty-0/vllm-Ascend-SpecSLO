"""Rebind V116 full target caller proof; qualification-only actor repairs excluded."""
import ast
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
D=Path(__file__).resolve().parent;B=D.parent
OLD=B/'tree-s35-qualified-counts-v119';NEW=B/'tree-s35-serialized-evidence-v129'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise RuntimeError(m)
prior=D/'V119_ACTUAL_TARGET_FACTORY_CALLER_SOURCE_SCOPE.json'
require(sha(prior)=='348f9e5c52dcb156e85d08159908048906bcfe527af78a4e29f9a46e29905d4a','parent proof SHA')
base=json.loads(prior.read_text());require(all(sha(p)==h for p,h in base['source_sha256'].items()),'permanent parent/source changed')
qualification={'shape_scope_v101.py','combined_runtime_v113.py','audit_combined_v113.py'}
framework={'check_v113.py','source_snapshot_v113.py'}
shells={'launch_combined_v113.sh','combined-startup-body.sh'}
newhashes={};changed=[];identical=[]
for name,wanted in base['source_sha256'].items():
 p=Path(name);q=NEW/p.relative_to(OLD) if p.is_relative_to(OLD) else p
 actual=sha(q)
 if actual!=wanted:
  require(p.parent==OLD and p.name in qualification|framework|shells,'unexpected fulltarget/prototype source delta: '+str(q))
  changed.append(dict(old_path=str(p),new_path=str(q),old_sha256=wanted,new_sha256=actual,
   scope='exact serialized evidence retention/readback or raw metadata only; numerical validators unchanged' if p.name in qualification else 'literal private launcher task_root change only' if p.name in shells else 'HOST source/readiness framework only'))
 else:
  if q!=p:identical.append(dict(old_path=str(p),new_path=str(q),sha256=wanted))
 newhashes[str(q)]=actual
require({Path(r['new_path']).name for r in changed}==qualification|framework|shells,'declared7 modified source inventory')
for name in ('combined_controls_v113.py','cpu_default_device_cleanup_host.py','candidate_budget_policy.py','production_probe_v101.py'):
 require((OLD/name).read_bytes()==(NEW/name).read_bytes(),'actual fulltarget runtime/control must be identical')
spec=importlib.util.spec_from_file_location('_v129_census',D/'audit_cpu_factory_sources_host.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def calls(root):
 allrows=[]
 for p in root.glob('*.py'):
  if p.name.startswith('test_'):continue
  rows,a,e=mod.scan(p);require(not a and not e and not any(r['expanded_kwargs'] for r in rows),'unreviewed factory alias/escaped/**kwargs')
  allrows.extend(rows)
 return allrows
a,b=calls(OLD),calls(NEW)
def canon(rows):return sorted((Path(r['path']).name,ast.dump(ast.parse(r['call_source'].strip(),mode='eval'),include_attributes=False),r['device_expression'],r['default_device_applies']) for r in rows)
# Exact additions are HOST serialization/metadata only; unchanged factory ASTs.
spec2=importlib.util.spec_from_file_location('_v129_serialized_contract',NEW/'serialized_structure_evidence_v129.py')
helper=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(helper)
require(all(v is True for v in helper.source_contract().values()),'exact source retention contract')
require(canon(a)==canon(b),'actual factory call inventory/expression delta')
for name in ('launch_combined_v113.sh','combined-startup-body.sh'):
 require((NEW/name).read_text()==(OLD/name).read_text().replace(str(OLD),str(NEW)),'only literal shell task_root path may change')
 subprocess.run(['/bin/bash','-n',str(NEW/name)],check=True);newhashes[str(NEW/name)]=sha(NEW/name)
for p in NEW.glob('*.py'):
 if not p.name.startswith('test_'):newhashes[str(p)]=sha(p)
newhashes[str(prior)]=sha(prior)
r=copy.deepcopy(base)
r.update(schema='HOST.V129.serialized-evidence-target-default-CPU-factory-caller-scope.v1',created_utc=datetime.now(timezone.utc).isoformat(),
 actual_scope_directory=str(NEW),source_dataflow_closed_for_V119_actual_target_pipeline=True,source_dataflow_closed_for_V129_actual_target_pipeline=True,
 source_sha256=newhashes,source_sha256_after={p:sha(p) for p in newhashes},
 inherited_dataflow_proof=dict(path=str(prior),sha256=sha(prior),actual_fulltarget_algorithm_copies_identical=identical,
  modified_qualification_or_HOST_only_sources=changed,qualification_only_actors_execute_in_original_CPU_context=True,
  fulltarget_dataflow_reproved=False,factory_expressions_unchanged=True,factory_line_numbers_moved=True,
  Native_source_unchanged=True,old_V119_actual_failure_not_hidden=True,new_actual21scope_qualification_proved=False,serialized_retention_is_not_numerical_qualification=True),
 actual_execution=dict(argv=sys.argv,python=sys.version,auditor_path=str(Path(__file__).resolve()),auditor_sha256=sha(__file__),Torch_imported='torch' in sys.modules,NPU_started=False,shell_checks='bash -n only'))
for row in r['checked_source_anchors']:
 p=Path(row['path'])
 if p.is_relative_to(OLD):row['path']=str(NEW/p.relative_to(OLD))
require(r['source_sha256']==r['source_sha256_after'],'source drift')
out=D/'V129_ACTUAL_TARGET_FACTORY_CALLER_SOURCE_SCOPE.json'
with out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),source_files=len(newhashes),identical_copied_sources=len(identical),changed_sources=len(changed),factory_implicit=sum(r['default_device_applies'] for r in b),passed=True,Torch_imported='torch' in sys.modules)))
