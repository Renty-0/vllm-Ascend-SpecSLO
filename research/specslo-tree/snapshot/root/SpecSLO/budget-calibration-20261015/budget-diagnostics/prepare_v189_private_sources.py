"""Selected source projection only; no freeze, device or model actions."""
import ast,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-causal80-real16-bankfix-full-v173';P=B/'tree-slo-admission1-causal80-active32-full-v189';G=D/'v189-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def main():
 assert not (P/'FROZEN_SOURCE189.json').exists() and not (P/'CPU_RESULT189.json').exists()
 f=read(OLD/'FROZEN_SOURCE173.json');c=read(OLD/'CPU_RESULT173.json')
 assert c['status']=='passed' and c['local_before']==c['local_after']==f['files']
 rows={}
 for path,h in f['files'].items():
  old=Path(path);assert sha(old)==h
  if old.name in ('INPUTS173.json','RUNNER_DERIVATION173.json','SOURCE_PORT_173.json','test_host173.py'):continue
  name=old.name.replace('173','189');text=old.read_text();changes=[]
  for before,after in ((OLD.name,P.name),('173','189'),("ACTION='causal80-active32-full463'","ACTION='slo-admission1-causal80-active32-full463'")):
   count=text.count(before)
   if count:changes.append(dict(old=before,new=after,count=count));text=text.replace(before,after)
  actual=(P/name).read_text();row=dict(parent=str(old),parent_sha256=h,sha256=sha(P/name),changes=changes)
  if actual!=text:
   assert name in ('runtime189.py','audit189.py','check189.py','host189.py'),name
   row['replacement_text']=actual;row['replacement_scope']='untimed policy guard/report, retained audit or necessary HOST/source identity; no model math'
  if (P/name).suffix=='.py':compile(ast.parse(actual),str(P/name),'exec')
  rows[name]=row
 archive=OLD/'PREPARED_SOURCE_ARCHIVE173.tar.gz'
 save(P/'ADMISSION_PROJECTION189.json',dict(schema='V189.sole-frozen173-production-parent.v1',files=rows,
  new_policy_sources={'admission189.py':sha(P/'admission189.py')},only_scheduling_change='original SLO_AWARE_ADMISSION branch0 to1',
  TARGET_SLO_FIRST=False,model_math_changed=False,device_qualification_inherited=False))
 inputs=read(OLD/'INPUTS173.json');external=dict(inputs['files'])
 for key in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):external.update(f[key])
 external.update({str(OLD/'CPU_RESULT173.json'):sha(OLD/'CPU_RESULT173.json'),str(OLD/'FROZEN_SOURCE173.json'):sha(OLD/'FROZEN_SOURCE173.json'),
  f['root_review']['path']:f['root_review']['sha256'],str(Path(__file__).resolve()):sha(__file__)})
 frozen177=D/'v177-root-safe-storage-overlay'
 for name in ('safe_storage177.py','CPU_RESULT177.json','FROZEN_OVERLAY177.json'):external[str(frozen177/name)]=sha(frozen177/name)
 manifest=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl');external[str(manifest)]=sha(manifest)
 native_root=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend')
 for path in (native_root/'spec_decode/pearl/native_engine.py',native_root/'envs.py'):external[str(path)]=sha(path)
 inputs.update(schema='V189.selected-source-policy-arm.v1',files=external,
  parent173=dict(path=str(OLD),CPU_sha256=sha(OLD/'CPU_RESULT173.json'),freeze_sha256=sha(OLD/'FROZEN_SOURCE173.json'),archive=dict(path=str(archive),sha256=sha(archive))),
  only_production_parent173=True,parent_device_qualification_inherited=False,old_parent_suites_rerun=False,
  SLO_AWARE_ADMISSION=True,TARGET_SLO_FIRST=False,coalesce_min_requests=1,coalesce_max_wait_ms=0.)
 save(P/'INPUTS189.json',inputs)
 print(json.dumps(dict(projected_sources=len(rows),local_files=len(rows)+3,external_files=len(external),projection_sha256=sha(P/'ADMISSION_PROJECTION189.json'),NPU_used=False,freeze_created=False)))
if __name__=='__main__':main()
