"""Thin source-only190+original-SLO-admission composition; no freeze/device."""
import ast,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-vacancy-cohort-prefill-cap4-active32-full-v190';P=B/'tree-slo-admission1-vacancy4-cap4-active32-full-v191';G=D/'v191-root-execution'
P189=B/'tree-slo-admission1-causal80-active32-full-v189'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def normalize(text):return text.replace(str(OLD),str(P)).replace(OLD.name,P.name).replace('190','191')
def main():
 assert not (P/'CPU_RESULT191.json').exists() and not (P/'FROZEN_SOURCE191.json').exists()
 f=read(OLD/'FROZEN_SOURCE190.json');c=read(OLD/'CPU_RESULT190.json');assert c['status']=='passed' and c['tests_run']==4 and c['local_before']==c['local_after']==f['files']
 omitted={'INPUTS190.json','PROJECTIONS190.json','check190.py','host_full190.py','test_full190.py','host_next4_child190.py'};rows={}
 for path,h in f['files'].items():
  src=Path(path);assert sha(src)==h
  if src.name in omitted:continue
  name=src.name.replace('190','191');text=normalize(src.read_text());changes=[]
  def change(old,new):
   nonlocal text
   assert text.count(old)==1,(name,old);text=text.replace(old,new);changes.append(dict(old=old,new=new,count=1))
  if name=='base191.py':change("ACTION='vacancy-cohort-prefill-cap4-active32-full463'","ACTION='slo-admission1-vacancy4-cap4-active32-full463'")
  if name=='runtime191.py':
   change('  require_ready(own_identity)','  require_ready(own_identity)\n  from admission191 import require_launch_environment,policy_report\n  require_launch_environment()')
   change('   origins_after=extension.loaded_origins()','   origins_after=extension.loaded_origins()\n   actual_admission_policy=policy_report(engine,native)')
   change('      overlay=overlay,old_21scope26phase_executed=False','      overlay=overlay,admission_policy=actual_admission_policy,old_21scope26phase_executed=False')
   change(' readiness=require_ready();spec=', ' from admission191 import require_launch_environment\n require_launch_environment()\n readiness=require_ready();spec=')
  if name=='audit191.py':
   change('from window160 import windows','from window160 import windows\nfrom admission191 import audit_trace as audit_admission_trace')
   change("  startup=r['startup'];counts=", "  require(r['admission_policy']['environment']=={'VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION':'1','VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST':'0'}\n   and r['admission_policy']['SLO_AWARE_ADMISSION'] is True and r['admission_policy']['TARGET_SLO_FIRST'] is False\n   and r['admission_policy']['coalesce_min_requests']==4 and r['admission_policy']['coalesce_max_wait_ms']==150.,'actual policy1/0 with vacancy4/150')\n  startup=r['startup'];counts=")
   change("   require(trace['admission_window160']==cap", "   audit_admission_trace(trace,len(source_rows))\n   require(trace['admission_window160']==cap")
  if name=='SCOPE191.txt':change(text,text+'\nV191 keeps frozen190 cap4/active32/vacancy4/150, C96/E96/global192, arrivals/clocks/output463x256 and actual same-engine gates. Only original SLO_AWARE_ADMISSION=1; TARGET_SLO_FIRST=0. Native priority order may change vacancy episode anchor and restart its150ms polling timer. Finite463 completion/no-drop required; no unbounded-flow fairness or latency/performance gain is claimed. Original190/185 suites are retained, not rerun.\n')
  target=P/name;target.write_text(text)
  if target.suffix=='.py':ast.parse(text)
  rows[name]=dict(parent=str(src),parent_sha256=h,namespace=[str(OLD),str(P)],basename_namespace=[OLD.name,P.name],epoch_namespace=['190','191'],changes=changes,sha256=sha(target))
 for oldname in ('guard190.py','launch190.sh','startup190.sh','GUARD_INPUTS190.json'):
  text=normalize((D/'v190-root-execution'/oldname).read_text());name=oldname.replace('190','191')
  text=text.replace('vacancy-cohort-prefill-cap4-active32-full463','slo-admission1-vacancy4-cap4-active32-full463')
  if name=='guard191.py':
   text=text.replace("c['tests_run']==4","c['tests_run']==3").replace('actual own4 HOST','actual own3 HOST')
   text=text.replace('old_21scope26phase_claimed=False,cost_measured=True,','old_21scope26phase_claimed=False,cost_measured=True,SLO_AWARE_ADMISSION=True,TARGET_SLO_FIRST=False,')
   text=text.replace("P173=require_parent(archive=True)","P190=require_parent(archive=True)")
  if name=='startup191.sh':
   old='export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=0 VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0'
   assert text.count(old)==1;text=text.replace(old,'export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1\nexport VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0')
   marker='export PYTHONDONTWRITEBYTECODE=1\n';text=text.replace(marker,marker+'export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1\nexport VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0\n',1)
   # One explicit setting before checker; remove the later original setting.
   second=text.find('export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1',text.index(marker)+len(marker)+1)
   assert second>=0;text=text[:second]+text[second:].replace('export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1\nexport VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0\n','',1)
  (G/name).write_text(text)
 external={}
 for key in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):external.update(f[key])
 external.update({str(OLD/'CPU_RESULT190.json'):sha(OLD/'CPU_RESULT190.json'),str(OLD/'FROZEN_SOURCE190.json'):sha(OLD/'FROZEN_SOURCE190.json'),f['root_review']['path']:f['root_review']['sha256'],str(Path(__file__).resolve()):sha(__file__)})
 for name in ('admission189.py','host189.py','CPU_RESULT189.json'):external[str(P189/name)]=sha(P189/name)
 for path in (Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl'),Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/envs.py')):external[str(path)]=sha(path)
 c189=read(P189/'CPU_RESULT189.json');assert c189['status']=='passed' and c189['tests_run']==3
 inputs=read(OLD/'INPUTS190.json');inputs.update(schema='V191.selected-source-composition.v1',files=external,P190=dict(path=str(OLD),CPU_sha256=sha(OLD/'CPU_RESULT190.json'),freeze_sha256=sha(OLD/'FROZEN_SOURCE190.json'),archive=dict(path=str(OLD/'PREPARED_SOURCE_ARCHIVE190.tar.gz'),sha256=sha(OLD/'PREPARED_SOURCE_ARCHIVE190.tar.gz'))),only_SLO_admission_flag_changed=True,current190_device_results_used=False,old190_4HOST_and185_suites_rerun=False)
 save(P/'INPUTS191.json',inputs);save(P/'PROJECTIONS191.json',dict(schema='V191.exact-frozen190-production-projection.v1',files=rows,only_production_schedule_change='original SLO_AWARE_ADMISSION=1, TARGET_SLO_FIRST remains0; vacancy4/150 retained',device_qualification_inherited=False))
 print(json.dumps(dict(projected_sources=len(rows),external=len(external),NPU_used=False,freeze_created=False)))
if __name__=='__main__':main()
