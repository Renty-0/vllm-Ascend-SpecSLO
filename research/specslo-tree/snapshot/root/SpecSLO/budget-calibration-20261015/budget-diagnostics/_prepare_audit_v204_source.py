"""SOURCE preparation only. Does not import/execute an auditor or read actual."""
from pathlib import Path
import ast,copy,hashlib,json,re
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=D/'audit_v201_terminal_HOST_revision2.py';OUT=D/'audit_v204_terminal_HOST.py'
OLD_SHA='1c3eb7aafb01e346c43d9d76e750e7448ce910a69a52ef3595a479ff7da56729'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()==OLD_SHA
old=OLD.read_text();tokens={}
def protect(match):
 key='__OPAQUE_SHA_'+str(len(tokens))+'__';tokens[key]=match.group();return key
ns=re.sub(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',protect,old).replace('201','204')
for key,value in tokens.items():ns=ns.replace(key,value)
ns=ns.replace('Original204/44/114','Original201/44/114')
ns=ns.replace('tree-normal-b4-final-kv5-cohort16-full-v204','tree-normal-b4-final-kv5-single-writer-full-v204')
ns=ns.replace("ACTION='normal-b4-final-kv5-cohort16-full463'","ACTION='normal-b4-final-kv5-single-writer-full463'")
ns=ns.replace("READY=D/'V204_FINAL_KV5_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW_REVISION2.json'","READY=D/'V204_SINGLE_WRITER_FINAL_KV5_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'")
for name,value in dict(FREEZE_SHA='ROOT-FREEZE-PIN-PENDING',CPU_SHA='e4f53a661bc2ed8bdd0b54e8891549a2183a64c855e70861982b4363f75fe7ef',ARCHIVE_SHA='ROOT-ARCHIVE-PIN-PENDING',ROOT_PREPARED_SHA='ROOT-PREPARED-PIN-PENDING',READY_SHA='7661a8fcc3ca6e164464655f80d32f76c7087db32b0fa90bf1fabb6819e084a4').items():
 ns=re.sub(r'^'+name+r'=.*$',name+'='+repr(value),ns,flags=re.M)
ns=ns.replace("NEW_KEY=('draft-greedy:151936|steps:5|paged|final-kv|confidence',16)","NEW_KEY=('draft-greedy:151936|steps:5|paged|final-kv|confidence',16)\nSOURCE_PARENT=B/'tree-normal-b4-final-kv5-cohort16-full-v201'\nSOURCE_PARENT_FREEZE_SHA='035cf69b61bcf6871baa472d5f9eb12001454e2ba076de07baf025d703ee460b'\nSOURCE_PARENT_CPU_SHA='38f406d674370d1de61658a08c2508e01b943a78871d87834214b1b654b92c15'\nSOURCE_PARENT_ARCHIVE_SHA='edbb8cb19e3a851331fdbbba10ed54e37f99250e838584429ce5d5eeda7e36c8'")
new_provenance='''def provenance(receipt):
 """Own204 six-HOST schema plus exact frozen201 source-only derivation."""
 freeze_path=P/'FROZEN_SOURCE204.json';cpu_path=P/'CPU_RESULT204.json'
 need(all(re.fullmatch(r'[0-9a-f]{64}',h)is not None for h in(FREEZE_SHA,ARCHIVE_SHA,ROOT_PREPARED_SHA)),
      'ROOT exact freeze/archive/prepared pins required; pending source cannot audit actual')
 for path,h in((freeze_path,FREEZE_SHA),(cpu_path,CPU_SHA),(ROOT_PREPARED,ROOT_PREPARED_SHA),(READY,READY_SHA),(CLOCKS,CLOCK_SHA),(STAT_SOURCE,STAT_SHA)):
  need(sha(path)==h,'exact204 source/preparation/helper SHA '+str(path))
 f=read(freeze_path);c=read(cpu_path);prepared=read(P/'PREPARED_HOST204.json')
 need(f['schema']=='V204.ROOT-source-freeze.v1'and f['ROOT_source_reviewed']is True,'own ROOT204 source freeze')
 pins=merge_pins(*(f[k]for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews')),
  {str(cpu_path):f['CPU_sha256'],str(freeze_path):FREEZE_SHA,f['root_review']['path']:f['root_review']['sha256']})
 need(len(pins)==315 and all(sha(path)==h for path,h in pins.items()),'all315 current selected source/CPU/guard/peer bytes')
 need(c['schema']=='V204.full-source-HOST.v1'and c['status']=='passed'and c['tests_run']==6 and c['failures']==c['errors']==0,
      'actual current six-check source run, not inherited201 five-check permission')
 need(all(c[k]is False for k in('Torch_imported','NPU_used','model_constructed','model_math_qualified','old_suites_rerun','full463_executed','freeze_created')),
      'current source-only HOST backend/math fields')
 need(c['local_before']==c['local_after']==f['files']==prepared['files']and
      c['external_before']==c['external_after']==f['external_files']==prepared['external_files']and
      f['CPU_sha256']==prepared['CPU_sha256']==CPU_SHA and prepared['current_actual_tests_run']==6 and
      prepared['NPU_used']is prepared['freeze_created']is prepared['parent_suites_rerun']is False,'current204 actual source/CPU/prepared/freeze identity')
 permit=c['details']['permit_identity']
 need(all(permit[k]is True for k in('actual_P148_checker','actual_activate_and_bounded_scope','actual_runtime_spec_assignment','wrong197identity_rejected'))and
      permit['model_backend_imported']is False,'same actual original P148 permit positive/foreign197 negative evidence')
 writer=c['details']['single_writer_directory']
 need(all(writer[k]is True for k in('actual_actor_lambda','actual_original_stage4','real_four_process_shared_FS','HOST_collective_and_sync_carriers'))and
      all(writer[k]is False for k in('NPU_used','Torch_library_imported','model_constructed')),'actual four-process directory source-only control proof')
 need(writer['cases']==[dict(all4_passed=False,case='old-race',common_all4_result_identical=True,local_FileExists_errors=3),
  dict(all4_passed=True,case='new-single-writer',common_all4_result_identical=True,local_FileExists_errors=0),
  dict(all4_passed=False,case='new-preexisting-refused',common_all4_result_identical=True,local_FileExists_errors=1)],
  'actual old3FileExists/newpass/preexisting leader refusal plus common all4 status')
 attempts=prepared['actual_HOST_attempts'];retained=prepared['retained_HOST_files']
 need(set(attempts)=={str(p)for p in P.glob('HOST_ATTEMPT204_*.json')}and len(attempts)==1 and len(retained)==30,'one actual current HOST attempt and30 snapshots/log/results')
 evidence=merge_pins({str(P/'PREPARED_HOST204.json'):sha(P/'PREPARED_HOST204.json')},attempts,retained)
 need(evidence==f['retained_HOST_evidence']and len(evidence)==32,'exact current204 source evidence map')
 root=read(f['root_review']['path'])
 need(root['status']=='passed_source_review'and root['reviewed_inputs']=={path:h for path,h in pins.items()if path not in(str(freeze_path),f['root_review']['path'])},
      'ROOT exact313 current source/CPU/guard/peer reviewed inputs')
 for path in f['peer_reviews']:
  peer=read(path);need(peer['status'].replace('-','_')=='passed_source_review'and peer['candidate']==str(P),'source peer current candidate')
 expected=dict(active_windows_requested=[32],configured_capacity=64,queued=463,storage=527,
  empty_active_only=True,current_multiowner_eager_math=True,new_target_keys=2,causal80_domain=[16,80,'causal'],original_parent173_cap4=True,
  new_final_KV_draft_keys=1,original_BONUS_g_target_protocol_unchanged=True,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
 need(all(f[k]==value and receipt[k]==value for k,value in expected.items()),'unchanged real fifthKV scope/current original target BONUS/g/clock bounds')
 readiness=receipt['readiness']
 need(readiness['freeze_sha256']==FREEZE_SHA and readiness['CPU_sha256']==CPU_SHA and readiness['root_review']==f['root_review']and
      readiness['peers']==f['peer_reviews']and readiness['device_qualification_inherited']is False,'actual own204 receipt readiness')
 ar=readiness['archive'];need(Path(ar['path'])==P/'PREPARED_SOURCE_ARCHIVE204.tar.gz'and ar['members']==315 and sha(ar['path'])==ar['sha256']==ARCHIVE_SHA,'own315 source archive SHA')
 own_archive=archive(ar['path'],pins)
 # Explicit independent donor bytes/archive, not just accepting the original
 # require_source_parent return as a recomputed source proof.
 pin=read(P/'INPUTS204.json')['source_parent201'];donor_freeze=SOURCE_PARENT/'FROZEN_SOURCE201.json'
 need(sha(donor_freeze)==pin['freeze_sha256']==SOURCE_PARENT_FREEZE_SHA and sha(SOURCE_PARENT/'CPU_RESULT201.json')==SOURCE_PARENT_CPU_SHA,
      'original201 immutable source-donor freeze/CPU exact')
 df=read(donor_freeze);donor_pins=merge_pins(*(df[k]for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews')),
  {str(SOURCE_PARENT/'CPU_RESULT201.json'):df['CPU_sha256'],str(donor_freeze):SOURCE_PARENT_FREEZE_SHA,df['root_review']['path']:df['root_review']['sha256']})
 need(len(donor_pins)==531 and all(sha(path)==h for path,h in donor_pins.items()),'all531 immutable201 source-donor bytes')
 da=pin['archive'];need(Path(da['path'])==SOURCE_PARENT/'PREPARED_SOURCE_ARCHIVE201.tar.gz'and sha(da['path'])==da['sha256']==SOURCE_PARENT_ARCHIVE_SHA,'source-donor201 archive SHA in place')
 donor_archive=archive(da['path'],donor_pins)
 dc=read(SOURCE_PARENT/'CPU_RESULT201.json')
 need(dc['status']=='passed'and dc['tests_run']==5 and dc['failures']==dc['errors']==0 and dc['Torch_imported']is dc['NPU_used']is False and
      dc['local_before']==dc['local_after']==df['files']and dc['external_before']==dc['external_after']==df['external_files'],
      'source donor five-HOST source proof does not become current204 six-HOST or device qualification')
 previous=read(SOURCE_PARENT/'INPUTS201.json')['previous_revision'];old=read(previous['manifest']['path'])
 need(previous['manifest']['sha256']=='befb26b67a452a10fea94e49df5fe0924e38cfedfae33ad5b8c28990b3334229'and
      sha(previous['manifest']['path'])==previous['manifest']['sha256']and len(old['files'])==218,'all218 older pre-permit-revision source preservation')
 for original,row in old['files'].items():need(sha(row['snapshot'])==row['sha256'],'preserved historical byte '+original)
 prior=read(previous['CPU']['path'])
 need(sha(previous['CPU']['path'])==previous['CPU']['sha256']=='b03c3cbb61c0a5e954d49016a8dac7a815db663e6a935bb89a55545495ce9046'and
      prior['status']=='passed'and prior['tests_run']==4 and prior['failures']==prior['errors']==0 and prior['local_before']==prior['local_after']and
      prior['external_before']==prior['external_after'],'preserved earlier four-HOST revision remains separate historical source evidence')
 projection=read(P/'PROJECTIONS204.json')['projection'];need(len(projection)==21,'actual21 whole namespace projections')
 for row in projection:
  path=Path(row['parent']);need(sha(path)==row['parent_sha256']and path.is_relative_to(SOURCE_PARENT),'whole source-donor201 origin')
  text=path.read_text()
  for a,b,count in row['namespace']:need(text.count(a)==count,'exact namespace anchor');text=text.replace(a,b)
  parts=[];offset=0
  for op in row['deltas']:
   need(op['start']>=offset and text[op['start']:op['end']]==op['old'],'exact projected source delta');parts.extend((text[offset:op['start']],op['new']));offset=op['end']
  need(''.join(parts)+text[offset:]==(P/row['file']).read_text(),'whole projected204 current source '+row['file'])
 import base204,check204
 deps=check204.require_parent(archive=True);parent=base204.parent_ready();old_proof=check204.previous_HOST()
 need(deps==f['dependency_archives_verified_in_place']==readiness['dependency_archives']and
      json.loads(json.dumps(parent,sort_keys=True,allow_nan=False))==readiness['parent_ready'],
      'actual P197 production lineage/original P148 permit; JSON canonical tuple/list equality')
 need(deps['source_derivation201']['members']==531 and deps['source_derivation201']['archive_verified_in_place']is True and
      deps['source_derivation201']['device_qualification_inherited']is False and old_proof['tests_run']==4 and old_proof['mathematical_source_unchanged']is True,
      'source donor and old HOST record never issue current device permission')
 safe=D/'v177-root-safe-storage-overlay/safe_storage177.py';need(f['external_files'][str(safe)]==sha(safe),'frozen177 storage race fix')
 cleanup=terminal_cleanup(receipt)
 return dict(freeze=desc(freeze_path),CPU=desc(cpu_path),ROOT_prepared=desc(ROOT_PREPARED),source_ready=desc(READY),
  selected_current_SHA_checked=315,current_HOST_tests=6,current_HOST_attempts=1,source_projection_rows=21,single_writer_HOST_cases=writer['cases'],
  archive=own_archive,source_donor201_archive=donor_archive,source_donor_current_HOST_tests=5,previous_revision_preserved_files=218,previous_HOST_tests=4,
  historical_tests_not_added_to_current=True,parent_archives=deps,original_P148_permit_identity=parent,
  saved_baseline_and_cleanup=cleanup,complete_environment_import_closure_claimed=False)
'''
tree=ast.parse(ns);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='provenance')
lines=ns.splitlines(keepends=True);ns=''.join(lines[:node.lineno-1])+new_provenance+'\n'+''.join(lines[node.end_lineno:])
# The201 source manifest is a newly explicit read-only donor origin. No math
# or full metrics function changes are needed for the single-writer repair.
ns=ns.replace("for path in (B/'tree-empty-active-cohort-causal80-active32-full-v197/FROZEN_SOURCE197.json',", "for path in (SOURCE_PARENT/'FROZEN_SOURCE201.json',\n              B/'tree-empty-active-cohort-causal80-active32-full-v197/FROZEN_SOURCE197.json',")
compile(ast.parse(ns),str(OUT),'exec')
assert not OUT.exists();OUT.write_text(ns)
# Retain the exact namespace reference for later per-function inverse proof.
REFERENCE=D/'V204_AUDITOR_NAMESPACE_REFERENCE_SOURCE.txt';assert not REFERENCE.exists()
REFERENCE.write_text(ns)
print(json.dumps(dict(script=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),SOURCE_ONLY=True,executed=False,pending_ROOT_freeze_pins=True)))
