"""Supplemental read-only archive peer; no candidate import/backend/test suite."""
import ast,datetime,hashlib,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-live-small-mixed197-cap64-active32-full-v192'
A=D/'ROOT_V192_PARENT163_172_ARCHIVES_IN_PLACE_AUDIT_REVISION2.json';S=D/'audit_v192_parent_archives_ROOT_revision2.py';OLD_A=D/'ROOT_V192_PARENT163_172_ARCHIVES_IN_PLACE_AUDIT.json';OLD_S=D/'audit_v192_parent_archives_ROOT.py';REJECT=D/'ROOT_V192_PARENT_ARCHIVE_AUDIT_REV1_REJECTION.json';PEER=D/'V192_INDEPENDENT_SOURCE_PEER_REVIEW.json';OUT=D/'V192_PARENT_ARCHIVE_REVISION2_INDEPENDENT_REVIEW.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert sha(A)=='bebb8bb1bd94ff92ef858b5d11cfd62778601e9abe7801775a7e2ede3cbf0f01'
assert sha(S)=='a6b0ab2874092b57d576f36b93ec950e68da6b135c976ed2111408694dbb9006'
assert sha(REJECT)=='5adb109fb1fdc50003a395c16e7d453d8b186c403e8ade650fca130eb2f0c5d3'
assert sha(OLD_A)=='9d458931c99a9e20b28e6f64627e94c30b9a925c0102ef9c0b4efff654a93bcc'and sha(PEER)=='0ed1f638a1cb75359008d8111468b4bdf1a6cfaee153086db90569542a89cc27'
old=OLD_S.read_text();new=S.read_text();expected=old.replace("OUT = D / 'ROOT_V192_PARENT163_172_ARCHIVES_IN_PLACE_AUDIT.json'","OUT = D / 'ROOT_V192_PARENT163_172_ARCHIVES_IN_PLACE_AUDIT_REVISION2.json'").replace('expected_member_map=wanted,','expected_member_map=dict(wanted),')
assert expected==new and old.count('expected_member_map=wanted,')==1
compile(ast.parse(new),str(S),'exec')
a=read(A);assert a['status']=='passed'and a['candidate']==str(P)and a['Torch_imported']is a['NPU_used']is a['model_initialized']is a['source_changed']is False
assert a['script']==dict(path=str(S),sha256=sha(S))and a['old_suites_not_rerun']is True
assert read(REJECT)['status']=='rejected_before_ROOT_review_or_freeze'
inputs=read(P/'SOURCE_INPUTS192.json');descriptor=read(a['base_descriptor']['path']);assert sha(a['base_descriptor']['path'])==a['base_descriptor']['sha256']
assert sha(P/'SOURCE_INPUTS192.json')==a['source_inputs']['sha256']
base={'/'+n.lstrip('/'):h for n,h in descriptor['members'].items()};assert len(base)==6390
assert a['base_archive']['expected_member_map']==base and a['base_archive']['regular_members']==len(base)
archive_results=[]
def check_archive(row,wanted):
 assert row['expected_member_map']==wanted and row['regular_members']==len(wanted) and row['all_member_SHA256_checked']is True and row['archive_recopied']is False
 assert sha(row['path'])==row['sha256']
 with tarfile.open(row['path'],'r:gz')as t:
  entries=t.getmembers();assert len(entries)==len(wanted)==len({v.name for v in entries})
  actual={}
  for entry in entries:
   name='/'+entry.name.lstrip('/');assert entry.isfile()and name in wanted
   h=hashlib.sha256(t.extractfile(entry).read()).hexdigest();assert h==wanted[name];actual[name]=h
  assert actual==wanted
 archive_results.append(dict(path=row['path'],sha256=row['sha256'],regular_members=len(wanted),expected_map_SHA256=hashlib.sha256(json.dumps(wanted,sort_keys=True,separators=(',',':')).encode()).hexdigest(),exact_member_set_and_SHA256=True))
check_archive(a['base_archive'],base)
archived=dict(base)
for epoch,row in zip((163,172),a['parent_archives'],strict=True):
 assert row['epoch']==epoch
 pin=inputs['original_P'+str(epoch)];root=Path(pin['path']);fpath=root/f'FROZEN_SOURCE{epoch}.json';cp=root/f'CPU_RESULT{epoch}.json';f=read(fpath)
 assert sha(fpath)==pin['freeze_sha256']==row['freeze']['sha256']and sha(cp)==pin['CPU_sha256']==row['CPU']['sha256']
 selected={}
 for name in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
  for path,h in f[name].items():assert path not in selected or selected[path]==h;selected[path]=h
 for path,h in {str(cp):f['CPU_sha256'],str(fpath):pin['freeze_sha256'],f['root_review']['path']:f['root_review']['sha256']}.items():
  assert path not in selected or selected[path]==h;selected[path]=h
 assert len(selected)==row['selected_current_sources']
 for path,h in selected.items():assert Path(path).is_file()and not Path(path).is_symlink()and sha(path)==h
 wanted={p:h for p,h in selected.items()if archived.get(p)!=h};check_archive(row,wanted);archived.update(selected)
# The serialized base snapshot remains the original6390 after all later updates.
assert a['base_archive']['expected_member_map']==base and len(a['base_archive']['expected_member_map'])==a['base_archive']['regular_members']==6390
assert [r['regular_members']for r in archive_results]==[6390,94,45]
row=dict(schema='V192.supplemental-parent-archive-revision2-source-peer.v1',status='passed_source_review',candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revised_archive_audit=dict(path=str(A),sha256=sha(A)),revised_script=dict(path=str(S),sha256=sha(S)),rejected_prior_audit=dict(path=str(OLD_A),sha256=sha(OLD_A)),ROOT_rejection=dict(path=str(REJECT),sha256=sha(REJECT)),original_source_peer=dict(path=str(PEER),sha256=sha(PEER)),
 reviewed_source_delta=dict(only_changes=['fresh REVISION2 output path','expected_member_map=dict(wanted) shallow immutable-value snapshot'],inverse_to_original_exact=True,caller_archived_mutation_cannot_change_saved_expected_map=True),independent_actual_archive_checks=archive_results,
 closure='This supplement supersedes only the original peer archive-evidence reference to rejected revision1. All three original archives were independently reopened, exact regular member sets and every uncompressed member SHA rechecked, selected parent current sources rehashed. No candidate production/HOST source changed.',ROOT_freeze_obligation='Bind both peer artifacts, full revised audit and script/rejection SHAs in the explicit ROOT review; actual warm schedule remains one warm65/two zero65/seal/full463, not max8.',no_reviewed_remaining_archive_blocker=True,
 no_old_suite_rerun=True,candidate_imported=False,Torch_imported=False,NPU_used=False,device_queried=False,model_initialized=False,candidate_source_modified=False,original_peer_modified=False,freeze_created=False,actual_model_qualification_inherited=False,formal_Goodput=False)
assert not OUT.exists();OUT.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),bytes=OUT.stat().st_size,regular_members=[r['regular_members']for r in archive_results])))
