"""Freeze the HOST-tested new693 diagnostic; do not issue NN permission."""
from pathlib import Path
import datetime,hashlib,json,tarfile
P=Path(__file__).resolve().parent;B=P.parent;D=B/'budget-diagnostics'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
host=json.loads((P/'HOST459_CONSTRUCTION_RESULT.json').read_text())
assert host['status']=='PASS_HOST459_SYMBOLIC693_REFERENCE_SOURCE_AND_ACTUAL_LIFECYCLE'and host['NPU_used']is False
assert host['symbolic_schedules']==693 and host['actual_worker_predicates_exercised']==4
prior=json.loads((D/'ROOT458_RECONCILED457_WHOLE_MODEL_AUDIT_HOST.json').read_text())
assert prior['all3_eager_and_graph_math_exact']is True and prior['all4idle']is True and prior['original_supervisor_status']=='failed'and prior['NN_permission_issued']is False
files={str(p):sha(p)for p in P.iterdir()if p.is_file()and(p.suffix in('.py','.sh')or p.name in('PLAN459.json','HOST459_CONSTRUCTION_RESULT.json'))}
for path in[D/'ROOT458_RECONCILED457_WHOLE_MODEL_AUDIT_HOST.json',D/'ROOT458_ACTUAL_NATIVE_ORIGINAL_PROFILE_DIGEST_HOST.json',D/'ROOT458_ACTUAL438_SMALL_GEOMETRY_ELIGIBILITY_HOST.json']:
    files[str(path)]=sha(path)
freeze=dict(schema='ROOT459.same-physical-independent-node-source-freeze.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    parent457_freeze_sha256='6046ff35cab90a252560b2be0bb757b3395174c8fa5b6e33a4225afbb0966228',
    parent457_archive_sha256='e89132e0e60825586ce0bcd94bcf9fdb2f62ca92af45352de40c3541bfbd63d4',files=files,
    same_physical_reference_math_protocol_declared=True,cross_M120_equivalence_not_claimed=True,
    all_original_greedy_Native_and_bitwise_predicates_retained=True,global_active_ceiling=64,
    original_MC2_profile_unchanged=True,Native_flags_assigned=False,NN_permission_inherited=False,full463=False,Raw=None,Goodput=None)
frozen=P/'FREEZE459.json'
with frozen.open('x')as stream:json.dump(freeze,stream,indent=2);stream.write('\n')
frozen_sha=sha(frozen);members=dict(files);members[str(frozen)]=frozen_sha
archive=P/'SOURCE459.tar.gz'
with tarfile.open(archive,'x:gz')as stream:
    for name in sorted(members):stream.add(name,arcname=name.lstrip('/'),recursive=False)
receipt=dict(freeze_sha256=frozen_sha,archive_sha256=sha(archive),own_members=len(members),source_manifest=str(frozen),source_archive=str(archive),NPU_launched=False,NN_permission=False,Raw=None,Goodput=None)
with(P/'SOURCE459_RECEIPT.json').open('x')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt))
