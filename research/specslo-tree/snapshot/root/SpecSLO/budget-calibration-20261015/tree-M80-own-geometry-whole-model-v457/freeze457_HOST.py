"""Freeze a local diagnostic after actual HOST construction, no NPU launch."""
from pathlib import Path
import datetime,hashlib,json,tarfile
P=Path(__file__).resolve().parent;B=P.parent;D=B/'budget-diagnostics'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
host=json.loads((P/'HOST457_CONSTRUCTION_RESULT.json').read_text())
assert host['status']=='PASS_HOST457_ACTUAL385_NAMESPACE_SOURCE_PARSER_AND_SETTINGS'and host['NPU_used']is False
prior=json.loads((D/'ROOT455_TERMINAL_INDEPENDENT_WHOLE_MODEL_AUDIT_HOST.json').read_text())
assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_eager_and_graph_math_exact']is True
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
files={str(p):sha(p)for p in P.iterdir()if p.is_file()and(p.suffix in('.py','.sh')or p.name in('PLAN457.json','HOST457_CONSTRUCTION_RESULT.json'))}
for p in[D/'ROOT455_TERMINAL_INDEPENDENT_WHOLE_MODEL_AUDIT_HOST.json',D/'ROOT378_NEGATIVE_FIRST_MODEL_CASE.json',D/'ROOT379_REAL_FIRST_QKV_DIVERGENCE_RESULT.json']:
 files[str(p)]=sha(p)
freeze=dict(schema='ROOT457.own80-source-freeze.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 parent455_original_export_sha256='f55502f11b1636893ab6cc47b2442aaebc34d9d66f2d2795418120464a83a435',files=files,
 global_active_ceiling=64,physical_M=[80,120],original_MC2_profile_retained=True,
 cross_M120_discrepancy_not_qualified=True,NN_permission=False,Raw=None,Goodput=None)
path=P/'FREEZE457.json'
with path.open('x')as f:json.dump(freeze,f,indent=2);f.write('\n')
frozen=sha(path);members=dict(files);members[str(path)]=frozen
archive=P/'SOURCE457.tar.gz'
with tarfile.open(archive,'x:gz')as stream:
 for name in sorted(members):stream.add(name,arcname=name.lstrip('/'),recursive=False)
receipt=dict(freeze_sha256=frozen,archive_sha256=sha(archive),own_members=len(members),
 source_manifest=str(path),source_archive=str(archive),NPU_launched=False,NN_permission=False,Raw=None,Goodput=None)
with(P/'SOURCE457_RECEIPT.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
