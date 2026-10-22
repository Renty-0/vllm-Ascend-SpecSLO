"""Freeze only new438 sources after actual CPU policy and API checks."""
from pathlib import Path
import datetime,hashlib,json,re,tarfile
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-slo-aware-refill-complete-interface-full-v438';OLD=B/'tree-refill-witness-boolean-retention-full-v437'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rename(t):
 t=re.sub(r'[0-9a-f]{64}|(?<!\d)437(?!\d)',lambda m:'438'if m.group()=='437'else m.group(),t)
 return t.replace(str(OLD).replace('437','438'),str(P))
assert not(P/'FREEZE438.json').exists()
plan=json.loads((P/'PLAN438.json').read_text());plan.pop('only_neural_callbacks_change');(P/'PLAN438.json').write_text(json.dumps(plan,indent=2)+'\n')
for name in ['HOST438_POLICY_RESULT.json','HOST438_COMPLETE_INTERFACE_RESULT.json','HOST438_CLEAN_STARTUP_RUN.json']:
 r=json.loads((P/name).read_text());assert r['status'].startswith('PASS_HOST438')and r['NPU_used']is False
for old,new in [('check_ROOT437_frozen_parent_ready_HOST.py','check_ROOT438_frozen_parent_ready_HOST.py'),('check_ROOT437_actual_preconstructor_HOST.py','check_ROOT438_actual_preconstructor_HOST.py')]:
 t=rename((D/old).read_text())
 if 'preconstructor' in new:
  anchor="        assert result['compiled_admission_has_witness431'] is True"
  t=t.replace(anchor,anchor+"\n        assert 'admission438' in nested[0].co_names\n        import inspect\n        attrs={n.attr for n in ast.walk(ast.parse(runtime438.source()))if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='extension'}\n        assert all(hasattr(extension438,name)for name in attrs)\n        from extension433 import preflight_snapshot\n        assert extension438.preflight_snapshot is preflight_snapshot\n        result['all_actual_worker_extension_attributes_present']=sorted(attrs)\n        result['compiled_admission_has_current_SLO_policy']=True")
  t=t.replace('import witness438','import witness437').replace('witness438.snapshot','witness437.snapshot')
 (D/new).write_text(t)
files=sorted(p for p in P.iterdir()if p.is_file())+[D/'prepare_ROOT438_slo_refill.py',D/'run_ROOT438_actual_CPU_HOST.py',Path(__file__).resolve()]
assert len(files)==len(set(files))
for p in files:
 if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
freeze=dict(schema='ROOT438.current-SLO-aware-refill-complete-worker-interface.v1',files={str(p):sha(p)for p in sorted(files)},parent437_freeze=sha(OLD/'FREEZE437.json'),parent437_archive=sha(OLD/'SOURCE437.tar.gz'),global_active_limit=64,NN_permission_inherited=False,Native_flags_assigned=False,private_tag_MiB=256,aggregate_private_MiB=3584,runtime_seconds=18000,Raw=None,Goodput=None)
(P/'FREEZE438.json').write_text(json.dumps(freeze,indent=2)+'\n')
archive=P/'SOURCE438.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6)as t:
 for p in sorted([*files,P/'FREEZE438.json']):t.add(p,arcname=str(p).lstrip('/'),recursive=False)
with tarfile.open(archive,'r:gz')as t:
 members=t.getmembers();expected={str(p).lstrip('/'):sha(p)for p in [*files,P/'FREEZE438.json']}
 assert len(members)==len(expected)
 for m in members:assert m.isfile()and hashlib.sha256(t.extractfile(m).read()).hexdigest()==expected[m.name]
record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=sha(P/'FREEZE438.json'),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=len(members),NN_permission_inherited=False)
(P/'SOURCE438_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
# Independently bound terminal readers, outside frozen candidate sources.
t=rename((D/'audit_ROOT437_terminal_full_HOST.py').read_text())
t=t.replace("FREEZE = '8ecd9f97279d1ae6023e4ca14188954b34575b26940ec64d1848c8b93a99f8d8'",f"FREEZE = '{record['freeze_sha256']}'")
t=t.replace("ARCHIVE = '1553265a38fa19caf8bcb32067e9e921627f97f9c892429d68bd891e97aa366e'",f"ARCHIVE = '{record['archive_sha256']}'")
t=t.replace("from admission_audit432 import audit_trace","from admission438 import audit_trace")
(D/'audit_ROOT438_terminal_full_HOST.py').write_text(t)
(D/'read_ROOT438_common_result_HOST.py').write_text(rename((D/'read_ROOT437_common_result_HOST.py').read_text()))
t=rename((D/'check_ROOT437_terminal_admission_HOST.py').read_text())
t=t.replace("for folder in ('tree-nonempty-current-incumbent-full-proposal-v432',","for folder in (P.name, 'tree-nonempty-current-incumbent-full-proposal-v432',")
t=t.replace("trace = dict(scope_count=16,", "trace = dict(sampling=[dict(slo_tpot_ms=40.)for _ in range(16)],scope_count=16,")
# Terminal projection now enforces actual current-SLO membership, above original clocks.
t=t.replace("'positive_nonempty_cohorts=1'", "'positive_nonempty_cohorts=1'")
(D/'check_ROOT438_terminal_admission_HOST.py').write_text(t)
print(json.dumps(record))
