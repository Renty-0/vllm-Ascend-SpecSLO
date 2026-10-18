"""Close actual21 HOST +28 CPU evidence and new shadow-qualified source."""
import ast,datetime,hashlib,json,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-packed-KV-preflight-with-own-FIA-AR-full-v217'
DONOR=B/'tree-cost-aware-N9to16-with-own-FIA-AR-full-v216'
assert not(P/'FREEZE217.json').exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
host=D/'ROOT_V217_PACKED_MAPPING_SOURCE_AND_HOST_PREPARATION.json';evidence=read(host)
assert evidence['status']=='passed'and evidence['total_tests']==21 and evidence['Torch_imported']is evidence['NPU_used']is False
meta=read(P/'METADATA_ONLY_AFTER_HOST217.json')
for path,digest in evidence['source_sha256'].items():
    if path==meta['path']:
        assert digest==meta['before_sha256']and sha(path)==meta['after_sha256']
        saved=next(p for p in (P/'HOST-attempts').glob('*/guard217.py')if sha(p)==digest)
        assert saved.read_text().replace(meta['exact_old'],meta['exact_new'])==Path(path).read_text()
    else:assert sha(path)==digest,path
cpu=evidence['controlled_Torch_CPU'];assert sha(cpu['path'])==cpu['sha256']
actual=read(cpu['path']);assert actual['status']=='passed'and actual['cases']==28 and len(actual['negative_cases'])==3
assert actual['Torch_imported']is True and actual['NPU_used']is actual['Torch_npu_imported']is actual['model_constructed']is False
for path,digest in actual['source_sha256'].items():assert sha(path)==digest
for p in P.glob('*.py'):ast.parse(p.read_text())
donor=read(DONOR/'FREEZE216.json');files=dict(donor['files']);files[str(DONOR/'FREEZE216.json')]=sha(DONOR/'FREEZE216.json')
for p in P.rglob('*'):
    if p.is_file()and'__pycache__'not in p.parts:files[str(p)]=sha(p)
for p in(D/'prepare_ROOT217_source.py',Path(__file__),host):files[str(p)]=sha(p)
review=dict(schema='ROOT217.packed-target-preflight-source-review.v1',UTC=now,status='passed_source_review',
    source_peer_review=False,reviewed_inputs=files.copy(),actual21HOST=evidence,
    actual_Torch_CPU_control=cpu,metadata_only_after_HOST=meta,
    source_delta='one target-only slot_values expression: actual original packed target mapping .detach.cpu.tolist instead of per-row .to(long) and torch.cat',
    preserved=['Original canonical-eager transform replayed from frozen source; actual installed method deep-code fingerprint must match before binding',
      'All original bounds/dtype/device/packed-count/slot-alias/payload/deferred-eager/capacity checks remain bytebody-equivalent',
      'Original/new actual current-engine component calls shadowed and require identical complete HOST mapping and pending device view pointers before warm',
      'Draft mapping and staged-eager route unchanged; global structural/numerical consensus unchanged',
      'Current96/96/global192 budget and cost-aware9..16 continue; full original model qualification/warm/twozero/seal/full463 required',
      'Postfull independent463 ownFIA80 AR remains outside performance interval'],
    frozen_native_files_modified=False,model_precision_changed=False,new_Native_graph_keys=0,
    device_shadow_qualified=False,NPU_run=False,Goodput=None,adopted=False)
rp=P/'ROOT_SOURCE_REVIEW217.json';save(rp,review);files[str(rp)]=sha(rp)
freeze=dict(schema='ROOT217.packed-target-preflight-source-freeze.v1',UTC=now,files=files,
    source_review=dict(path=str(rp),sha256=sha(rp)),original209_source_only_dependency=donor['original209_source_only_dependency'],
    source_donor216=dict(path=str(DONOR),freeze_sha256=sha(DONOR/'FREEZE216.json'),device_qualification_inherited=False),
    HOST_tests=21,Torch_CPU_cases=28,Torch_CPU_negative_cases=3,source_peer_review=False,SLO_AWARE_ADMISSION=1,
    budget_grants_may_change=True,packed_target_KV_preflight=True,device_shadow_qualified=False,
    full463_required=True,own_AR_device_qualified=False,Goodput=None,adopted=False)
fp=P/'FREEZE217.json';save(fp,freeze);archived=dict(files);archived[str(fp)]=sha(fp);ar=P/'SOURCE217.tar.gz'
with tarfile.open(ar,'w:gz')as saved:
    for p in sorted(archived):saved.add(p,arcname=p.lstrip('/'),recursive=False)
with tarfile.open(ar,'r:gz')as saved:
    members=saved.getmembers();assert len(members)==len(archived)
    for m in members:assert m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==archived['/'+m.name]
row=dict(schema='ROOT217.frozen-unrun-source.v1',UTC=now,directory=str(P),freeze_sha256=sha(fp),archive_sha256=sha(ar),
    archive_members=len(archived),archive_bytes=ar.stat().st_size,HOST_tests=21,Torch_CPU_cases=28,Torch_CPU_negative_cases=3,
    source_peer_review=False,NPU_started=False,device_shadow_qualified=False,own_AR_device_qualified=False,Goodput=None,adopted=False)
save(D/'ROOT_V217_SOURCE_FROZEN_UNRUN.json',row);print(json.dumps(row))
