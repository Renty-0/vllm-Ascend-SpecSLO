import ast,datetime,gzip,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';OLD=B/'tree-contiguous-branch-tail-M80-math-v233';P=B/'tree-contiguous-branch-tail-final-metadata-math-v234'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
cpu=json.loads((P/'CPU_BRANCH_TAIL234_RESULT.json').read_text())
assert cpu['status']=='CPU_exact_visible_KV_order_and_Native_ABI_passed'and cpu['cases']==203 and cpu['branch_cases']==164 and cpu['max_actual_copy_slots']<=1792 and not cpu['NPU_used']
for path,digest in cpu['files_sha256'].items():assert sha(path)==digest
meta=json.loads((P/'CPU_METADATA_STATE231_RESULT.json').read_text());assert meta['status']=='CPU_State_metadata_equals_actual_Native_model_input_passed'and meta['post_Native_padding_metadata_identity_checked']and not meta['NPU_used']
assert(P/'FIXTURE234.json').read_bytes()==(OLD/'FIXTURE233.json').read_bytes()
files=json.loads((OLD/'FREEZE233.json').read_text())['files'];assert all(sha(p)==h for p,h in files.items())
files[str(OLD/'FREEZE233.json')]=sha(OLD/'FREEZE233.json')
extras=[D/'prepare_ROOT234_metadata_alignment.py',D/'ROOT_V233_TERMINAL_SAVED_METADATA_SETUP_FAILURE.json',Path(__file__)]
for p in [*P.iterdir(),*extras]:
    if p.is_file()and p.suffix in('.py','.sh','.json'):
        if p.suffix=='.py':ast.parse(p.read_text())
        files[str(p)]=sha(p)
for p in P.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
sys.path[:0]=[str(P),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),str(B/'tree-high-normal-cap4-native-families-v148')]
from neural_oracle231 import cases
assert len(cases())==155 and {v['S']for v in cases()}=={14,15,16,17,18,22}
save(P/'FREEZE234.json',dict(schema='ROOT234.contiguous-branch-tail-neural-math.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
    scope='155 independent node math cases; constantM80; side-query contiguous scratch KV, original BF16 weights/greedy; old-layout negative control; no performance or full workload qualification'))
files=dict(files);files[str(P/'FREEZE234.json')]=sha(P/'FREEZE234.json')
with(P/'SOURCE234.tar.gz').open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for path in sorted(files):tar.add(path,arcname=path.lstrip('/'),recursive=False)
import guard234
guard234.EXPECTED_FREEZE=sha(P/'FREEZE234.json');guard234.EXPECTED_ARCHIVE=sha(P/'SOURCE234.tar.gz');guard234.prepared()
result=dict(status='source_frozen_CPU203_verified_device_unrun',directory=str(P),freeze_sha256=guard234.EXPECTED_FREEZE,archive_sha256=guard234.EXPECTED_ARCHIVE,
    source_members=len(files),neural_cases_planned=155,old_layout_separate_negative_control=True,original_Native_matcher_unchanged=True,
    original_startup201_44_114_not_run=True,performance=False,whole_workload_AR_qualified=False,adopted=False)
save(D/'ROOT_V234_SOURCE_PREPARED.json',result);print(json.dumps(result))
