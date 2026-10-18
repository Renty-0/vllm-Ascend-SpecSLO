import ast,datetime,gzip,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';OLD=B/'tree-canonical-M80-independent-branch-math-v232';P=B/'tree-contiguous-branch-tail-M80-math-v233'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
cpu=json.loads((P/'CPU_BRANCH_TAIL233_RESULT.json').read_text())
assert cpu['status']=='CPU_exact_visible_KV_order_and_Native_ABI_passed'and cpu['cases']==203 and cpu['branch_cases']==164 and cpu['max_actual_copy_slots']<=1792 and not cpu['NPU_used']
for path,digest in cpu['files_sha256'].items():assert sha(path)==digest
assert(P/'FIXTURE233.json').read_bytes()==(OLD/'FIXTURE232.json').read_bytes()
files=json.loads((OLD/'FREEZE232.json').read_text())['files'];assert all(sha(p)==h for p,h in files.items())
files[str(OLD/'FREEZE232.json')]=sha(OLD/'FREEZE232.json')
extras=[D/'prepare_ROOT233_branch_copy.py',D/'baseline233_SOURCE.py',D/'ROOT_V232_TERMINAL_FIRST_FORK_HIDDEN_FAILURE.json',Path(__file__)]
for p in [*P.iterdir(),*extras]:
    if p.is_file()and p.suffix in('.py','.sh','.json'):
        if p.suffix=='.py':ast.parse(p.read_text())
        files[str(p)]=sha(p)
for p in P.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
sys.path[:0]=[str(P),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),str(B/'tree-high-normal-cap4-native-families-v148')]
from neural_oracle231 import cases
assert len(cases())==155 and {v['S']for v in cases()}=={14,15,16,17,18,22}
save(P/'FREEZE233.json',dict(schema='ROOT233.contiguous-branch-tail-neural-math.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
    scope='155 independent node math cases; constantM80; side-query contiguous scratch KV, original BF16 weights/greedy; old-layout negative control; no performance or full workload qualification'))
files=dict(files);files[str(P/'FREEZE233.json')]=sha(P/'FREEZE233.json')
with(P/'SOURCE233.tar.gz').open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for path in sorted(files):tar.add(path,arcname=path.lstrip('/'),recursive=False)
import guard233
guard233.EXPECTED_FREEZE=sha(P/'FREEZE233.json');guard233.EXPECTED_ARCHIVE=sha(P/'SOURCE233.tar.gz');guard233.prepared()
result=dict(status='source_frozen_CPU203_verified_device_unrun',directory=str(P),freeze_sha256=guard233.EXPECTED_FREEZE,archive_sha256=guard233.EXPECTED_ARCHIVE,
    source_members=len(files),neural_cases_planned=155,old_layout_separate_negative_control=True,original_Native_matcher_unchanged=True,
    original_startup201_44_114_not_run=True,performance=False,whole_workload_AR_qualified=False,adopted=False)
save(D/'ROOT_V233_SOURCE_PREPARED.json',result);print(json.dumps(result))
