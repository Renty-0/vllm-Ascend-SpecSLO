import ast,datetime,gzip,hashlib,importlib.util,json,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-canonical-FULL-attention-numeric-cost-diagnostic-v225';P=B/'tree-exact-FULL-attention-matched-reference-diagnostic-v226'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
cpu=json.loads((P/'CPU226_RESULT.json').read_text());assert cpu['status']=='passed'and cpu['frames']==427 and cpu['NPU_used']is False
files=json.loads((OLD/'FREEZE225.json').read_text())['files']
assert all(sha(p)==h for p,h in files.items())
files[str(OLD/'FREEZE225.json')]=sha(OLD/'FREEZE225.json')
for p in [*P.iterdir(),D/'prepare_ROOT226_exact_FULL.py',Path(__file__),D/'ROOT_V225_TERMINAL_INVALID_COMPARATOR_AND_CORRECTION.json']:
    if p.is_file()and p.suffix in('.py','.sh','.json'):
        if p.suffix=='.py':ast.parse(p.read_text())
        files[str(p)]=sha(p)
for p in P.glob('*.sh'):
    import subprocess
    subprocess.run(['bash','-n',str(p)],check=True)
assert 'bucket_tree_fia_attention_metadata'not in(P/'history226.py').read_text()
save(P/'FREEZE226.json',dict(schema='ROOT226.exact-FULL-matched-metadata-diagnostic.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
    scope='identical exact-length eager/graph reference; uninstalled FP32 tie observer; four contexts and isolated body/head costs; not production qualification'))
files=dict(files);files[str(P/'FREEZE226.json')]=sha(P/'FREEZE226.json')
with(P/'SOURCE226.tar.gz').open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for path in sorted(files):tar.add(path,arcname=path.lstrip('/'),recursive=False)
sys.path[:0]=[str(P),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')]
import guard226
guard226.EXPECTED_FREEZE=sha(P/'FREEZE226.json');guard226.EXPECTED_ARCHIVE=sha(P/'SOURCE226.tar.gz')
guard226.prepared()
result=dict(status='source_frozen_HOST_CPU_verified_device_unrun',directory=str(P),freeze_sha256=guard226.EXPECTED_FREEZE,archive_sha256=guard226.EXPECTED_ARCHIVE,
    source_members=len(files),CPU_frames=427,original218_closure_preserved=True,original225_failure_retained=True,performance_qualified=False,adopted=False)
save(D/'ROOT_V226_SOURCE_PREPARED.json',result)
print(json.dumps(result))
