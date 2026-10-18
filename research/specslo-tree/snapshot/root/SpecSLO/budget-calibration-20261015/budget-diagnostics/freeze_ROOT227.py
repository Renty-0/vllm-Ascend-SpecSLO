import ast,datetime,gzip,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-exact-FULL-attention-matched-reference-diagnostic-v226';P=B/'tree-query1-FULL-canonical-numeric-cost-diagnostic-v227'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
cpu=json.loads((P/'CPU227_RESULT.json').read_text());v=json.loads((P/'CPU227_VISIBILITY_RESULT.json').read_text())
assert cpu['status']=='passed'and cpu['frames']==427 and cpu['NPU_used']is False
assert v['status']=='passed'and v['cases']==60 and len(v['negative_cases'])==4 and v['original_Native_mask_validator_passed']
assert(P/'FIXTURE227.json').read_bytes()==(OLD/'FIXTURE226.json').read_bytes()
files=json.loads((OLD/'FREEZE226.json').read_text())['files'];assert all(sha(p)==h for p,h in files.items())
files[str(OLD/'FREEZE226.json')]=sha(OLD/'FREEZE226.json')
for p in [*P.iterdir(),D/'prepare_ROOT227_query1.py',Path(__file__)]:
    if p.is_file()and p.suffix in('.py','.sh','.json'):
        if p.suffix=='.py':ast.parse(p.read_text())
        files[str(p)]=sha(p)
for p in P.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
save(P/'FREEZE227.json',dict(schema='ROOT227.query1-FULL-canonical-diagnostic.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
    scope='M80 fresh body, one-query FULL visibility, own physical exact KV length; uninstalled tie observer; diagnostic only'))
files=dict(files);files[str(P/'FREEZE227.json')]=sha(P/'FREEZE227.json')
with(P/'SOURCE227.tar.gz').open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for path in sorted(files):tar.add(path,arcname=path.lstrip('/'),recursive=False)
sys.path[:0]=[str(P),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')]
import guard227
guard227.EXPECTED_FREEZE=sha(P/'FREEZE227.json');guard227.EXPECTED_ARCHIVE=sha(P/'SOURCE227.tar.gz');guard227.prepared()
result=dict(status='source_frozen_HOST_CPU_verified_device_unrun',directory=str(P),freeze_sha256=guard227.EXPECTED_FREEZE,archive_sha256=guard227.EXPECTED_ARCHIVE,
    source_members=len(files),CPU_frames=427,additional_visibility_cases=60,negative_cases=4,original_inputs_byte_identical=True,
    original_Native_graph_validation_retained=True,production_installed=False,neural_qualified=False,performance_qualified=False,adopted=False)
save(D/'ROOT_V227_SOURCE_PREPARED.json',result);print(json.dumps(result))
