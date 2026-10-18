import ast,datetime,gzip,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-query1-FULL-canonical-numeric-cost-diagnostic-v227';P=B/'tree-query1-FULL-vs-FULL9-numeric-cost-diagnostic-v228'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
cpu=json.loads((P/'CPU228_RESULT.json').read_text());v=json.loads((P/'CPU228_VISIBILITY_RESULT.json').read_text())
assert cpu['status']=='passed'and cpu['frames']==427 and cpu['NPU_used']is False
assert v['status']=='passed'and v['cases']==60 and len(v['negative_cases'])==4 and v['original_Native_mask_validator_passed']
assert(P/'FIXTURE228.json').read_bytes()==(OLD/'FIXTURE227.json').read_bytes()
files=json.loads((OLD/'FREEZE227.json').read_text())['files'];assert all(sha(p)==h for p,h in files.items())
files[str(OLD/'FREEZE227.json')]=sha(OLD/'FREEZE227.json')
for p in [*P.iterdir(),D/'prepare_ROOT228_query1_FULL_cost.py',Path(__file__),D/'ROOT_V226_TERMINAL_FAILED_COST_WITH_COMPLETED_NUMERICS.json']:
    if p.is_file()and p.suffix in('.py','.sh','.json'):
        if p.suffix=='.py':ast.parse(p.read_text())
        files[str(p)]=sha(p)
for p in P.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
save(P/'FREEZE228.json',dict(schema='ROOT228.query1-FULL-canonical-diagnostic.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
    scope='M80 fresh body, one-query FULL visibility, own physical exact KV length; uninstalled tie observer; diagnostic only'))
files=dict(files);files[str(P/'FREEZE228.json')]=sha(P/'FREEZE228.json')
with(P/'SOURCE228.tar.gz').open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for path in sorted(files):tar.add(path,arcname=path.lstrip('/'),recursive=False)
sys.path[:0]=[str(P),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')]
import guard228
guard228.EXPECTED_FREEZE=sha(P/'FREEZE228.json');guard228.EXPECTED_ARCHIVE=sha(P/'SOURCE228.tar.gz');guard228.prepared()
result=dict(status='source_frozen_HOST_CPU_verified_device_unrun',directory=str(P),freeze_sha256=guard228.EXPECTED_FREEZE,archive_sha256=guard228.EXPECTED_ARCHIVE,
    source_members=len(files),CPU_frames=427,additional_visibility_cases=60,negative_cases=4,original_inputs_byte_identical=True,
    original_Native_graph_validation_retained=True,production_installed=False,neural_qualified=False,performance_qualified=False,adopted=False)
save(D/'ROOT_V228_SOURCE_PREPARED.json',result);print(json.dumps(result))
