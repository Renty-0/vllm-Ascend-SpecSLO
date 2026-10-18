import ast,datetime,gzip,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';OLD=B/'tree-M80-FULL9-numeric-isolation-diagnostic-v230';P=B/'tree-canonical-M80-independent-branch-math-v232'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
binding=json.loads((P/'PRIMITIVE_SOURCE_BINDING232.json').read_text())
for name,info in binding['helpers'].items():assert sha(P/name)==info['sha256']
mapper=json.loads((P/'HOST_MAPPER231_RESULT.json').read_text());schedule=json.loads((P/'CPU_REFERENCE_SCHEDULE231_RESULT.json').read_text());bank=json.loads((P/'CPU_BANK_PADDING231_RESULT.json').read_text())
assert mapper['geometry_cases']==890 and mapper['root_only_Q79_padding_geometry_checked']and not mapper['NPU_used']
assert schedule['cases']==512 and schedule['independent_candidate_query_paths']==18240 and not schedule['NPU_used']
assert bank['status']=='CPU_Native_allocator_metadata_forward_protocol_passed'and bank['original_real_positive_budget_guard_preserved']and bank['root_only_excluded_from_real_output_and_budget']and not bank['NPU_used']
files=json.loads((OLD/'FREEZE230.json').read_text())['files'];assert all(sha(p)==h for p,h in files.items())
files[str(OLD/'FREEZE230.json')]=sha(OLD/'FREEZE230.json')
extras=[D/'prepare_ROOT232_independent_math.py',D/'install232_SOURCE.py',D/'probes232_SOURCE.py',D/'ROOT_V230_TERMINAL_RESULT_AND_HANDOFF.json',D/'ROOT_V230_INDEPENDENT_NUMERIC_AND_ISOLATION_AUDIT.json',Path(__file__)]
for p in [*P.iterdir(),*extras]:
    if p.is_file()and p.suffix in('.py','.sh','.json'):
        if p.suffix=='.py':ast.parse(p.read_text())
        files[str(p)]=sha(p)
for p in P.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
sys.path[:0]=[str(P),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),str(B/'tree-high-normal-cap4-native-families-v148')]
from neural_oracle231 import cases
assert len(cases())==151 and {v['S']for v in cases()}=={14,15,16,17,18,22}
save(P/'FREEZE232.json',dict(schema='ROOT232.independent-M80-neural-node-math.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
    scope='151 synthetic tree-node math cases; real16/spine16/side16 separately prefilled; typed root-only dummy; no whole workload or performance qualification'))
files=dict(files);files[str(P/'FREEZE232.json')]=sha(P/'FREEZE232.json')
with(P/'SOURCE232.tar.gz').open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for path in sorted(files):tar.add(path,arcname=path.lstrip('/'),recursive=False)
import guard232
guard232.EXPECTED_FREEZE=sha(P/'FREEZE232.json');guard232.EXPECTED_ARCHIVE=sha(P/'SOURCE232.tar.gz');guard232.prepared()
result=dict(status='source_frozen_CPU_protocol_verified_device_unrun',directory=str(P),freeze_sha256=guard232.EXPECTED_FREEZE,archive_sha256=guard232.EXPECTED_ARCHIVE,
    source_members=len(files),neural_cases_planned=151,original_startup201_44_114_not_run=True,performance=False,whole_workload_AR_qualified=False,adopted=False)
save(D/'ROOT_V232_SOURCE_PREPARED.json',result);print(json.dumps(result))
