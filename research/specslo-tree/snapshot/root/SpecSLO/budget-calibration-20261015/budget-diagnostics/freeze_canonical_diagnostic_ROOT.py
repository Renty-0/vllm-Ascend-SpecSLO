"""ROOT source/evidence freeze only. No backend, model, or test execution."""
import ast, hashlib, importlib.util, json, pathlib, subprocess, sys, tarfile
from datetime import datetime, timezone

B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'; P=B/'canonical-ar-common-prefix-diagnostic-v1'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def pin(p): return dict(path=str(p),sha256=sha(p))
def save(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'CANONICAL_AR_COMMON_PREFIX_DIAGNOSTIC_V1_SOURCE_AND_HOST_READY_FOR_REVIEW_REVISION3.json'
peer=D/'CANONICAL_AR_COMMON_PREFIX_DIAGNOSTIC_V1_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='17ac1b0af25bcf3a54d4831de41f5c6644e93ef6fe256f06405ec4383ab2b154'
assert sha(peer)=='b4ca32be4137c301d572645f2fb6b377de696d830059f2aa65d3cde78671dbc6'
r=json.loads(ready.read_text()); prepared=json.loads((P/'PREPARED.json').read_text());cpu=json.loads((P/'CPU_RESULT.json').read_text())
assert len(r['pins'])==111 and r['source_counts']==dict(local=10,external=52)
assert len(prepared['sources']['local'])==10 and len(prepared['sources']['external'])==52
for p,h in r['pins'].items(): assert pathlib.Path(p).is_file() and sha(p)==h,p
assert cpu['status']=='passed' and cpu['tests_run']==3 and cpu['errors']==cpu['failures']==0
assert cpu['source_before']==cpu['source_after']==prepared['sources']
assert prepared['CPU_sha256']==sha(P/'CPU_RESULT.json')=='dc2b771711103d9bcd2fa49c8d96b02c829908ea5538858a7dab6581c7e95a26'
assert not cpu['NPU_used'] and not cpu['models_run'] and not cpu['old_suites_rerun']
for p in prepared['sources']['local']:
    if pathlib.Path(p).suffix=='.py': ast.parse(pathlib.Path(p).read_text(),filename=p)
subprocess.run(['bash','-n',str(P/'launch.sh')],check=True)
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('_ROOT_canonical_contract',P/'contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
assert c.require_source(approved=False)==prepared['sources']
assert 'torch' not in sys.modules
review=D/'ROOT_CANONICAL_AR_COMMON_PREFIX_DIAGNOSTIC_REV3_SOURCE_REVIEW.json'
record=dict(schema='ROOT.canonical-common-prefix-diagnostic.source-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
    status='passed_source_and_retained_HOST_review',ready=pin(ready),peer=pin(peer),CPU=pin(P/'CPU_RESULT.json'),
    exact_verified_pins=r['pins'],source_counts=r['source_counts'],new_HOST_tests=3,inherited_HOST_tests=4,
    HOST_tests_rerun=False,model_run=False,NPU_used=False,Torch_imported=False,
    reviewed_semantics=[
      'Four immutable first-difference prefixes from completed 173/190 and two saved own-AR prompts; exact token IDs and hashes.',
      'Own-AR advances from original prompt using only its own previous prediction; report independent of teacher-forced routes.',
      'Every fresh whole-prefix PA, incremental-root PA, eager FIA and graph FIA route starts with independently recomputed prefix KV.',
      'Reconstructed legal 16-owner B4 M80 S16 cohort is labelled diagnostic, never historical packed-call replay.',
      'Full physical 80-row hidden state reaches the original LM head before root-row selection; full actual head shape is retained.',
      'Original V86 whole-module worker code object and constructor/command/finally lifecycle retained with isolated globals.',
      'All three target ranks must agree on retained canonical outputs and route observations; original all-worker failure agreement is reused.',
      'Natural graph capture/validation/fallback flags are observed; no flags or qualification are forced.',
      'Self-owned ROOT lock covers baseline pre/post, one child group, bounded time/storage, cleanup and handler restoration.',
      'SIGTERM/SIGINT defer raising across child assignment, enter owned cleanup once, and cannot interrupt cleanup with repeated signals.',
      'Three new retained signal HOST tests used benign separate-session children; four retained original core HOST tests were not rerun.',
      'Only explicit selected local/external source and evidence bytes are packaged; no parent binary archive copied.'
    ],limitations=[
      'Unscored diagnosis only; no production permission, formal Goodput, whole463 own-AR equality or numerical-losslessness claim.',
      'Synthetic legal FIA cohort cannot reconstruct unretained historical packed proposals.',
      'Different PA/FIA and physical matrix shapes are explicit variables; margins alone cannot prove the root cause.',
      'All reviewed and HOST evidence is source-only; no actual diagnostic has run.'
    ],ROOT_source_reviewed=True,production_qualified=False,whole463_AR_qualified=False,formal_Goodput=False,adopted=False)
save(review,record)
members=dict(r['pins']);members.update({str(p):sha(p) for p in (ready,peer,review)})
assert len(members)==114
archive=D/'CANONICAL_AR_COMMON_PREFIX_DIAGNOSTIC_REV3_ROOT_SOURCE_EVIDENCE.tar.gz'
with tarfile.open(archive,'x:gz') as t:
    for p,h in sorted(members.items()):
        assert sha(p)==h
        t.add(p,arcname=p.lstrip('/'),recursive=False)
with tarfile.open(archive,'r:gz') as t:
    mm=t.getmembers();assert len(mm)==len(members) and all(m.isfile() for m in mm)
    assert set('/'+m.name for m in mm)==set(members)
    for m in mm: assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==members['/'+m.name]
freeze=P/'FROZEN_DIAGNOSTIC.json'
save(freeze,dict(schema='ROOT.canonical-common-prefix-diagnostic.source-freeze.v1',UTC=datetime.now(timezone.utc).isoformat(),
    ROOT_source_reviewed=True,sources=prepared['sources'],CPU_sha256=sha(P/'CPU_RESULT.json'),root_review=pin(review),
    peer_review=pin(peer),source_archive=pin(archive),source_members=len(members),ready=pin(ready),
    HOST_tests_rerun=False,NPU_used=False,not_launched=True,production_qualified=False,whole463_AR_qualified=False,
    numerical_losslessness_proved=False,formal_Goodput=False,adopted=False))
assert c.require_source(approved=True)==prepared['sources'] and 'torch' not in sys.modules
print(json.dumps(dict(status='ROOT_source_frozen_not_launched',freeze=pin(freeze),review=pin(review),archive=pin(archive),members=len(members))))
