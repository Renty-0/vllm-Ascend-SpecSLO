"""ROOT review and selected-byte freeze; no backend, tests, lock, or device."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'canonical-ar-prefix-history-isolation-diagnostic-v2'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pin(path):
    return dict(path=str(path), sha256=sha(path))

def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

ready = D / 'CANONICAL_AR_PREFIX_HISTORY_ISOLATION_DIAGNOSTIC_V2_SOURCE_AND_HOST_READY_FOR_ROOT_REVIEW.json'
peer = D / 'CANONICAL_AR_PREFIX_HISTORY_ISOLATION_DIAGNOSTIC_V2_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready) == '99401526ef1abeb8549519349f629765a354d7abc4d2250216d59d8a7937a398'
assert sha(peer) == '213176bfeabece4d06d6ccb4070eb9d60deea8f39aa004ec912bf1678d30aa39'
r = json.loads(ready.read_text())
p = json.loads((P / 'PREPARED.json').read_text())
c = json.loads((P / 'CPU_RESULT.json').read_text())
pins = {}
for key in ('local_sources', 'selected_external', 'retained_HOST_evidence'):
    for path, digest in r[key].items():
        assert path not in pins or pins[path] == digest
        assert sha(path) == digest, path
        pins[path] = digest
assert len(pins) == r['unique_pins'] == 89
assert p['sources'] == dict(local=r['local_sources'], external=r['selected_external'])
assert len(p['sources']['local']) == 10 and len(p['sources']['external']) == 69
assert c['status'] == 'passed' and c['tests_run'] == 4
assert c['failures'] == c['errors'] == 0
assert c['source_before'] == c['source_after'] == p['sources']
assert p['CPU_sha256'] == sha(P / 'CPU_RESULT.json') == 'a6716b242a3635dade0fa94c634ac0c74519f222e3f0ab7f9548465348829072'
assert c['Torch_imported'] is c['NPU_used'] is c['model_constructed'] is False
assert c['required_future_own_target_model_and_head_calls'] == 9200
assert json.loads(peer.read_text())['status'] == 'passed_source_review'
for path in p['sources']['local']:
    if Path(path).suffix == '.py':
        ast.parse(Path(path).read_text(), filename=path)
subprocess.run(['bash', '-n', str(P / 'launch.sh')], check=True)
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_prefix_history_v2_contract', P / 'contract.py')
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
assert contract.require_source(approved=False) == p['sources']
assert 'torch' not in sys.modules

review = D / 'ROOT_CANONICAL_AR_PREFIX_HISTORY_ISOLATION_V2_SOURCE_REVIEW.json'
save(review, dict(
    schema='ROOT.canonical-prefix-history-isolation-v2.source-review.v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='passed_source_and_retained_HOST_review',
    ready=pin(ready), peer=pin(peer), CPU=pin(P / 'CPU_RESULT.json'), exact_verified_pins=pins,
    source_counts=dict(local=10, external=69), new_HOST_tests=4, HOST_tests_rerun=False,
    reviewed_semantics=[
        'All 16 owners independently advance from original prompts by their own predictions, with reference-prefix assertions and 9200 model/head calls required per target rank.',
        'Reserve the complete root-to-root+4 query span; protect every prior position on all 16 owners using 128 original-dtype byte hashes after each controlled route.',
        'Controlled root PA, eager FIA and natural graph hidden/greedy F/T/F share that protected past; private TP KV shard hashes are not compared across ranks.',
        'The same actual physical80 hidden tensor reaches both original full80 and owner0 single-row vocabulary heads; record byte hashes, margins and all differing vocabulary coordinates without forcing cross-shape equality.',
        'Original V86 whole-module worker constructor/command/finally lifecycle and original V1 helper AST are retained with isolated globals.',
        'Only operational worker receive timeout changes from1800 to3600; source and HOST prove that the model, math, sampling and other configuration fields are unchanged.',
        'Guardian requires ROOT freeze before lock, has one owned process group, original baseline pre/post checks, bounded storage/time and mandatory signal-safe cleanup.',
        'Only selected text and HOST evidence are archived; parent binary archives and weights are not copied.'
    ], limitations=[
        'No model has run; source and HOST evidence do not establish actual9200 calls, whole463 own-AR equality, numerical losslessness, performance, or production qualification.',
        'Shared-past controls are isolation diagnostics, not independent fresh numerical qualification; a partial or divergent history cannot be called all-cases success.',
        'The outer3600-second deadline begins before constructor and preflight; the inner3600-second receive budget is not guaranteed, and cleanup may extend total elapsed.',
        'The first failed HOST fixture and its complete source snapshot remain retained; original parent signal HOST tests were not rerun.'
    ], ROOT_source_reviewed=True, Torch_imported=False, NPU_used=False,
    production_qualified=False, whole463_AR_qualified=False,
    numerical_losslessness_proved=False, formal_Goodput=False, adopted=False))

members = dict(pins)
members.update({str(path):sha(path) for path in (ready, peer, review)})
assert len(members) == 92
archive = D / 'CANONICAL_AR_PREFIX_HISTORY_ISOLATION_V2_ROOT_SOURCE_EVIDENCE.tar.gz'
with tarfile.open(archive, 'x:gz') as tar:
    for path, digest in sorted(members.items()):
        assert sha(path) == digest
        tar.add(path, arcname=path.lstrip('/'), recursive=False)
with tarfile.open(archive, 'r:gz') as tar:
    entries = tar.getmembers()
    assert len(entries) == len(members) and all(m.isfile() for m in entries)
    assert {'/' + m.name for m in entries} == set(members)
    for entry in entries:
        assert hashlib.sha256(tar.extractfile(entry).read()).hexdigest() == members['/' + entry.name]
freeze = P / 'FROZEN_DIAGNOSTIC.json'
save(freeze, dict(schema='ROOT.canonical-prefix-history-isolation-v2.source-freeze.v1',
    UTC=datetime.now(timezone.utc).isoformat(), ROOT_source_reviewed=True,
    sources=p['sources'], CPU_sha256=sha(P / 'CPU_RESULT.json'), root_review=pin(review),
    peer_review=pin(peer), source_archive=pin(archive), source_members=len(members),
    ready=pin(ready), HOST_tests_rerun=False, NPU_used=False, not_launched=True,
    production_qualified=False, whole463_AR_qualified=False,
    numerical_losslessness_proved=False, formal_Goodput=False, adopted=False))
assert contract.require_source(approved=True) == p['sources'] and 'torch' not in sys.modules
print(json.dumps(dict(status='ROOT_source_frozen_not_launched', freeze=pin(freeze),
    review=pin(review), archive=pin(archive), members=len(members))))
