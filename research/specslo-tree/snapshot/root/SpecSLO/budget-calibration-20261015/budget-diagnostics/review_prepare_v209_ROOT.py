"""ROOT freeze of role-aware retained audit only; actual math must rerun."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
G = D / 'v209-root-execution'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pin(p):
    return dict(path=str(p), sha256=sha(p))

def save(p, value):
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--expected-peer-sha256', required=True)
args = parser.parse_args()
assert sys.version_info[:2] == (3, 11)
assert re.fullmatch('[0-9a-f]{64}', args.expected_peer_sha256)
ready = D / 'V209_ROLE_AWARE_RETAINED_AUDIT_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer = D / 'V209_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready) == '6b4a8e617cc128cae7ae98a72730ecefc19e3c2af03e2fe71cf1ffd5656a52d6'
assert sha(peer) == args.expected_peer_sha256
r, pr = json.loads(ready.read_bytes()), json.loads(peer.read_bytes())
assert r['source_stopped'] is True and r['freeze_created'] is False
assert pr['status'].replace('-', '_') == 'passed_source_review' and pr['candidate'] == str(P)
assert pr.get('blocking_findings', []) == pr.get('unresolved_source_blockers', []) == []
cl = r['source_closure']
for group in ('local', 'external', 'retained_HOST_evidence', 'guard'):
    for path, expected in cl[group].items():
        assert sha(path) == expected, path
for key in ('actual_cold_guard', 'cold_guard_source'):
    value = r['HOST'][key]
    assert sha(value['path']) == value['sha256']
mechanical = r['exact_source_projection']['mechanical']
assert sha(mechanical['path']) == mechanical['sha256']
projection = json.loads((P / 'PROJECTIONS209.json').read_bytes())
assert len(projection['projection']) == 20 and len(projection['guard_projection']) == 4
assert {v['file'] for v in projection['projection'] if v['deltas']} == {'base209.py', 'audit_usage209.py'}
for group, root in (('projection', P), ('guard_projection', G)):
    for row in projection[group]:
        assert sha(row['parent']) == row['parent_sha256']
        assert row['namespace'] == ['205', '209']
        original = Path(row['parent']).read_text().replace(*row['namespace'])
        parts, offset = [], 0
        for op in row['deltas']:
            assert op['start'] >= offset and original[op['start']:op['end']] == op['old']
            parts.extend((original[offset:op['start']], op['new']))
            offset = op['end']
        assert ''.join(parts) + original[offset:] + row.get('append_text', '') == (root / row['file']).read_text()
for path in cl['local']:
    if Path(path).suffix == '.py':
        ast.parse(Path(path).read_text(), filename=path)
ast.parse((G / 'guard209.py').read_text())
for name in ('startup209.sh', 'launch209.sh'):
    subprocess.run(['bash', '-n', str(G / name)], check=True)
spec = importlib.util.spec_from_file_location('_ROOT209_source_review_guard', G / 'guard209.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state, guards = guard.source_state(archive=True), guard.guard_inputs()
assert state['local'] == cl['local'] and state['external'] == cl['external']
assert state['evidence'] == cl['retained_HOST_evidence'] and guards == cl['guard']
assert [len(state[k]) for k in ('local', 'external', 'evidence')] == [26, 1395, 100]
assert len(guards) == 4
assert sha(guard.CPU) == '8c1c085cb831ff26a28fda0ba92e908cd730bfdf067b09c25304a5708f5e60d0'
cpu = json.loads(guard.CPU.read_bytes())
assert cpu['tests_run'] == 3 and cpu['errors'] == cpu['failures'] == 0
assert cpu['child_Torch_CPU_control_only'] is False
details = cpu['details']
assert details['mutated_negatives']['actual_audit_function_executed'] is True
assert details['mutated_negatives']['old_math_consensus_Native_and_matcher_gates_not_relaxed'] is True
cases = details['mutated_negatives']['cases']
assert len(cases) == 15 and all(v['rejected'] is True for v in cases)
replay = details['retained_role_replay']
assert replay['actual1176_zero_subtractions'] == 1176
assert len(replay['fixture_raw_sha256']) == 424
assert replay['actual_changed_TTU'] > 0
assert r['retained_failure205']['old_audit_TypeError_reproduced'] is True
assert r['HOST']['current_test_count_includes_old205seven'] is False
assert r['retained_failure205']['actual_old205_qualification_not_reused'] is True
inputs = guard.merge(state['local'], state['external'], state['evidence'], guards,
                     {str(guard.CPU): sha(guard.CPU), str(peer): sha(peer)})
assert len(inputs) == 1527 and 'torch' not in sys.modules
assert {k: v for k, v in inputs.items() if k != str(peer)} == cl['flat_before_peer']
review = D / 'ROOT_V209_ROLE_AWARE_RETAINED_AUDIT_SOURCE_AND_GUARD_REVIEW.json'
save(review, dict(schema='ROOT.V209.role-aware-retained-audit-source-review.v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='passed_source_review', candidate=str(P),
    ready=pin(ready), peer=pin(peer), reviewed_inputs=inputs, ROOT_source_reviewed=True,
    HOST_tests_rerun=False, NPU_used=False, Torch_imported=False,
    parent_archives_verified_in_place=state['parent_archive'], whole_source_projections_verified=20,
    guard_projections_verified=4, sole_production_parent=197, source_donor=205,
    reviewed_semantics=[
        '205 remains immutable after its failed retained-audit stage. Its968-member source archive and all upstream sources are independently verified in place; old receipts or seven donor HOST checks confer no209 device permission.',
        '20 whole source projections are reconstructed. Actor, draft, target, mapper, original protocol/commit/ownnext, budgets, original197 admission and clocks have no computation delta after namespace; only role-sensitive audit and source-donor provenance change.',
        'Common cycle obligations remain for all4. Full16 variable draft count and all56KV/raw8 proof are required on draft rank0; target ranks1..3 must retain draft=None and their own target/reference/crossM/old-control dictionaries. Rank0 must not fabricate target proofs.',
        'Own-next records likewise require complete independent draft proof on rank0 and None on target ranks. All integer BONUS/head/verdict/compaction/history/Native21 subtraction/twozero1176/task-update/prefix and final object gates remain.',
        'Three current HOST tests run together. Exact authorized terminal205424 raw files reproduce the old TypeError and exercise the corrected actual auditor over four reports/336cycles/84next.15 negative tests reject missing role fields, heads, original integer/zero and matcher failures. This is schema/integer regression, not recomputed tensors or209 model qualification.',
        'Four execution guard projections bind the actual new P path, launch/startup and original permit assignment. Original selflock,10496/15/212 prepost, signal/owned cleanup,7200s/private caps and immutable-source checks remain.',
        'New current209 model must rerun original201/44/114,B4/M80,197multiowner N2/N32, seven independent same-M realN9..15 protocols and ownnext/1176zero, then original warm/twozero/seal/full463. No new key/profile/budget/policy/token/arrival/publication or metric definition change.',
        'Current CPU counts are3; old205 seven are separately verified donor evidence. All three new attempts and96 input/log snapshots, including two failures, remain. Cold actual guard rejects absent own ROOT freeze.'
    ], limitations=[
        'Source freeze is not209 numerical or performance qualification. The authorized old failed raw data is included as explicit private HOST regression input only.',
        'Missing required physical-head negative currently rejects by TypeError while explicit role gates use RuntimeError; both prevent success. No failure is reclassified as a passing run.',
        'Cross-M80/M128 compatibility and current same-M math are separate; finite component checks do not establish whole463 canonical own-AR identity.',
        'Native charged/256, worker publication/255, original-arrival TTFT and future common-origin final-nine metrics stay separate. No Goodput, adoption or formal1.5x is assigned.'
    ], formal_Goodput=False, adopted=False))
command = [sys.executable, '-B', str(G / 'guard209.py'), '--prepare-only', '--root-review', str(review),
           '--expected-root-review-sha256', sha(review), '--peer-review', str(peer), sha(peer)]
result = subprocess.run(command, capture_output=True, text=True)
save(D / 'ROOT_V209_GUARD_PREPARE_STDOUT_AND_STDERR.json', dict(
    command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
assert result.returncode == 0, result.stderr
frozen, sources, prepared = guard.prepared()
assert len(sources) == prepared['archive']['members'] == 1529 and 'torch' not in sys.modules
output = D / 'ROOT_V209_PREPARED_SOURCE_HOST_AUDIT.json'
save(output, dict(schema='ROOT.V209.prepared-source-HOST-audit.v1', status='passed_source_frozen_unrun',
    candidate=str(P), review=pin(review), peer=pin(peer), ready=pin(ready), freeze=pin(guard.FREEZE),
    archive=pin(guard.ARCHIVE), actual_source_members=len(sources),
    actual_archive_members=prepared['archive']['members'], prepared=prepared,
    NPU_used=False, Torch_imported=False, models_launched=False, formal_Goodput=False, adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun', record=pin(output),
    freeze=pin(guard.FREEZE), archive=pin(guard.ARCHIVE), members=len(sources))))
