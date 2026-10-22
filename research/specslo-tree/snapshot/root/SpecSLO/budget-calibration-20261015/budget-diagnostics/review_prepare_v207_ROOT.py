"""ROOT-only source freeze; no full-run authorization or numerical inheritance."""
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
P = B / 'tree-normal-cap3-original-geometries-full-v207'
G = D / 'v207-root-execution'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pin(path):
    return dict(path=str(path), sha256=sha(path))

def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--expected-peer-sha256', required=True)
args = parser.parse_args()
assert sys.version_info[:2] == (3, 11), 'original source/AST environment'
assert re.fullmatch('[0-9a-f]{64}', args.expected_peer_sha256)
ready = D / 'V207_NORMAL_CAP3_ORIGINAL_GEOMETRIES_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer = D / 'V207_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready) == '0814b7f013a2828ccc2e676a2fe87dad9cf01b86ce7a8228f1185116e59cb35c'
assert sha(peer) == args.expected_peer_sha256
r = json.loads(ready.read_bytes())
pr = json.loads(peer.read_bytes())
assert r['freeze_created'] is False and r['entry_enabled'] is False
assert pr['status'].replace('-', '_') == 'passed_source_review'
assert pr['candidate'] == str(P)
assert pr.get('blocking_findings', []) == pr.get('unresolved_source_blockers', []) == []
closure = r['source_closure']
for group in ('local', 'external', 'retained_HOST_evidence', 'execution_guard_inputs'):
    for path, expected in closure[group].items():
        assert sha(path) == expected, path
for path, expected in r['auxiliary_review_evidence'].items():
    assert sha(path) == expected, path
projection = json.loads((P / 'PROJECTIONS207.json').read_bytes())['projection']
assert len(projection) == 20
for row in projection:
    assert sha(row['parent']) == row['parent_sha256']
    original = Path(row['parent']).read_text()
    parts, offset = [], 0
    for op in row['deltas']:
        assert op['start'] >= offset and original[op['start']:op['end']] == op['old']
        parts.extend((original[offset:op['start']], op['new']))
        offset = op['end']
    assert ''.join(parts) + original[offset:] == (P / row['file']).read_text(), row['file']
for path in closure['local']:
    if Path(path).suffix == '.py':
        ast.parse(Path(path).read_text(), filename=path)
ast.parse((G / 'guard207.py').read_text())
for name in ('startup207.sh', 'launch207.sh'):
    subprocess.run(['bash', '-n', str(G / name)], check=True)
spec = importlib.util.spec_from_file_location('_ROOT207_source_review_guard', G / 'guard207.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guards = guard.guard_inputs()
assert state['local'] == closure['local']
assert state['external'] == closure['external']
assert state['evidence'] == closure['retained_HOST_evidence']
assert guards == closure['execution_guard_inputs']
assert [len(state[k]) for k in ('local', 'external', 'evidence')] == [43, 319, 327]
assert len(guards) == 4
assert sha(guard.CPU) == '7d606307d6fc7254741c16eb03d5a69bab9cd536b200a609aa530187fb1f175b'
cpu = json.loads(guard.CPU.read_bytes())
assert cpu['tests_run'] == 6 and cpu['errors'] == cpu['failures'] == 0
assert cpu['child_Torch_CPU_control_only'] is False
roles = cpu['details']['actual_role_aware_auditor_schemas']
assert roles['actual_source_audits_executed'] is True
assert roles['explicit_synthetic_JSON_carriers'] is True
assert roles['current_model_math_qualified'] is False
assert roles['normal']['roles_with_target_draft_None'] is True
assert roles['normal']['bad_target_None_rejected'] is True
assert roles['eager']['roles_with_target_next_None'] is True
assert roles['eager']['missing_reference_rejected'] is True
contract = r['E_source_contract']
assert contract['legal_B1_B2_with_E_and_small_total_E_are_reachable_outside_new_scope'] is True
assert contract['normal_and_child_budgets'] == [3]
assert contract['shared_and_target_total_N'] == list(range(17, 33))
assert 'raise before model work' in contract['outside_new_E_contract']
assert not [path for path in state['external'] if path.endswith('.tar.gz')]
inputs = guard.merge(state['local'], state['external'], state['evidence'], guards,
                     {str(guard.CPU): sha(guard.CPU), str(peer): sha(peer)})
assert len(inputs) == 695 and 'torch' not in sys.modules
assert closure['selected_flat_count'] == 694
assert {k: v for k, v in inputs.items() if k != str(peer)} == closure['selected_flat_before_peer']
review = D / 'ROOT_V207_NORMAL_CAP3_ORIGINAL_GEOMETRIES_SOURCE_AND_GUARD_REVIEW.json'
save(review, dict(
    schema='ROOT.V207.cap3-original-geometries-source-review.v1',
    UTC=datetime.now(timezone.utc).isoformat(), status='passed_source_review', candidate=str(P),
    ready=pin(ready), peer=pin(peer), reviewed_inputs=inputs, ROOT_source_reviewed=True,
    HOST_tests_rerun=False, NPU_used=False, Torch_imported=False,
    parent_archives_verified_in_place=state['parent_archive'], whole_text_projections_verified=20,
    full463_execution_authorized=False,
    known_reachable_production_E_outside_current_qualification_contract=True,
    reviewed_semantics=[
        'Frozen197 is the sole production parent. Original shape/mode/clock/allocator and all current173+B4/M80+197 multiowner gates remain required. Source archives are verified in place, not numerical permission.',
        'Normal cap changes4 to3 only when admitted normal N>8; lowN<=8 still cap5. C96/E96/global192, original gain/prior/posterior and priority allocator remain; original logical homes16 and actual max_target_batch32/ready merging remain.',
        'Actual three model/head steps produce raw6/canonical8 zeros3,7 and confidence denominator6. Draft input KV ledger root,d1,d2 and subsequent d3 repair are checked with genuine independent history, full physical hidden and56 KV; target remains original B3 BONUS max4.',
        'N16/B3 target uses original M96/S32 with16 q2 dummy rows. New hidden H96/S32 is diagnostic; new E controlled shapes additionally require H128/S38. Original G entries and MC2 profiles are naturally validated, never assigned flags.',
        'E math layer covers all16 controlled totals17..32 at allB3, actual original padding/metadata/full physical hidden/head/128 KV/verdict/commit/ownnext and three passes/two zero sweeps. It is a parametric representative contract, not exhaustive token tuple testing.',
        'Independent natural original nonunion P3/shared3 layer requires actual parent target/finish/planner32 and positive promotion/consume plus invalid/rollback-next; zero real coverage yields subset and no E issuer. No union/P4 receipt is substituted.',
        'Same-engine B3/base/prefill/E proof identities and actual Native object identities govern issuance. Per-target original ticket/prefix/epoch checks precede a CPU E scope gate; outside domain raises without changing original grants or disabling E.',
        'Component-only protocol/target observers are installed before constructor and removed before warm; no new hot-path collective. Original warm65 fixedpoint/twozero/seal/full463 and original21 failure/zero counters remain.',
        'Six current HOST checks ran together. Real four-role actor/auditor schema functions ran on explicitly synthetic JSON flags, with correct role None handling and negative required-field tests; these are not floating tensor evidence.',
        '20 whole-file source projections are independently reconstructed, all selected sources and retained attempts hashed. Original guardian selflock/prepost10496-15-212/owned cleanup/private limits remain.'
    ], limitations=[
        'This freezes source only. No207 model gate or complete workload result exists, and no full-run launch is authorized by this review.',
        'Legal normal B1/B2 plus E3, or total E cohorts below17, remain reachable outside this new E evidence. Warm/full can fail closed. Source readiness does not establish a usable full-workload implementation; E is not suppressed and budgets are not forced.',
        'Finite representative normal/E math and old numerical flags do not prove whole463 own-AR identity. Original charged/256, publication/255, original-arrival TTFT and future common-origin clocks stay separate.',
        'Lower candidate cap or more E promotion need not improve Goodput. No model performance, adoption or formal1.5x is claimed.'
    ], formal_Goodput=False, adopted=False))
command = [sys.executable, '-B', str(G / 'guard207.py'), '--prepare-only',
           '--root-review', str(review), '--expected-root-review-sha256', sha(review),
           '--peer-review', str(peer), sha(peer)]
completed = subprocess.run(command, capture_output=True, text=True)
save(D / 'ROOT_V207_GUARD_PREPARE_STDOUT_AND_STDERR.json', dict(
    command=command, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr))
assert completed.returncode == 0, completed.stderr
frozen, sources, prepared = guard.prepared()
assert len(sources) == prepared['archive']['members'] == 697 and 'torch' not in sys.modules
output = D / 'ROOT_V207_PREPARED_SOURCE_HOST_AUDIT.json'
save(output, dict(schema='ROOT.V207.prepared-source-HOST-audit.v1',
    status='passed_source_frozen_unrun', candidate=str(P), review=pin(review), peer=pin(peer),
    ready=pin(ready), freeze=pin(guard.FREEZE), archive=pin(guard.ARCHIVE),
    actual_source_members=len(sources), actual_archive_members=prepared['archive']['members'],
    prepared=prepared, full463_execution_authorized=False,
    known_reachable_E_scope_gaps=True, NPU_used=False, Torch_imported=False,
    models_launched=False, formal_Goodput=False, adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun', record=pin(output),
    freeze=pin(guard.FREEZE), archive=pin(guard.ARCHIVE), members=len(sources), full463_authorized=False)))
