"""Replay saved parent verdict gates to locate the genuinely absent mixed case.

The alternative changes only diagnostic SLO inputs and planner switches. It
does not execute, qualify, or infer neural results for the new mixed batch.
"""
import gzip
import hashlib
import json
from pathlib import Path
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
ACTUAL = (B / 'tree-bounded-parent4-shared4-eager-diagnostic-v199/actual'
          / 'parent4-child4-current-M120-root-v395-import-repair-v1')


def reconstruct(rhythm, row, *, diagnostic_SLO):
    states = {}
    for owner in range(64):
        fields = {}
        if diagnostic_SLO:
            label = 'tight' if owner >= 16 else ('normal' if owner < 12 else 'loose')
            fields = dict(slo_class=label, slo_tpot_ms={'tight': 40., 'normal': 50., 'loose': 150.}[label])
        states[owner] = rhythm.SpecRhythmRuntimeState(owner, 0 if owner < 16 else 1, max_gamma=4, **fields)
    controller = rhythm.SpecRhythmPipelineController(states)
    gates = {g['owner']: g for g in row['promotion_gates']}
    for owner in range(32):
        controller.publish([controller.new_ticket(owner, gamma=4, eager=False)])
        if owner in gates:
            controller.publish([controller.new_ticket(owner, gamma=4, eager=True)])
    parent = row['phases'][0]
    assert parent['owners'] == list(range(16, 32))
    actual_promotions = []
    for saved in parent['tickets']:
        owner = saved['owner']
        assert controller.ready[owner].proposal_id == saved['proposal_id']
        # Native's retained promotable gate includes dependency/frontier
        # validation; replay it rather than inventing a successful verdict.
        effective_gate = bool(gates.get(owner, {}).get('promotable', False))
        promoted = controller.finish_verification(
            owner, fully_accepted=effective_gate, proposed_tokens=saved['gamma'],
            accepted_tokens=len(saved['accepted_nodes']), delivered_tokens=len(saved['emitted_tokens']),
            draft_confidence=None, ema_alpha=.2)
        if promoted is not None:
            assert promoted.proposal_id == saved['promoted_proposal_id']
            actual_promotions.append(owner)
    controller.finish_cycle(1)
    assert sorted(actual_promotions) == row['promoted']
    ready = {i: ticket.gamma for i, ticket in controller.ready.items()}
    plan = controller.build_plan(range(32), active_indices_registered_unique=True,
                                 merge_ready_homes=True, max_target_requests=16,
                                 verification_budget=96, ready_candidate_counts=ready,
                                 priority=diagnostic_SLO, target_slo_first=diagnostic_SLO)
    owners = list(plan.target_request_indices)
    eager = [i for i in owners if controller.ready[i].eager]
    return dict(owners=owners, eager=eager, N=len(owners), mixed=bool(eager) and len(eager) < len(owners))


def main():
    qualification = json.loads((D / 'ROOT395_TERMINAL_INDEPENDENT_FINITE_AUDIT.json').read_text())
    assert qualification['status'] == 'terminal_complete_finite_structural_audit_passed'
    assert qualification['all_required_natural_coverage'] is False
    sys.path.insert(0, str(B / 'tree-frugal-S32-paid-budget-control-host-v401'))
    from check401_HOST import original_control_sources, BACKENDS
    rhythm = original_control_sources()
    results = []
    for path in sorted((ACTUAL / 'bounded199').glob('rank0-pass0-*.json.gz')):
        row = json.loads(gzip.decompress(path.read_bytes()))
        old = reconstruct(rhythm, row, diagnostic_SLO=False)
        assert old['owners'] == row['planner'][0]['owners'] and old['eager'] == row['planner'][0]['eager']
        assert old['N'] == 16 and not old['eager'] and row['mixed16Q80_child_observed'] is False
        assert row['planner'][1]['N'] == len(row['promoted']) < 16
        assert row['planner'][1]['eager'] == row['promoted']
        proposed = reconstruct(rhythm, row, diagnostic_SLO=True)
        assert proposed['N'] == 16 and proposed['mixed'] is True
        assert proposed['eager'] == row['promoted']
        results.append(dict(case=row['name'], actual=str(path),
            actual_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_first_plan=old, original_second_plan=row['planner'][1],
            actual_promotions_retained=row['promoted'], proposed_diagnostic_SLO_first_plan=proposed,
            proposed_mixed_neural_checks_executed=False))
    assert len(results) == 5
    assert not any(name.split('.')[0] in BACKENDS for name in sys.modules)
    output = dict(status='PASS_HOST395_SAVED_GATE_REPLAY_AND_GENUINELY_MISSING_MIXED_CASE',
                  cases=results, actual395_coverage_status_changed=False,
                  reason='all16 ordinary tickets selected first; naturally promoted child1/2 tickets selected separately',
                  misleading_explanation_rejected='M80 label versus actualM120 alone does not explain the false predicate; it tests real owner count and eager membership',
                  proposal='fresh finite protocol using diagnostic SLO inputs plus real production priority/target_slo_first switches',
                  proposed_mixed_neural_qualification=False, NPU_used=False,
                  Native_qualification_flags_assigned=False, production_permission=False,
                  Raw=None, Goodput=None)
    (D / 'ROOT395_MIXED_COVERAGE_REPLAY_HOST.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(status=output['status'], replayed_cases=len(results), new_NPU_used=False)))


if __name__ == '__main__':
    main()
