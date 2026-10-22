"""Derive observed, selected-budget gain from actual prefix advances only.

Exclude every mixed-verification owner and every owner whose complete normal
call ledger does not match original Native verification counters. This is
observational calibration, never a counterfactual or a held-out evaluation.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
TRACE = BASE / 'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1/rank1-full463-actual-trace.json.gz'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = json.loads(gzip.decompress(TRACE.read_bytes()))
    assert raw['scope_count'] == 463 and raw['role'] == 'full463'
    assert all(value == 0 for value in raw['native']['delta'].values())
    normal = defaultdict(list)
    for ordinal, call in enumerate(raw['normal_target_calls']):
        n = call['N']
        assert len(call['budgets']) == len(call['contexts']) == n
        for owner, budget, context in zip(call['sequence_ids'][:n], call['budgets'], call['contexts'], strict=True):
            normal[owner].append(dict(ordinal=ordinal, B=budget, context=context,
                                      physical_M=call['physical_M'], physical_S=call['physical_S'],
                                      family=call['family']))
    mixed = {owner for event in raw['logical_events'] if event['kind'] == 'plan'
             for owner in event['target_ids']}
    values, owners, excluded, histograms = defaultdict(list), defaultdict(set), Counter(), defaultdict(Counter)
    retained_owners, terminal_acceptance_clipping = [], {}
    for owner, result in enumerate(raw['generate_results']):
        calls = normal[owner]
        if owner in mixed:
            excluded['owner_participated_in_mixed_verification'] += 1
            continue
        if (len(calls) != result['verification_rounds']
                or sum(call['B'] for call in calls) != result['verified_draft_tokens']):
            excluded['owner_native_verification_ledger_not_equal'] += 1
            continue
        prompt_length = len(raw['tokenized_inputs'][owner])
        assert calls and calls[0]['context'] == prompt_length
        contexts = [call['context'] for call in calls] + [prompt_length + 255]
        advances = [b-a for a, b in zip(contexts, contexts[1:])]
        assert sum(advances) == 255
        if any(not 1 <= advance <= min(call['B'], 4)+1
               for advance, call in zip(advances, calls, strict=True)):
            excluded['owner_unexplained_prefix_advance'] += 1
            continue
        # Native counts accepted draft nodes before clipping the final output
        # to 256 tokens. Prefix advances describe the actual clipped output.
        accepted_difference = result['accepted_draft_tokens'] - sum(advance-1 for advance in advances)
        assert 0 <= accepted_difference <= min(calls[-1]['B'], 4) + 1 - advances[-1]
        if accepted_difference:
            terminal_acceptance_clipping[owner] = accepted_difference
        retained_owners.append(owner)
        for advance, call in zip(advances, calls, strict=True):
            remaining = prompt_length + 255 - call['context']
            if remaining < 5:
                excluded['terminal_horizon_round'] += 1
                continue
            key = (result['slo_class'], call['B'])
            values[key].append(advance)
            owners[key].add(owner)
            histograms[key][advance] += 1
    curves = {}
    for (cls, budget), gains in sorted(values.items()):
        curves.setdefault(cls, {})[str(budget)] = dict(
            actual_rounds=len(gains), actual_owners=len(owners[cls, budget]),
            mean_delivered_tokens=statistics.mean(gains), delivered_histogram=dict(sorted(histograms[cls, budget].items())))
    result = dict(schema='ROOT.V96.observed-selected-budget-gain.v1',
                  UTC=datetime.now(timezone.utc).isoformat(), source_script_sha256=digest(Path(__file__)),
                  trace=str(TRACE), trace_sha256=digest(TRACE), retained_owners=retained_owners,
                  excluded=dict(excluded), curves=curves,
                  complete_native_verification_ledger_matched=True,
                  complete_prefix_advances_matched=True,
                  accepted_counts_consistent_with_final_output_clipping=True,
                  accepted_nodes_beyond_final_output=terminal_acceptance_clipping,
                  terminal_horizon_excluded=True,
                  same_context_counterfactual=False, allocation_selection_bias_present=True,
                  contexts_are_correlated_within_requests=True,
                  heldout_validated=False, new_model_or_NPU_execution=False,
                  replacement_policy_installed=False, performance_gain_proved=False)
    path = BASE / 'budget-diagnostics/V96_ACTUAL_SELECTED_BUDGET_GAIN.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(retained_owners=len(retained_owners), excluded=dict(excluded), curves=curves)))


if __name__ == '__main__':
    main()
