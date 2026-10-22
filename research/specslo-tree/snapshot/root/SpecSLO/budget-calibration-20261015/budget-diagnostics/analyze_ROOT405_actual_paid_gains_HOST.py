"""Observed delivered-token distributions for actual paid budgets, not causal g.

The production selector chooses each budget in different states. These rows
cannot substitute for a controlled same-history budget calibration.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/normal24-global64-derived-controls-root-v405-full463-ownAR-v1/windows/window64/full463/rank1-full463-actual-trace.json.gz'
value = json.loads(gzip.decompress(P.read_bytes()))
seen, chunks = set(), []
for event in value['stream_events']:
    i = event['request_index']
    if i in seen:
        chunks.append(event)
    else:
        seen.add(i)
requests = {r['request_id']:r for r in value['generate_results']}
assert len(requests) == len(seen) == 463
rows = defaultdict(Counter)
terminal = defaultdict(Counter)
position = 0
delivered = 0
for call in value['normal_target_calls']:
    events = chunks[position:position+call['N']]
    position += call['N']
    assert [e['request_index'] for e in events] == call['sequence_ids'][:call['N']]
    for budget, event in zip(call['budgets'], events, strict=True):
        request = requests[event['request_id']]
        key = (request['slo_class'], budget)
        count = len(event['token_ids'])
        assert 0 < count <= budget+1
        rows[key][count] += 1
        delivered += count
        if event['finished']:
            terminal[key][count] += 1
assert position == len(chunks)
assert delivered + len(seen) == sum(len(r['completion_token_ids']) for r in requests.values()) == 463*256
result_rows = []
for (kind, budget), histogram in sorted(rows.items()):
    calls = sum(histogram.values())
    tokens = sum(k*v for k,v in histogram.items())
    uncut = histogram-terminal[(kind,budget)]
    uncut_calls = sum(uncut.values())
    result_rows.append(dict(slo_class=kind, paid_B=budget, owner_rounds=calls,
        delivered_tokens=tokens, mean_delivered_per_owner_round=tokens/calls,
        delivered_histogram=dict(histogram), terminal_clipped_owner_rounds=sum(terminal[(kind,budget)].values()),
        nonterminal_owner_rounds=uncut_calls,
        nonterminal_mean_delivered=sum(k*v for k,v in uncut.items())/uncut_calls if uncut_calls else None,
        delivered_per_paid_candidate=tokens/(calls*budget)))
out = dict(status='HOST_ACTUAL405_ALL_PAID_BUDGETS_BOUND_TO_DELIVERED_OUTPUTS', source=str(P),
    source_sha256=hashlib.sha256(P.read_bytes()).hexdigest(), owner_rounds=len(chunks),
    total_decode_tokens=delivered, first_tokens=len(seen), by_class_budget=result_rows,
    controlled_same_history_g_calibration=False,
    reason='Budget grants are selected using current urgency and learned acceptance; request states and cohorts differ. This is descriptive evidence, not causal gain from increasing B.',
    NPU_used=False, new_performance_run=False, Raw=None, Goodput=None)
path = B/'budget-diagnostics/ROOT405_ACTUAL_PAID_GAIN_DISTRIBUTION_HOST.json'
with path.open('x') as stream:
    json.dump(out, stream, indent=2); stream.write('\n')
print(json.dumps(dict(status=out['status'], by_class_budget=result_rows), indent=2))
