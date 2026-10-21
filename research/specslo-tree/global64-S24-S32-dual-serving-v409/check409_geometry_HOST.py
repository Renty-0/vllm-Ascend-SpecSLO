"""Raw geometry reconstruction, real old witnesses, and wide corruptions."""
import copy
import gzip
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent


def main():
    sys.path.insert(0, str(B / 'tree-M120-S32-frugal-independent-target-host-v399'))
    from bootstrap399 import install
    install()
    sys.path.insert(0, str(P))
    from geometry409 import valid, audit_pair, query_ends_wide
    from bucket409 import extent
    donor = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/actual'
    old = donor / 'active64-whole-ticket-S24-segments-root-v398-full463-ownAR-v1'
    report = json.loads((old / 'rank1-local-diagnostic.json').read_text())
    with gzip.open(old / 'windows/window64/full463/rank1-full463-actual-trace.json.gz', 'rt') as stream:
        trace = json.load(stream)
    records = report['ROOT257_actual_geometry']['actual_calls_by_generation'][str(trace['source_generation'])]
    retained = 0
    for record, call in zip(records, trace['normal_target_calls'], strict=True):
        row = dict(record, budgets=list(call['budgets']),
                   uniform_KV_bucket409=record['uniform_KV_bucket398'])
        assert valid(row) and audit_pair(row, call)
        retained += 1
    from mapper399 import envelope
    from projection397 import plan
    wide = 0
    for n in range(25, 33):
        for budgets in ([1] * n, [3] * 24 + [2] * (n - 24),
                        [2] * (n - 24) + [3] * 24):
            q = n + sum(budgets)
            m, s, padding = envelope(n, q, 'causal')
            parts = [b + 1 for b in budgets] + padding
            total = 0
            ends = []
            for size in parts:
                total += size
                ends.append(total)
            mapping = plan(ends, q)
            contexts = [128 + 17 * i for i in range(n)]
            lengths = [v + b + 1 for v, b in zip(contexts, budgets)] + padding
            selected = extent(tuple(lengths))
            row = dict(N=n, C=sum(budgets), M=120, logical_queries=q,
                budgets=list(budgets), branches=0, actual_copy_slots=0,
                copied=False, mask_capacity=10, attention_segments=32,
                primary_side_groups=n, actual_q_partition=query_ends_wide(),
                physical_to_original=mapping['physical_to_original'],
                original_to_physical=mapping['original_to_physical'],
                fixed_KV_extent=[selected] * 32,
                uniform_KV_bucket409=dict(enabled=True, extent=selected,
                    original_sequence_lens=lengths))
            call = dict(N=n, budgets=list(budgets), physical_M=m, physical_S=s,
                        real_query_count=q, physical_query_partition=parts,
                        contexts=contexts, family='causal')
            assert valid(row) and audit_pair(row, call)
            broken = copy.deepcopy(row)
            broken['physical_to_original'] = list(reversed(broken['physical_to_original']))
            try:
                audit_pair(broken, call)
            except RuntimeError:
                pass
            else:
                raise AssertionError('corrupted full-row permutation accepted')
            broken = copy.deepcopy(row)
            broken['actual_copy_slots'] = 1
            assert valid(broken) is False
            wide += 1
    bad = dict(row, budgets=[3] * 25 + [1] * 7, C=82, logical_queries=114)
    assert valid(bad) is False
    assert not any(name in sys.modules for name in ('torch', 'torch_npu'))
    result = dict(status='PASS_HOST409_REAL_2690_NARROW_AND24_WIDE_GEOMETRY_CORRUPTIONS',
        retained_actual398_narrow_calls_checked=retained,
        new_wide_structural_witnesses=wide,
        actual_extent_rebuilt_from_real_contexts=True,
        full_row_permutation_and_unqualified_copy_and_B3_quota_corruptions=True,
        no_neural_qualification_inherited=True, NPU_used=False,
        production_permission=False, Raw=None, Goodput=None)
    (P / 'HOST409_GEOMETRY_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
