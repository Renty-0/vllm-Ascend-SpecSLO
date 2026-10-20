"""Exercise actual finite grants, original prefix sources and reversible deltas."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(B / 'tree-bounded-parent4-shared4-eager-diagnostic-v199'))
import base199
import budget396
import draft396
import live396
import scope396
import split_budget_v132


def main():
    original = split_budget_v132.DraftWorkLimit
    actual = []
    for case in scope396.cases():
        work = budget396.work_plan(case)
        assert work.allocated_selected_candidates == 112
        ids = list(range(24)) + list(case.eager_ids)
        assert len(set(ids)) == 28
        assert case.eager_ids == (24, 25, 32, 33)
        assert case.repair_owners == tuple(range(case.repair_count))
        assert sum(case.repair_vector) == case.repair_count
        work.require_work(ids, [4] * 28, [False] * 24 + [True] * 4,
                          [SimpleNamespace(candidate_budget=4) for _ in ids])
        work.require_published_tickets([SimpleNamespace(request_index=i, gamma=4) for i in ids])
        contexts = draft396.contexts(case)
        assert len(contexts) == 64
        assert [len(row['token_ids']) for row in contexts] == [
            129 + v for v in case.repair_vector
        ]
        assert draft396.counts(case)[:24] == [4 + v for v in case.repair_vector[:24]]
        assert live396._capability.allows_shared4(24, 4)
        assert live396._capability.allows_union(4, case.repair_count) == bool(case.repair_count)
        assert not live396._capability.allows_shared4(24, 3)
        actual.append(dict(N=24, E=4, R=case.repair_count, M=scope396.bucket(case.count),
                           frontier_M=4, union_M=case.union_M,
                           normal_C=96, eager_C=16, total_C=112))
    assert len(actual) == 25
    assert {row['M'] for row in actual} == {28}
    assert {row['R'] for row in actual} == set(range(25))
    assert max(row['union_M'] for row in actual) == 28
    for bad in (-1, 25, 64, True, 1.5):
        try:
            scope396.Case(4, bad)
        except (RuntimeError, AssertionError):
            pass
        else:
            raise AssertionError(('invalid repair count admitted', bad))
    for bad in (0, 1, 3, 5, 8, True):
        try:
            scope396.Case(bad, 0)
        except (RuntimeError, AssertionError):
            pass
        else:
            raise AssertionError(('unqualified eager count admitted', bad))
    assert split_budget_v132.DraftWorkLimit is original and original().eager_per_request == 3
    budget396.clear()
    assert split_budget_v132.DraftWorkLimit is original
    projection = json.loads((P / 'SOURCE_PROJECTION396.json').read_text())
    for name, spec in projection.items():
        if not spec.get('inverse_verified'):
            continue
        inverse = (P / name).read_text()
        for before, after, count in reversed(spec['edits']):
            assert inverse.count(after) == count, (name, after)
            inverse = inverse.replace(after, before)
        assert inverse == Path(spec['source']).read_text(), name
    source = live396.fast_source()
    ast.parse(source)
    path = P / 'fast396.generated.py'
    if path.exists():
        assert path.read_text() == source
    else:
        path.write_text(source)
    assert 'from live396 import fused_prelude,validate' in source
    assert 'from live393 ' not in source
    assert 'torch' not in sys.modules
    result = dict(status='PASS_HOST396_ACTUAL_TYPED_SCOPE_PREFIXES_AND_INVERSE',
                  NPU_used=False, Torch_imported=False, global_max_num_seqs=64,
                  cases=actual, invalid_scope_rejections=True,
                  original_production_cap3_unchanged=True,
                  original173_baseline_still_required=True,
                  changed_source_inverse_verified=True,
                  canonical_Fast396_materialized=True,
                  model_qualification=False, production_permission=False,
                  full463=False, Raw=None, Goodput=None)
    (P / 'HOST396_SCOPE_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], cases=len(actual), NPU_used=False)))


if __name__ == '__main__':
    main()
