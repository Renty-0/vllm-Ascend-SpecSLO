"""Enumerate all effective N<=16,C<=64 B1..5 geometries, without a device."""
import hashlib
import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parent.parent
P = B / 'tree-contiguous-branch-tail-final-metadata-math-v234'
sys.path[:0] = [str(P), str(B / 'tree-high-normal-cap4-native-families-v148')]
from mapper231 import envelope
from neural_oracle231 import cases


def key(n, budgets):
    forks = budgets.count(5)
    m, segments, _ = envelope(n, n + sum(budgets), 'full9' if forks else 'causal')
    return (m, segments + forks, 10 if forks else 9)


def main():
    previous = {(0, 0): ()}
    witnesses = {}
    geometries = []
    for n in range(1, 17):
        current = {}
        for (cost, forks), budgets in previous.items():
            for budget in range(1, 6):
                total = cost + budget
                if total <= 64 and n + total <= 80:
                    current.setdefault((total, forks + (budget == 5)), budgets + (budget,))
        for (cost, forks), budgets in sorted(current.items()):
            actual = key(n, list(budgets))
            witnesses.setdefault(actual, dict(N=n, candidates=cost, forks=forks, budgets=list(budgets)))
            geometries.append(dict(N=n, candidates=cost, forks=forks, key=list(actual)))
        previous = current
    tested = {key(case['N'], case['budgets']) for case in cases()}
    missing = sorted(set(witnesses) - tested)
    result = dict(
        status='HOST_exhaustive_count_geometry_enumeration_only',
        effective_contract=dict(max_target_requests=16, max_verification_candidates=64,
                                max_logical_queries=80, min_B=1, max_B=5),
        reachable_geometries=len(geometries),
        key_fields=['physical_queries', 'attention_segments', 'mask_query_capacity'],
        keys=[dict(key=list(k), witness=v) for k, v in sorted(witnesses.items())],
        existing_155_case_keys=[list(k) for k in sorted(tested)],
        missing_keys=[dict(key=list(k), witness=witnesses[k]) for k in missing],
        all_four_fork_anchors_required_for_each_new_branch_key=True,
        other_graph_key_dimensions_and_dynamic_lengths_still_require_original_Native_validation=True,
        runtime_installed=False, NPU_qualified=False, performance_qualified=False,
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), P/'mapper231.py', P/'neural_oracle231.py']})
    output = Path(__file__).with_name('ROOT231_EFFECTIVE_C64_M80_KEY_DOMAIN_HOST.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources_sha256','keys')}))


if __name__ == '__main__':
    main()
