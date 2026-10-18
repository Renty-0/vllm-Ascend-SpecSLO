"""Read retained actual device records; make no model or performance claims."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    require(receipt['status'] in ('complete', 'failed'), 'terminal original supervisor')
    require(receipt['sources_unchanged'] and not receipt['owned_remaining'], 'immutable sources and owned cleanup')
    require(receipt['devices_after']['all_idle'], 'all four actual devices idle')
    for phase in ('pre_verifier', 'post_verifier'):
        require(receipt[phase]['actual_result']['status'] == 'pass', 'original baseline '+phase)
    inputs = [Path(__file__), args.receipt]

    def read(path):
        inputs.append(path)
        return json.loads(path.read_text())

    require(read(args.actual/'worker-exit.json') == dict(closed=True, all_exited=True), 'actual owned roles exited')
    rows = []
    per_rank = []
    for rank in (1, 2, 3):
        records = [read(p) for p in sorted(args.actual.glob(f'rank{rank}-canonical-case*.json'))]
        for record in records:
            observation = record['observation']
            require(record['rank'] == rank and record['neural_math_on_rank'] is True, 'target role identity')
            require(observation['exact_greedy'] and observation['all128_KV_original_matcher_passed'], 'unchanged complete gates')
            require(observation['candidate_tokens'] == observation['reference_tokens'], 'complete real integer heads')
            require(len(observation['candidate_tokens']) == record['case']['query_count'], 'declared real node coverage')
            require(not observation['candidate_KV_copied'], 'reference not seeded with candidate KV')
            for frame in observation['independent_reference_frames']:
                require(frame['reference_eager_only'] and not frame['candidate_KV_copied'], 'own reference execution')
        per_rank.append(records)
        bank_path = args.actual/f'rank{rank}-independent-bank.json'
        if bank_path.exists():
            bank = read(bank_path)
            rows.append(dict(rank=rank, independent_bank=bank))
    counts = [len(v) for v in per_rank]
    require(len(set(counts)) == 1, 'same complete checkpoint count on all target ranks')
    for index in range(counts[0]):
        observations = [v[index] for v in per_rank]
        require(observations[0]['case'] == observations[1]['case'] == observations[2]['case'], 'same predeclared case')
        tokens = [v['observation']['candidate_tokens'] for v in observations]
        require(tokens[0] == tokens[1] == tokens[2], 'three target ranks agree')
    failures = [read(p) for p in sorted(args.actual.glob('rank*-FAILED-*.json'))]
    negatives = [read(p) for p in sorted(args.actual.glob('rank[123]-old-layout-negative-control.json'))]
    completed = receipt['status'] == 'complete'
    if completed:
        require(counts == [155]*3 and not failures, 'all 155 actual positive cases, no retained positive failure')
        for rank in (1, 2, 3):
            final = read(args.actual/f'rank{rank}-numeric.json')
            require(final['original_greedy_sampler_unchanged'] is True, 'original sampler unchanged')
            require(final['math']['all_independent_prefixes_unchanged'], 'all 48 committed independent prefixes protected')
    output = dict(
        schema='ROOT234.independent-branch-math-terminal-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='bounded_math_completed' if completed else 'bounded_math_failed_retained',
        positive_cases_per_target_rank=counts,
        all_three_target_positive_heads_agree=True,
        banks=rows, retained_failures=failures, separate_old_layout_negative_controls=negatives,
        original_supervisor_status=receipt['status'], original_supervisor_errors=receipt.get('errors', []),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        NPU_used_by_this_auditor=False, whole_workload_AR_qualified=False,
        original_startup201_44_114_not_run=True, formal_Goodput=False, adopted=False)
    require(not args.output.exists(), 'fresh audit artifact')
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('source_sha256','banks','retained_failures','separate_old_layout_negative_controls')}))


if __name__ == '__main__':
    main()
