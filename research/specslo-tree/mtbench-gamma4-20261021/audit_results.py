"""Independently recompute acceptance and verify retained two-turn history."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    receipt = read('receipt.json')
    assert receipt['status'] == 'complete' and receipt['child_exit_code'] == 0
    assert receipt['sources_unchanged'] is True and receipt['all4idle'] is True
    assert read('worker-exit.json') == dict(closed=True, all_exited=True)
    assert read('pre-verifier.json')['status'] == read('post-verifier.json')['status'] == 'pass'
    for name, expected in receipt['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    dependencies = read('dependencies/manifest.json')
    for dependency in dependencies['files']:
        for name in ('path', 'snapshot'):
            assert hashlib.sha256(Path(dependency[name]).read_bytes()).hexdigest() == dependency['sha256']
    plan = read('run-plan.json')
    assert plan['fixed_gamma'] == 4 and plan['config']['gamma'] == 4
    assert plan['config']['spec_rhythm_min_gamma'] == 4
    assert plan['config']['draft_dtype'] == plan['config']['target_dtype'] == 'bfloat16'
    assert plan['config']['spec_rhythm_linear_bonus_token'] is False
    assert hashlib.sha256(Path(plan['dataset']).read_bytes()).hexdigest() == plan['dataset_sha256']
    questions = [json.loads(line) for line in Path(plan['dataset']).read_text().splitlines() if line.strip()]
    assert len(questions) == 80 and all(len(q['turns']) == 2 for q in questions)
    question_map = {q['question_id']: q for q in questions}
    rows = []
    for turn in (1, 2):
        phase = read(f'turn{turn}-results.json')
        assert len(phase) == 80 and {r['question_id'] for r in phase} == set(question_map)
        for row in phase:
            a, v, n = row['accepted_draft_tokens'], row['verified_draft_tokens'], row['verification_rounds']
            assert type(a) is type(v) is type(n) is int and 0 <= a <= v and n >= 0
            assert row['turn'] == turn and row['category'] == question_map[row['question_id']]['category']
            assert 0 < len(row['completion_token_ids']) <= 256
            assert row['acceptance_rate'] == (a / v if v else 0.0)
            assert row.get('temperature', 0.0) == row.get('draft_temperature', 0.0) == 0.0
        rows.extend(phase)
    assert len(rows) == 160 and all(n == 20 for n in Counter(r['category'] for r in rows).values())
    summary = read('summary.json')
    selections = [('overall', rows, summary['overall'])]
    selections.extend((f'turn{turn}', [r for r in rows if r['turn'] == turn], summary['by_turn'][str(turn)]) for turn in (1, 2))
    selections.extend((category, [r for r in rows if r['category'] == category], aggregate)
                      for category, aggregate in summary['by_category'].items())
    selections.extend((f'{category}/turn{turn}', [r for r in rows if r['category'] == category and r['turn'] == turn],
                       summary['by_category_and_turn'][category][str(turn)])
                      for category in summary['by_category_and_turn'] for turn in (1, 2))
    for label, selected, aggregate in selections:
        accepted = sum(r['accepted_draft_tokens'] for r in selected)
        verified = sum(r['verified_draft_tokens'] for r in selected)
        assert aggregate['turns'] == len(selected), label
        assert aggregate['accepted_draft_tokens'] == accepted, label
        assert aggregate['verified_draft_tokens'] == verified, label
        assert aggregate['acceptance_rate'] == (accepted / verified if verified else None), label
        rounds = sum(r['verification_rounds'] for r in selected)
        assert aggregate['verification_rounds'] == rounds, label
        assert aggregate['accepted_per_verification_round'] == (accepted / rounds if rounds else None), label
        assert aggregate['output_tokens'] == sum(len(r['completion_token_ids']) for r in selected), label
        assert aggregate['reached_output_cap'] == sum(len(r['completion_token_ids']) == 256 for r in selected), label
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan['config']['draft_model_path'], local_files_only=True)
    first = {r['question_id']: r for r in rows if r['turn'] == 1}
    for turn in (1, 2):
        inputs = read(f'turn{turn}-inputs.json')
        assert len(inputs) == 80
        for item in inputs:
            q = question_map[item['question_id']]
            expected = [dict(role='user', content=q['turns'][0])]
            if turn == 2:
                expected.extend([dict(role='assistant', content=tokenizer.decode(
                    first[q['question_id']]['completion_token_ids'], skip_special_tokens=True)),
                                 dict(role='user', content=q['turns'][1])])
            assert item['messages'] == expected
            rendered = tokenizer.apply_chat_template(expected, tokenize=False, add_generation_prompt=True)
            assert item['token_ids'] == tokenizer.encode(rendered, add_special_tokens=False)
    evidence = dict(status='pass', all80_questions_both_turns=True,
                    first_turn_actual_answers_in_second_turn=True,
                    exact_prompt_tokens_reencoded=True, independent_weighted_acceptance_recomputed=True,
                    frozen_source_and_environment_checked=True, helper_sources_unchanged=True, all4idle=True,
                    target_AR_equivalence_tested=False, new_model_calls=False,
                    overall=summary['overall'],
                    evidence_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in HERE.glob('*.json') if p.name != 'independent-audit.json'})
    (HERE / 'independent-audit.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: evidence[k] for k in ('status', 'all80_questions_both_turns', 'all4idle', 'overall')}))


if __name__ == '__main__':
    main()
