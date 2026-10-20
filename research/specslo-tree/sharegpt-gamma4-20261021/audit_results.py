"""CPU-only independent reconstruction of ShareGPT selection and acceptance."""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    receipt = read('receipt.json')
    assert receipt['status'] == 'complete' and receipt['child_exit_code'] == 0
    assert receipt['sources_unchanged'] and receipt['all4idle']
    assert read('worker-exit.json') == dict(closed=True, all_exited=True)
    assert read('pre-verifier.json')['status'] == read('post-verifier.json')['status'] == 'pass'
    for name, expected in receipt['source_hashes'].items():
        assert digest(name) == expected
    for row in read('dependencies/manifest.json')['files']:
        assert digest(row['path']) == digest(row['snapshot']) == row['sha256']
    plan = read('run-plan.json')
    cfg = plan['config']
    assert plan['fixed_gamma'] == cfg['gamma'] == cfg['spec_rhythm_min_gamma'] == 4
    assert cfg['draft_dtype'] == cfg['target_dtype'] == 'bfloat16'
    assert cfg['spec_rhythm_linear_full_window'] and not cfg['spec_rhythm_linear_bonus_token']
    assert cfg['precompile_serial_draft_graphs'] and plan['respect_eos']
    assert plan['temperatures'] == dict(target=0.0, draft=0.0)
    assert digest(plan['dataset']) == plan['dataset_sha256']
    assert digest(HERE / 'inputs.json') == plan['inputs_sha256'] == read('HOST-preflight.json')['inputs_sha256']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['draft_model_path'], local_files_only=True)
    with Path(plan['dataset']).open() as stream:
        corpus = json.load(stream)
    assert len(corpus) == plan['dataset_rows'] == 94145
    indices = list(range(len(corpus)))
    random.Random(plan['seed']).shuffle(indices)
    reconstructed = []
    for index in indices:
        source = corpus[index]
        turns = source.get('conversations', [])
        if len(turns) < 2 or turns[0].get('from') != 'human' or turns[1].get('from') != 'gpt':
            continue
        text, reference = turns[0].get('value'), turns[1].get('value')
        if not isinstance(text, str) or not text.strip() or not isinstance(reference, str):
            continue
        if len(tokenizer.encode(reference, add_special_tokens=False)) < 4:
            continue
        messages = [dict(role='user', content=text)]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        if not 4 <= len(tokens) <= 1024:
            continue
        reconstructed.append(dict(ordinal=len(reconstructed), source_index=index, source_id=source.get('id'),
                                  messages=messages, token_ids=tokens))
        if len(reconstructed) == 128:
            break
    assert reconstructed == read('inputs.json') and len(reconstructed) == 128
    results = read('results.json')
    assert len(results) == 128
    for expected, row in zip(reconstructed, results):
        for name in ('ordinal', 'source_id', 'source_index'):
            assert row[name] == expected[name]
        assert row['request_id'] == f'sharegpt-{expected["ordinal"]}'
        assert row['prompt_tokens'] == len(expected['token_ids'])
        a, v, n = row['accepted_draft_tokens'], row['verified_draft_tokens'], row['verification_rounds']
        assert type(a) is type(v) is type(n) is int and 0 <= a <= v and v == 4 * n
        assert 0 < len(row['completion_token_ids']) <= 256
        assert row['acceptance_rate'] == (a / v if v else 0.0)
    summary = read('summary.json')
    groups = [('overall', results, summary['overall'])]
    for label, lower, upper in [('4-128', 4, 128), ('129-512', 129, 512), ('513-1024', 513, 1024)]:
        groups.append((label, [r for r in results if lower <= r['prompt_tokens'] <= upper],
                       summary['by_prompt_length'][label]))
    for label, rows, aggregate in groups:
        accepted = sum(r['accepted_draft_tokens'] for r in rows)
        verified = sum(r['verified_draft_tokens'] for r in rows)
        rounds = sum(r['verification_rounds'] for r in rows)
        assert aggregate['requests'] == len(rows), label
        assert aggregate['accepted_draft_tokens'] == accepted, label
        assert aggregate['verified_draft_tokens'] == verified, label
        assert aggregate['verification_rounds'] == rounds, label
        assert aggregate['acceptance_rate'] == (accepted / verified if verified else None), label
        assert aggregate['accepted_per_verification_round'] == (accepted / rounds if rounds else None), label
        assert aggregate['output_tokens'] == sum(len(r['completion_token_ids']) for r in rows), label
        assert aggregate['reached_output_cap'] == sum(len(r['completion_token_ids']) == 256 for r in rows), label
    names = ['receipt.json', 'worker-exit.json', 'pre-verifier.json', 'post-verifier.json',
             'run-plan.json', 'inputs.json', 'results.json', 'summary.json', 'HOST-preflight.json']
    evidence = dict(status='pass', requests=128, deterministic_corpus_selection_reconstructed=True,
                    exact_prompt_tokens_reencoded=True, fixed_gamma4_every_verification=True,
                    independent_weighted_acceptance_recomputed=True,
                    frozen_source_and_environment_checked=True, helper_sources_unchanged=True,
                    all4idle=True, new_model_calls=False, target_AR_equivalence_tested=False,
                    audit_script_sha256=digest(__file__), overall=summary['overall'],
                    evidence_hashes={name: digest(HERE / name) for name in names})
    (HERE / 'independent-audit.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: evidence[k] for k in ['status', 'requests', 'all4idle', 'overall']}), flush=True)


if __name__ == '__main__':
    main()
