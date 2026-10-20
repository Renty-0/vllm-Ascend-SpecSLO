"""Deterministic ShareGPT first-turn acceptance diagnostic; fixed gamma=4."""
import argparse
import datetime
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path('/data/shared_datasets/strict-inputs/ShareGPT_V3_unfiltered_cleaned_split.json')
DATA_SHA = '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
SERIAL = Path('/root/SpecSLO/budget-calibration-20261015/serial-v1')
COUNT = 128
SEED = 0
MAX_TOKENS = 256
MIN_PROMPT_TOKENS = 4
MAX_PROMPT_TOKENS = 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def progress(stage, **extra):
    value = dict(stage=stage, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    save('progress.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def encode_messages(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    assert isinstance(tokens, list) and tokens and all(type(token) is int for token in tokens)
    return tokens


def summarize(rows):
    accepted = sum(row['accepted_draft_tokens'] for row in rows)
    verified = sum(row['verified_draft_tokens'] for row in rows)
    rounds = sum(row['verification_rounds'] for row in rows)
    rates = [row['accepted_draft_tokens'] / row['verified_draft_tokens']
             for row in rows if row['verified_draft_tokens']]
    return dict(requests=len(rows), accepted_draft_tokens=accepted,
                verified_draft_tokens=verified, verification_rounds=rounds,
                acceptance_rate=accepted / verified if verified else None,
                accepted_per_verification_round=accepted / rounds if rounds else None,
                output_tokens=sum(len(row['completion_token_ids']) for row in rows),
                reached_output_cap=sum(len(row['completion_token_ids']) == MAX_TOKENS for row in rows),
                mean_request_acceptance_rate=sum(rates) / len(rates) if rates else None)


def select_inputs(tokenizer):
    assert sha256(DATA) == DATA_SHA
    with DATA.open() as stream:
        corpus = json.load(stream)
    assert isinstance(corpus, list) and len(corpus) == 94145
    indices = list(range(len(corpus)))
    random.Random(SEED).shuffle(indices)
    selected = []
    inspected = 0
    for source_index in indices:
        inspected += 1
        row = corpus[source_index]
        turns = row.get('conversations', [])
        if len(turns) < 2 or turns[0].get('from') != 'human' or turns[1].get('from') != 'gpt':
            continue
        prompt = turns[0].get('value')
        reference = turns[1].get('value')
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(reference, str):
            continue
        if len(tokenizer.encode(reference, add_special_tokens=False)) < 4:
            continue
        messages = [dict(role='user', content=prompt)]
        tokens = encode_messages(tokenizer, messages)
        if not MIN_PROMPT_TOKENS <= len(tokens) <= MAX_PROMPT_TOKENS:
            continue
        selected.append(dict(ordinal=len(selected), source_index=source_index,
                             source_id=row.get('id'), messages=messages, token_ids=tokens))
        if len(selected) == COUNT:
            break
    assert len(selected) == COUNT
    assert len({row['source_index'] for row in selected}) == COUNT
    return selected, inspected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    sys.path.insert(0, str(SERIAL))
    from calibrate import config
    cfg = replace(config(), max_num_queued_seqs=160, draft_dtype='bfloat16', target_dtype='bfloat16',
                  precompile_serial_draft_graphs=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_path, local_files_only=True)
    inputs, inspected = select_inputs(tokenizer)
    assert all(len(row['token_ids']) + MAX_TOKENS + 8 <= cfg.max_model_len for row in inputs)
    if (HERE / 'inputs.json').exists():
        assert json.loads((HERE / 'inputs.json').read_text()) == inputs
    else:
        save('inputs.json', inputs)
    input_hash = sha256(HERE / 'inputs.json')
    if args.prepare_only:
        fixture = [dict(accepted_draft_tokens=1, verified_draft_tokens=4, verification_rounds=1,
                        completion_token_ids=[1]),
                   dict(accepted_draft_tokens=4, verified_draft_tokens=8, verification_rounds=2,
                        completion_token_ids=[2])]
        aggregate = summarize(fixture)
        assert aggregate['acceptance_rate'] == 5 / 12
        assert aggregate['mean_request_acceptance_rate'] == 0.375
        evidence = dict(status='pass', actual_integer_token_lists=True,
                        weighted_acceptance_aggregation_passed=True,
                        inputs_sha256=input_hash, requests=COUNT, dataset_rows=94145,
                        inspected_before_selecting=inspected, seed=SEED,
                        prompt_min=min(len(row['token_ids']) for row in inputs),
                        prompt_max=max(len(row['token_ids']) for row in inputs),
                        model_workers_started=False)
        save('HOST-preflight.json', evidence)
        print(json.dumps(evidence), flush=True)
        return
    assert json.loads((HERE / 'HOST-preflight.json').read_text())['inputs_sha256'] == input_hash
    from vllm_ascend.spec_decode.pearl.api import PEARLEngine, SamplingParams
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, draft_temperature=0.0,
                            ignore_eos=False, spec_rhythm_max_gamma=4)
    save('run-plan.json', dict(dataset=str(DATA), dataset_sha256=DATA_SHA, dataset_rows=94145,
         requests=COUNT, first_user_turn_only=True, seed=SEED, inputs_sha256=input_hash,
         sampling='Shuffle original corpus indices with random.Random(0); select first128eligible rows',
         eligibility=dict(first_roles=['human', 'gpt'], reference_min_tokens=4,
                          prompt_min_tokens=MIN_PROMPT_TOKENS, prompt_max_tokens=MAX_PROMPT_TOKENS),
         reference_answers_used_for_generation=False, max_new_tokens=MAX_TOKENS, respect_eos=True,
         temperatures=dict(target=0.0, draft=0.0), fixed_gamma=4, linear_full_window=True,
         bonus_token=False, tokenizer_chat_template='Qwen3 default; enable_thinking is not overridden',
         prompt_lengths=[len(row['token_ids']) for row in inputs], config=asdict(cfg),
         acceptance_definition='sum(actual accepted draft prefix tokens) / sum(actual verified draft candidates); target corrections excluded',
         timing_claim=False, SLO_or_Goodput_claim=False))
    engine = None
    try:
        progress('loading-models')
        engine = PEARLEngine(cfg)
        save('worker-processes.json', [dict(pid=p.pid, name=p.name) for p in engine._processes])
        for row in inputs:
            engine.add_request(row['token_ids'], params, request_id=f'sharegpt-{row["ordinal"]}')
        progress('generating', requests=COUNT, max_prompt_tokens=max(len(r['token_ids']) for r in inputs))
        started = time.perf_counter()
        outputs = engine.generate()
        wall_seconds = time.perf_counter() - started
        assert len(engine.last_metrics) == COUNT == len(outputs[0])
        results = []
        for source, metric in zip(inputs, engine.last_metrics):
            assert metric['request_id'] == f'sharegpt-{source["ordinal"]}'
            row = dict(metric, ordinal=source['ordinal'], source_index=source['source_index'],
                       source_id=source['source_id'], prompt_tokens=len(source['token_ids']))
            accepted, verified, rounds = (row[key] for key in
                                         ['accepted_draft_tokens', 'verified_draft_tokens', 'verification_rounds'])
            assert type(accepted) is type(verified) is type(rounds) is int
            assert 0 <= accepted <= verified and verified == 4 * rounds
            assert 0 < len(row['completion_token_ids']) <= MAX_TOKENS
            assert row['acceptance_rate'] == (accepted / verified if verified else 0.0)
            results.append(row)
        save('results.json', results)
        save('worker-metrics.json', engine.last_worker_metrics)
        summary = dict(schema='SpecSLO.ShareGPT.fixed-gamma4-acceptance.v1', status='complete',
                       dataset_sha256=DATA_SHA, inputs_sha256=input_hash,
                       fixed_gamma=4, max_new_tokens=MAX_TOKENS, respect_eos=True,
                       overall=summarize(results),
                       by_prompt_length={label: summarize([r for r in results if lower <= r['prompt_tokens'] <= upper])
                                         for label, lower, upper in [('4-128', 4, 128), ('129-512', 129, 512),
                                                                     ('513-1024', 513, 1024)]},
                       target_AR_equivalence_tested=False, no_Goodput_or_quality_score_claim=True)
        save('summary.json', summary)
        progress('complete', generation_wall_seconds=wall_seconds, **summary['overall'])
    finally:
        if engine is not None:
            engine.exit()
            save('worker-exit.json', dict(closed=engine._closed,
                 all_exited=all(not p.is_alive() for p in engine._processes)))


if __name__ == '__main__':
    main()
