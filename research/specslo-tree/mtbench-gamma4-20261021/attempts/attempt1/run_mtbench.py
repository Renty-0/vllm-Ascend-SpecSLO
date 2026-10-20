"""MT-Bench fixed-gamma4 acceptance diagnostic on the retained model pair."""
import datetime
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path('/data/shared_upload/SpecSLO/datasets/MT-Bench/question-b494d0c6b4e7.jsonl')
DATA_SHA = '119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7'
SERIAL = Path('/root/SpecSLO/budget-calibration-20261015/serial-v1')
MAX_TOKENS = 256


def save(name, value):
    path = HERE / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def progress(stage, **extra):
    value = dict(stage=stage, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    save('progress.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def summarize(rows):
    accepted = sum(row['accepted_draft_tokens'] for row in rows)
    verified = sum(row['verified_draft_tokens'] for row in rows)
    rounds = sum(row['verification_rounds'] for row in rows)
    return dict(turns=len(rows), accepted_draft_tokens=accepted,
                verified_draft_tokens=verified, acceptance_rate=accepted / verified if verified else None,
                verification_rounds=rounds,
                accepted_per_verification_round=accepted / rounds if rounds else None,
                output_tokens=sum(len(row['completion_token_ids']) for row in rows),
                reached_output_cap=sum(len(row['completion_token_ids']) == MAX_TOKENS for row in rows),
                mean_request_acceptance_rate=sum(row['accepted_draft_tokens'] / row['verified_draft_tokens']
                                                 for row in rows if row['verified_draft_tokens']) /
                                             sum(bool(row['verified_draft_tokens']) for row in rows)
                                             if any(row['verified_draft_tokens'] for row in rows) else None)


def encode_messages(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    values = tokenizer.encode(rendered, add_special_tokens=False)
    assert isinstance(values, list) and values and all(type(token) is int for token in values)
    return values


def main():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == DATA_SHA
    questions = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    assert len(questions) == 80 and all(len(row['turns']) == 2 for row in questions)
    sys.path.insert(0, str(SERIAL))
    from calibrate import config
    from vllm_ascend.spec_decode.pearl.api import PEARLEngine, SamplingParams
    from transformers import AutoTokenizer
    cfg = replace(config(), max_num_queued_seqs=160, draft_dtype='bfloat16', target_dtype='bfloat16')
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, draft_temperature=0.0,
                            ignore_eos=False, spec_rhythm_max_gamma=4)
    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_path)
    first_messages = [[dict(role='user', content=q['turns'][0])] for q in questions]
    first_ids = [encode_messages(tokenizer, messages) for messages in first_messages]
    assert all(len(tokens) + MAX_TOKENS + 8 <= cfg.max_model_len for tokens in first_ids)
    save('run-plan.json', dict(dataset=str(DATA), dataset_sha256=DATA_SHA,
         questions=80, turns_per_question=2, max_new_tokens=MAX_TOKENS,
         respect_eos=True, temperatures=dict(target=0.0, draft=0.0), fixed_gamma=4,
         linear_full_window=True, bonus_token=False, categories=dict(Counter(q['category'] for q in questions)),
         second_turn_history='actual first-turn generated assistant text using the same fixed-gamma run',
         tokenizer_chat_template='Qwen3 tokenizer default; enable_thinking is not overridden',
         first_prompt_lengths=[len(tokens) for tokens in first_ids], config=asdict(cfg),
         acceptance_definition='sum(actual accepted draft prefix tokens) / sum(actual verified draft candidates); target corrections excluded',
         timing_claim=False, SLO_or_Goodput_claim=False, MTBench_quality_score=False))
    engine = None
    retained = []
    try:
        progress('loading-models')
        engine = PEARLEngine(cfg)
        save('worker-processes.json', [dict(pid=p.pid, name=p.name) for p in engine._processes])
        progress('models-loaded')
        for turn in (1, 2):
            if turn == 1:
                messages, token_rows = first_messages, first_ids
            else:
                answers = {row['question_id']: tokenizer.decode(row['completion_token_ids'], skip_special_tokens=True)
                           for row in retained if row['turn'] == 1}
                messages = [first_messages[i] + [dict(role='assistant', content=answers[q['question_id']]),
                                                 dict(role='user', content=q['turns'][1])]
                            for i, q in enumerate(questions)]
                token_rows = [encode_messages(tokenizer, value) for value in messages]
            assert all(len(tokens) + MAX_TOKENS + 8 <= cfg.max_model_len for tokens in token_rows)
            save(f'turn{turn}-inputs.json', [dict(question_id=q['question_id'], category=q['category'],
                  messages=message, token_ids=tokens) for q, message, tokens in zip(questions, messages, token_rows)])
            for q, tokens in zip(questions, token_rows):
                engine.add_request(tokens, params, request_id=f'mtbench-{q["question_id"]}-turn{turn}')
            progress(f'turn{turn}-generating', questions=80, max_prompt_tokens=max(map(len, token_rows)))
            started = time.perf_counter()
            outputs = engine.generate()
            elapsed = time.perf_counter() - started
            metrics = engine.last_metrics
            assert len(metrics) == len(questions) == len(outputs[0])
            phase = []
            for q, metric in zip(questions, metrics):
                row = dict(metric, question_id=q['question_id'], category=q['category'], turn=turn)
                accepted, verified = row['accepted_draft_tokens'], row['verified_draft_tokens']
                assert type(accepted) is int and type(verified) is int and 0 <= accepted <= verified
                assert 0 < len(row['completion_token_ids']) <= MAX_TOKENS
                assert abs(row['acceptance_rate'] - (accepted / verified if verified else 0.0)) < 1e-12
                phase.append(row)
            save(f'turn{turn}-results.json', phase)
            save(f'turn{turn}-worker-metrics.json', engine.last_worker_metrics)
            retained.extend(phase)
            progress(f'turn{turn}-complete', measured_wall_seconds=elapsed, **summarize(phase))
        summary = dict(schema='SpecSLO.MTBench.fixed-gamma4-acceptance.v1', status='complete',
                       dataset_sha256=DATA_SHA, fixed_gamma=4, max_new_tokens=MAX_TOKENS,
                       respect_eos=True, overall=summarize(retained),
                       by_turn={str(t): summarize([r for r in retained if r['turn'] == t]) for t in (1, 2)},
                       by_category={c: summarize([r for r in retained if r['category'] == c])
                                    for c in sorted({q['category'] for q in questions})},
                       by_category_and_turn={c: {str(t): summarize([r for r in retained if r['category'] == c and r['turn'] == t])
                                                for t in (1, 2)} for c in sorted({q['category'] for q in questions})},
                       all_80_two_turn=True, target_AR_equivalence_tested=False,
                       no_Goodput_or_MTBench_quality_score_claim=True)
        save('summary.json', summary)
        progress('complete', **summary['overall'])
    finally:
        if engine is not None:
            engine.exit()
            save('worker-exit.json', dict(closed=engine._closed,
                 all_exited=all(not process.is_alive() for process in engine._processes)))


if __name__ == '__main__':
    main()
