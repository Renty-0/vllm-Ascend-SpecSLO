"""Read-only full-workload, official-source, MC2 and publication audit."""
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path('/root/SpecSLO/official-v021-baseline-20261019')
DIAGNOSTICS = Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')


def audit(run):
    run = Path(run)
    document = json.loads((run / 'result.json').read_text())
    assert len(document['results']) == 1
    result = document['results'][0]
    manifest = Path(document['request_manifest'])
    requested = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    measured = result['slo']['request_metrics']
    evidence = json.loads((run / 'model-evidence.json').read_text())
    environment = json.loads((run / 'launch-environment.json').read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == document['request_manifest_sha256'] == 'dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662'
    assert len(requested) == len(measured) == len(result['output_token_ids']) == 463
    assert result['prompt_token_ids_sha256'] == '6f543c04a4580ae9363c4469504847a708b277d13eb708f71975887389f21815'
    assert all(len(row) == 256 for row in result['output_token_ids'])
    assert result['output_tokens'] == 118528
    assert document['tensor_parallel_size'] == 4 and document['speculative_model'] is None and document['dtype'] == 'bfloat16'
    assert document['online_arrivals'] and document['prefill_coalesce_min_requests'] == 1 and document['prefill_coalesce_max_wait_ms'] == 0
    assert not document['respect_eos'] and document['warmup_excluded_from_measurement']
    assert len(evidence) == 4 and {row['rank'] for row in evidence} == {0, 1, 2, 3}
    assert environment['ascend_commit'] == '80610e4438dba05011b05f89fc45d91e96992671'
    assert environment['vllm_commit'] == 'ad7125a431e176d4161099480a66f0169609a690'
    assert environment['physical_devices'] == [2, 5, 6, 7]
    for row in evidence:
        assert Path(row['vllm_source']).resolve().is_relative_to(ROOT / 'vllm')
        assert Path(row['vllm_ascend_source']).resolve().is_relative_to(ROOT / 'vllm-ascend')
        assert row['enabled'] and row['selected_modules'] == 128 and row['quantization'] is None
    assert environment['environment']['VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE'] == '1'
    assert environment['environment']['VLLM_ASCEND_ENABLE_FUSED_MC2'] == '0'
    attained = Counter()
    tpot = {}
    max_error = 0.0
    for original, row in zip(requested, measured, strict=True):
        assert all(row[k] == original[k] for k in ('request_id', 'slo_class', 'slo_tpot_ms')) and row['output_tokens'] == 256
        first, last = row['first_token_ts'], row['last_token_ts']
        assert math.isfinite(first) and math.isfinite(last) and last >= first
        value = (last - first) * 1000 / 256
        max_error = max(max_error, abs(value - row['paper_tpot_ms']))
        tpot.setdefault(original['slo_class'], []).append(value)
        if value <= original['slo_tpot_ms']:
            attained[original['slo_class']] += 1
    wall = result['e2e_elapsed_seconds']
    assert math.isfinite(wall) and wall > 0 and max_error <= 1e-10
    gp, raw = sum(attained.values()) * 256 / wall, 118528 / wall
    assert math.isclose(gp, result['slo']['paper']['goodput_tokens_per_e2e_second'], rel_tol=1e-12)
    assert math.isclose(raw, result['output_throughput_tokens_per_second'], rel_tol=1e-12)
    sys.path.insert(0, str(DIAGNOSTICS))
    from final_common_comparison import collect
    witnesses = list((ROOT / 'publication-witness/actual' / run.name).glob('generation-*.json.gz'))
    assert len(witnesses) == 1
    common = collect('official_MC2_ON', witnesses[0])
    assert math.isclose(common['all']['Goodput'], gp, rel_tol=1e-12)
    assert math.isclose(common['all']['Raw'], raw, rel_tol=1e-12)
    summary = dict(passed=True, errors=[], paper_Goodput=gp, raw_throughput=raw, e2e_seconds=wall,
                   attained_requests=sum(attained.values()), attained_by_class=dict(attained),
                   TPOT_median_ms_by_class={k: statistics.median(v) for k, v in tpot.items()},
                   MC2_enabled=True, selected_MC2_modules_per_rank=[row['selected_modules'] for row in evidence],
                   physical_devices=[2, 5, 6, 7], ascend_commit=environment['ascend_commit'],
                   vllm_commit=environment['vllm_commit'], packages=environment['packages'],
                   max_TPOT_recompute_error_ms=max_error,
                   result_sha256=hashlib.sha256((run / 'result.json').read_bytes()).hexdigest(),
                   publication_witness=str(witnesses[0]), common_publication_metrics=common,
                   scope='One complete fresh old official engine; no cross-method AR token-equivalence claim')
    output = run / 'independent-audit.json'
    with output.open('x') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return summary


if __name__ == '__main__':
    row = audit(sys.argv[1])
    print(json.dumps({k: v for k, v in row.items() if k != 'common_publication_metrics'}))
