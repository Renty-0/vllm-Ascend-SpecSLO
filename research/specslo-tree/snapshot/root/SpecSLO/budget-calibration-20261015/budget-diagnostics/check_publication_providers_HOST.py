"""Read all three real artifact schemas using explicitly synthetic HOST data."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from common_publication_metrics import token_digest
from read_common_publication_metrics import tree, fixed, official


def save(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(gzip.compress(raw) if path.suffix == '.gz' else raw)


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='HOST-publication-providers-', dir=root) as temporary:
        p = Path(temporary)
        requests = []
        outputs = []
        events = []
        for index in range(463):
            kind, threshold, tpot = ('tight', 40, 35) if index < 271 else ('normal', 50, 60) if index < 367 else ('loose', 150, 140)
            rid = f'HOST-fixture-{index:06d}'
            requests.append(dict(request_id=rid, slo_class=kind, slo_tpot_ms=threshold, prompt='HOST synthetic fixture'))
            tokens = list(range(256))
            outputs.append(tokens)
            events.extend([dict(request_id=rid, token_ids=tokens[:1], finished=False, elapsed_seconds=1.),
                           dict(request_id=rid, token_ids=tokens[1:], finished=True, elapsed_seconds=1+tpot*256/1000)])
        events.sort(key=lambda event:event['elapsed_seconds'])
        source = p/'manifest.jsonl'
        source.write_text(''.join(json.dumps(r)+'\n' for r in requests))
        manifest_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        results = [dict(request_id=r['request_id'], completion_token_ids=t) for r,t in zip(requests,outputs,strict=True)]
        trace = p/'tree.json.gz'
        save(trace, dict(role='full463', scope_count=463, rank=1, sealed_before=True, sealed_after=True,
                         requests=requests, generate_results=results, stream_events=events, elapsed_seconds=200,
                         manifest_sha256=manifest_sha, prefix_sha256='HOST-synthetic-not-real-data'))
        a, _, _ = tree(trace)

        baseline = p/'fixed.json'
        save(baseline, dict(request_manifest=str(source), request_manifest_sha256=manifest_sha,
             results=[dict(num_prompts=463, graph_qualification_fixed_point=True,
                           slo=dict(request_metrics=requests), output_token_ids=outputs,
                           output_token_ids_sha256=hashlib.sha256(json.dumps(outputs,separators=(',',':')).encode()).hexdigest(),
                           e2e_elapsed_seconds=200)]))
        witness = p/'fixed-witness'
        witness.mkdir()
        save(witness/'generation-000009.json.gz', dict(prompt_count=463, rank=1, actual_target_leader_rank=1,
             status='complete', callback_error=None, graph_cache_sealed_before=True, events=events, generation_ordinal=9))
        b, _, fixed_contract = fixed(baseline, witness)
        assert fixed_contract['measured_generation_ordinal'] == 9

        official_events = []
        evidence = []
        offsets = {r['request_id']:0 for r in requests}
        for packet, event in enumerate(events):
            key = event['request_id']
            official_events.append(dict(request_id=key, token_ids=event['token_ids'], finished=event['finished'],
                engine_core_timestamp=100000+event['elapsed_seconds'], token_offset=offsets[key],
                original_processor_returned=True, packet=packet))
            offsets[key] += len(event['token_ids'])
        for request in requests:
            key = request['request_id']
            own = [event for event in official_events if event['request_id'] == key]
            first, last = own[0]['engine_core_timestamp'], own[-1]['engine_core_timestamp']
            evidence.append(dict(request_id=key, output_tokens=256, output_token_ids_sha256=token_digest(list(range(256))),
                                 original_metrics_first=first, original_metrics_last=last,
                                 paper_tpot_ms=(last-first)*1000/256, observed_tpot_ms=(last-first)*1000/255))
        official_path = p/'official.json.gz'
        save(official_path, dict(schema='official-TP4.raw-engine-publication-witness.v1', status='complete', count=463,
              observer_error=None, source_rows=requests, events=official_events, clock=dict(mono_start=100000),
              evidence=dict(all_raw_tokens_equal_original_final=True, all_first_last_equal_original_engine_metrics=True,
                            original_online_elapsed_seconds=200, request_evidence=evidence)))
        c, _, _ = official(official_path)
        assert a['all']['Goodput'] == b['all']['Goodput'] == c['all']['Goodput'] == 367*256/200
        assert a['all']['Raw'] == b['all']['Raw'] == c['all']['Raw'] == 463*256/200
        assert [v['by_class']['normal']['attained'] for v in (a,b,c)] == [0,0,0]
        for x,y,z in zip(a['rows'],b['rows'],c['rows'],strict=True):
            assert x['request_id'] == y['request_id'] == z['request_id']
            assert x['output_token_ids_sha256'] == y['output_token_ids_sha256'] == z['output_token_ids_sha256']
            assert abs(x['paper_tpot_ms']-z['paper_tpot_ms']) < 1e-8
    result = dict(status='HOST_three_artifact_providers_463_requests_each_passed',
                  synthetic_requests_per_provider=463, synthetic_tokens_per_provider=118528,
                  three_Raw_and_Goodput_values_equal=True, all_output_token_hashes_equal=True,
                  actual_hardware_mapping_qualified=False, real_device_or_model_operations=0,
                  real_performance_measured=False)
    with (root/'COMMON_PUBLICATION_PROVIDERS_HOST_RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
