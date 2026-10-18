"""Read private retained artifacts; never launch models or infer AR validity."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from common_publication_metrics import compute, need, token_digest


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def descriptor(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def tree(path):
    value = read(path)
    need(value['role'] == 'full463' and value['scope_count'] == 463 and value['rank'] == 1,
         'actual full463 output-leader trace')
    need(value['sealed_before'] is True and value['sealed_after'] is True, 'naturally sealed measured tree generation')
    outputs = {r['request_id']:r['completion_token_ids'] for r in value['generate_results']}
    result = compute(value['requests'], outputs, value['stream_events'], value['elapsed_seconds'],
                     clock_field='elapsed_seconds', method='SpecSLO_dynamic_tree', clock_origin=0)
    return result, [path], dict(manifest_sha256=value['manifest_sha256'], prefix_sha256=value['prefix_sha256'])


def fixed(path, witness_directory):
    payload = read(path)
    cells = [c for c in payload['results'] if c['num_prompts'] == 463]
    need(len(cells) == 1, 'one preselected measured fixed generation')
    cell = cells[0]
    need(cell['graph_qualification_fixed_point'] is True, 'original fixed graph qualification complete')
    rows = cell['slo']['request_metrics']
    tokens = cell['output_token_ids']
    need(len(rows) == len(tokens) == 463, 'all original fixed outputs')
    need(hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest() == cell['output_token_ids_sha256'],
         'actual fixed final token rows checksum')
    manifest = Path(payload['request_manifest'])
    need(hashlib.sha256(manifest.read_bytes()).hexdigest() == payload['request_manifest_sha256'], 'actual fixed request manifest bytes')
    requests = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    need(len(requests) == 463 and [r['request_id'] for r in requests] == [r['request_id'] for r in rows],
         'same actual fixed request order')
    for request, row in zip(requests, rows, strict=True):
        need(all(request[k] == row[k] for k in ('slo_class', 'slo_tpot_ms')), 'fixed actual SLO matches manifest')
    matches = []
    for candidate in sorted(Path(witness_directory).rglob('generation-*.json.gz')):
        value = read(candidate)
        if value.get('prompt_count') == 463 and value.get('rank') == value.get('actual_target_leader_rank') == 1:
            matches.append((candidate, value))
    need(len(matches) == 1, 'one actual full fixed worker witness, no fastest-run selection')
    witness_path, witness = matches[0]
    need(witness['status'] == 'complete' and witness['callback_error'] is None
         and witness['graph_cache_sealed_before'] is True, 'complete sealed fixed stream witness')
    outputs = {row['request_id']:token for row, token in zip(rows, tokens, strict=True)}
    result = compute(requests, outputs, witness['events'], cell['e2e_elapsed_seconds'],
                     clock_field='elapsed_seconds', method='fixed_gamma4', clock_origin=0)
    return result, [path, manifest, witness_path], dict(manifest_sha256=payload['request_manifest_sha256'],
        measured_generation_ordinal=witness['generation_ordinal'], gzip_included_in_original_API_wall=True)


def official(path):
    value = read(path)
    need(value['schema'] == 'official-TP4.raw-engine-publication-witness.v1'
         and value['status'] == 'complete' and value['count'] == 463
         and value['observer_error'] is None, 'complete actual official engine-publication witness')
    evidence = value['evidence']
    need(evidence['all_raw_tokens_equal_original_final'] is True
         and evidence['all_first_last_equal_original_engine_metrics'] is True,
         'actual observer-to-original final token and clock checks passed')
    rows = evidence['request_evidence']
    outputs = {r['request_id']:dict(output_tokens=r['output_tokens'], sha256=r['output_token_ids_sha256']) for r in rows}
    events = value['events']
    offsets = {key:0 for key in outputs}
    for event in events:
        key = event['request_id']
        need(key in offsets and event['token_offset'] == offsets[key]
             and event['original_processor_returned'] is True, 'actual official packet ownership and token offsets')
        offsets[key] += len(event['token_ids'])
    result = compute(value['source_rows'], outputs, events, evidence['original_online_elapsed_seconds'],
                     clock_field='engine_core_timestamp', method='official_MC2_ON',
                     clock_origin=value['clock']['mono_start'])
    retained = {r['request_id']:r for r in rows}
    for row in result['rows']:
        original = retained[row['request_id']]
        need(row['first_publication'] == original['original_metrics_first']
             and row['last_publication'] == original['original_metrics_last'], 'original engine first/last clocks preserved')
        need(row['paper_tpot_ms'] == original['paper_tpot_ms']
             and row['observed_tpot_ms'] == original['observed_tpot_ms'], 'independent exact official TPOT arithmetic')
    return result, [path], dict(original_engine_metrics_clock=True, observer_receive_clock_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('tree', 'fixed', 'official'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--witness-directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'fresh private normalized metrics artifact')
    if args.kind == 'fixed':
        need(args.witness_directory is not None, 'fixed actual worker witness directory is required')
        result, paths, contract = fixed(args.source, args.witness_directory)
    else:
        result, paths, contract = (tree if args.kind == 'tree' else official)(args.source)
    paths += [Path(__file__), Path(__file__).with_name('common_publication_metrics.py')]
    result.update(source_descriptors=[descriptor(p) for p in paths], input_contract=contract,
                  HOST_arithmetic_only=True, AR_qualification_inferred=False, performance_stability_inferred=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(method=result['method'], **result['all'])))


if __name__ == '__main__':
    main()
