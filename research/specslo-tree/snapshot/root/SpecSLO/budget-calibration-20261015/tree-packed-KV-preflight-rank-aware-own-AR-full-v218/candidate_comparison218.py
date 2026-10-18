"""Load original output-leader completions solely for postfull comparison.

Native generate returns None on every non-output rank. All ranks retain
their original input/cohort traces, and all independently generate their own
reference histories. No candidate token is supplied to a reference model.
"""
import gzip
import hashlib
import json
from pathlib import Path


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT218 candidate comparison: ' + message)


def load_candidate(artifact, own_trace, output_rank):
    need(output_rank == 1 and own_trace['rank'] in range(4), 'original fixed output-leader1 topology')
    artifact = Path(artifact)
    need(artifact.name == f"rank{own_trace['rank']}-full463-actual-trace.json.gz", 'own original trace identity')
    path = artifact.with_name('rank1-full463-actual-trace.json.gz')
    raw = path.read_bytes()
    leader = json.loads(gzip.decompress(raw))
    need(leader['rank'] == output_rank and leader['scope_count'] == 463, 'actual original output-leader full463 trace')
    for key in ('scope_count', 'manifest_sha256', 'prefix_sha256', 'requests',
                'tokenized_inputs', 'source_generation', 'standalone_prefill_calls'):
        need(own_trace[key] == leader[key], 'own/leader original input and cohort identity ' + key)
    results = leader['generate_results']
    need(type(results) is list and len(results) == 463, 'original output leader alone materializes all results')
    need((own_trace['generate_results'] is None) if own_trace['rank'] != output_rank
         else (own_trace['generate_results'] == results), 'original non-output None result ownership unchanged')
    candidate = {r['request_id']: r['completion_token_ids'] for r in results}
    need(len(candidate) == 463 and set(candidate) == {r['request_id'] for r in own_trace['requests']}
         and all(type(v) is list and len(v) == 256 and all(type(t) is int and 0 <= t < 151936 for t in v)
                 for v in candidate.values()), 'complete unique original463x256 integer candidate outputs')
    return candidate, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
