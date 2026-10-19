"""Check bounded diagnostics serialize the new causal predictor accurately."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import check346
from debug_snapshot346 import maybe_save
from risk294 import blank


def main():
    state = dict(generation=5, classes={c: blank() for c in ('tight', 'normal', 'loose')},
                 requests={}, history={'1': dict(classes={}, requests={})},
                 terminal_skipped=0, revision=0, original_gain=object())
    policy = SimpleNamespace(_ROOT294=state, counts={'old_counter': 7})
    with tempfile.TemporaryDirectory() as temporary:
        engine = SimpleNamespace(rank=0, _ROOT346_policy=policy,
            _ROOT346_debug_output=temporary, _ROOT346_debug_observations=255,
            _dynamic_tree_generation=5, _ROOT346_draft_counts={},
            _ROOT346_request_gain_observations={})
        request = SimpleNamespace(request_index=17)
        plan = SimpleNamespace(candidate_budget=4)
        maybe_save(engine, request, plan, [-1]*4, [20, -1, -1, -1, -1])
        path = Path(temporary) / 'rank0-online-state-latest.json'
        saved = path.read_bytes()
        row = json.loads(saved)
        assert row['generation'] == 5 and row['last_verdict']['request'] == 17
        assert row['observations_since_process_start'] == 256
        assert set(row['paired_gain']['by_generation']) == {'1', '5'}
        assert 'original_gain' not in row['paired_gain']
        assert row['paired_gain']['predictor'].startswith('request_conditional_risk294')
        assert policy.counts == {'old_counter': 7}
        maybe_save(engine, request, plan, [-1]*4, [20, -1, -1, -1, -1])
        assert path.read_bytes() == saved and not path.with_suffix('.tmp').exists()
    assert 'torch_npu' not in sys.modules
    value = dict(status='HOST_new_risk_predictor_bounded_snapshot_passed',
                 no_mutation_of_policy_counts=True, original_gain_not_serialized=True,
                 only_every256_verdicts=True, NPU_used=False)
    target = Path(__file__).resolve().parent / 'HOST_SNAPSHOT346_RESULT.json'
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
