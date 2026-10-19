"""Exercise projected and deliberately unprojected actual observer paths."""
import json
from pathlib import Path
from types import SimpleNamespace

from bucket344 import bind


def main():
    state = SimpleNamespace(static_current=None, static_graph_calls=[], prepare=lambda: None)
    mode = ['bypass']

    def greedy(*args, **kwargs):
        if mode[0] != 'bypass':
            state.static_graph_calls.append(dict(original_call=True))
        return 'original-result'

    engine = SimpleNamespace(is_draft=False, _branch_tail_state=state,
                             graph_runner=SimpleNamespace(run_target_greedy=greedy))
    bind(engine)
    observed = engine.graph_runner.run_target_greedy
    assert observed() == 'original-result' and state.static_graph_calls == []
    mode[0] = 'projected'
    state.static_current = dict(uniform_KV_bucket344=dict(extent=512, original_max_context=132, enabled=True))
    assert observed() == 'original-result'
    assert state.static_graph_calls[-1]['uniform_KV_bucket344'] == state.static_current['uniform_KV_bucket344']
    for value in (None, {}):
        state.static_current = value
        try:
            observed()
        except RuntimeError as error:
            assert 'missing uniform execution extent' in str(error)
        else:
            raise AssertionError('missing actual projection witness accepted')
    result = dict(status='passed_observer_actual_and_bypass_HOST', NPU_used=False,
                  original_layout_negative_control_does_not_require_static_projection=True,
                  original_greedy_result_retained=True, invalid_projected_calls_rejected=2,
                  no_model_qualification=True)
    (Path(__file__).resolve().parent/'HOST344_OBSERVER_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
