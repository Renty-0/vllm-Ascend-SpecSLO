"""Reproduce cached graph bypass and verify exact recorder restoration."""
import json
from pathlib import Path
from types import SimpleNamespace

from capture349 import capture_begin


class Tensor:
    def __init__(self, value): self.value = value
    def clone(self): return Tensor(self.value)


class Model:
    def forward(self, x, p, metadata): return Tensor(x.value+1)


class Runner:
    def __init__(self, model): self.model = model; self.cold = False; self.fail = False
    def run_tree_hidden(self, x, p, metadata):
        if self.fail: raise RuntimeError('controlled graph error')
        return self.model.forward(x,p,metadata) if self.cold else Tensor(777)


def main():
    model = Model(); runner = Runner(model)
    engine = SimpleNamespace(model=model, graph_runner=runner)
    args = (Tensor(10), Tensor(20), object())
    # The old model-only observer sees no forward on a cached graph replay.
    old_calls = []; original = model.forward
    model.forward = lambda *args: (old_calls.append(1), original(*args))[1]
    runner.run_tree_hidden(*args)
    assert old_calls == []
    del model.forward
    results = []
    for cold in (False,True):
        runner.cold = cold
        saved, restore = capture_begin(engine)
        try:
            value = runner.run_tree_hidden(*args)
            assert len(saved) == 1 and saved.routes == ['original_exact_singleton_graph']
            assert saved[0][3].value == value.value
            model.forward(*args)
            assert len(saved) == 2 and saved.routes[-1] == 'original_eager_model'
            runner.fail = True
            try: runner.run_tree_hidden(*args)
            except RuntimeError: pass
            else: raise AssertionError('controlled graph error absent')
            model.forward(*args)
            assert saved.routes[-1] == 'original_eager_model'
        finally:
            runner.fail = False
            restore()
        assert 'forward' not in vars(model) and 'run_tree_hidden' not in vars(runner)
        results.append(dict(cold=cold, no_double_count=True, graph_result_retained=True))
    value = dict(status='passed_singleton_graph_capture_regression_HOST',NPU_used=False,
        old_cached_graph_missing_forward_reproduced=True, cases=results,
        model_and_runner_instance_methods_restored=True, exception_restores_observation_depth=True,
        actual_model_execution=False, model_qualified=False)
    (Path(__file__).resolve().parent/'HOST349_CAPTURE_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
