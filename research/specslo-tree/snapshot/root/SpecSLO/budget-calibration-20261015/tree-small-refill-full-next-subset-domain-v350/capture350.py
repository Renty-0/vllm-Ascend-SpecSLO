"""Observe the actual graph result or eager forward, preserving both paths."""


class Captures(list):
    def __init__(self):
        super().__init__()
        self.routes = []


def capture_begin(engine):
    model = engine.model
    runner = engine.graph_runner
    had_model = 'forward' in vars(model)
    previous_model = vars(model).get('forward')
    original_model = model.forward
    had_graph = 'run_tree_hidden' in vars(runner)
    previous_graph = vars(runner).get('run_tree_hidden')
    original_graph = runner.run_tree_hidden
    saved = Captures()
    depth = [0]

    def record(args, value, route):
        saved.append((args[0].clone(), args[1].clone(), args[2], value.clone()))
        saved.routes.append(route)

    def eager(*args, **kwargs):
        value = original_model(*args, **kwargs)
        if not depth[0]:
            record(args, value, 'original_eager_model')
        return value

    def graph(*args, **kwargs):
        depth[0] += 1
        try:
            value = original_graph(*args, **kwargs)
        finally:
            depth[0] -= 1
        if not depth[0]:
            record(args, value, 'original_exact_singleton_graph')
        return value

    model.forward = eager
    runner.run_tree_hidden = graph

    def restore():
        if had_model:
            model.forward = previous_model
        else:
            vars(model).pop('forward', None)
        if had_graph:
            runner.run_tree_hidden = previous_graph
        else:
            vars(runner).pop('run_tree_hidden', None)

    return saved, restore


def warm_singleton(engine, native, bank):
    if bank.n != 1:
        return None
    # Run the unchanged all-role helper before installing the observation tap.
    # Its original exact-prefill gate may perform eager/changed/restored calls.
    # No root is appended or published in this untimed qualification warmup.
    return engine._prefill_and_sample_target_batch(bank.prompts,
        [native.PearlPipelineState(list(p), len(p), max_tokens=256, ignore_eos=True)
         for p in bank.prompts], list(range(bank.n)))
