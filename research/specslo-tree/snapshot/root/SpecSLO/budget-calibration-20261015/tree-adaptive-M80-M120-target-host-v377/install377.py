"""One original copy state; privately bind dimensions, without double copying."""
import types
import branch_tail255
import static_state255
from source377 import source_module, function

_small_layout, _layout_source = source_module(branch_tail255, 'branch_tail255.py', {'layout'}, {120: 80})


def layout(plans, *args, **kwargs):
    m = sum(int(p.candidate_budget) + 1 for p in plans)
    if m not in (80, 120):
        raise ValueError('ROOT377 original full padded query accounting')
    return (_small_layout['layout'] if m == 80 else branch_tail255.layout)(plans, *args, **kwargs)


def install(engine, native):
    adapted = function(static_state255.install, (
        ('M=120,\n            attention_segments=24,',
         'M=int(x.numel()),\n            attention_segments=len(meta.actual_seq_lengths_q),'),
        ('fixed_M=120,fixed_q5_groups=24',
         'physical_M_families=(80,120),physical_q5_families=(16,24)'),
    ))
    activation = adapted(engine, native)
    state = engine._branch_tail_state
    state.adaptive377_enabled = False
    prepared = state.prepare
    cells = dict(zip(prepared.__code__.co_freevars, prepared.__closure__, strict=True))
    old_prepare = cells['prepare'].cell_contents
    if old_prepare.__func__ is not branch_tail255.State.prepare:
        raise RuntimeError('ROOT377 actual one original branch-copy state required')
    private = types.FunctionType(old_prepare.__func__.__code__,
        dict(old_prepare.__func__.__globals__, layout=layout), old_prepare.__func__.__name__,
        old_prepare.__func__.__defaults__, old_prepare.__func__.__closure__)
    bound = types.MethodType(private, state)
    from physical377 import transform
    def cell(value):
        return (lambda: value).__closure__[0]
    replacement = tuple(cell(bound) if name == 'prepare' else cell(transform) if name == 'transform' else cells[name]
                        for name in prepared.__code__.co_freevars)
    state.prepare = types.FunctionType(prepared.__code__, prepared.__globals__, prepared.__name__,
        prepared.__defaults__, replacement)
    from geometry377 import pad_function
    pad = pad_function()
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = pad
    state.adaptive377_bindings = dict(prepare=state.prepare, branch_prepare=bound,
        transform=transform, pad=pad, greedy=engine.graph_runner.run_target_greedy)
    return dict(original=activation, physical_M_families=[80,120], physical_Q=5,
        physical_S_families=[16,24], exactly_one_original_branch_copy_state=True,
        original_draft_and_allocator_unchanged=True, Native_flags_assigned=False,
        NPU_qualified=False, Raw=None, Goodput=None)
