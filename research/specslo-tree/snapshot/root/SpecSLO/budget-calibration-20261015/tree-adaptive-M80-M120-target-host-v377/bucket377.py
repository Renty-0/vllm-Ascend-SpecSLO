"""Preserve all original visible contexts in either complete physical family."""
from dataclasses import replace


def extent(lengths):
    if not lengths or any(type(n) is not int or not 0 < n <= 2560 for n in lengths):
        raise ValueError('ROOT377 complete visible KV lengths within2560')
    return ((max(lengths) + 511) // 512) * 512


def bind(engine):
    if engine.is_draft:
        return None
    state = engine._branch_tail_state
    state.uniform_KV_bucket377_enabled = True
    previous = state.prepare
    old_greedy = engine.graph_runner.run_target_greedy
    def prepare(*args, **kwargs):
        result = previous(*args, **kwargs)
        current = state.static_current
        if current is None:
            raise RuntimeError('ROOT377 original complete checked projection required')
        lengths = tuple(current['original'][2].sequence_lens)
        selected = extent(lengths) if state.uniform_KV_bucket377_enabled else 2560
        x, p, metadata, inverse, mapping = current['physical']
        m = int(x.numel())
        if (m not in (80,120) or tuple(metadata.sequence_lens) != (2560,) * (m // 5)
                or tuple(metadata.actual_seq_lengths_q) != tuple(range(5,m+1,5))):
            raise RuntimeError('ROOT377 actual whole80/120 Q5 family before extent selection')
        current['physical'] = (x,p,replace(metadata,sequence_lens=(selected,)*(m//5)),inverse,mapping)
        current['uniform_KV_bucket377'] = dict(enabled=state.uniform_KV_bucket377_enabled,
            extent=selected, original_sequence_lens=list(lengths))
        return result
    def greedy(*args, **kwargs):
        before = len(state.static_graph_calls)
        result = old_greedy(*args, **kwargs)
        for call in state.static_graph_calls[before:]:
            call['uniform_KV_bucket377'] = dict(state.static_current['uniform_KV_bucket377'])
            call['adaptive377_enabled'] = state.adaptive377_enabled
        return result
    state.prepare = prepare
    engine.graph_runner.run_target_greedy = greedy
    return dict(physical_families=[80,120], visible_KV_never_truncated=True,
        Native_qualification_flags_assigned=False, model_qualified=False)
