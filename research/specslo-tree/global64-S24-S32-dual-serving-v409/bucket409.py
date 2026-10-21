"""Retain full M120 projections; check uniform visible extent for both routes."""
from dataclasses import replace


def extent(lengths):
    if not lengths or any(type(v) is not int or not 0 < v <= 2560 for v in lengths):
        raise ValueError('ROOT409 complete positive visible KV lengths at most2560')
    return ((max(lengths) + 511) // 512) * 512


def valid_record(record):
    values = record['fixed_KV_extent']
    detail = record['uniform_KV_bucket409']
    count = record['attention_segments']
    selected = extent(tuple(detail['original_sequence_lens']))
    return (count in (24, 32) and detail['enabled'] is True
            and detail['extent'] == selected and values == [selected] * count)


def bind(engine):
    if engine.is_draft:
        return None
    state = engine._branch_tail_state
    if hasattr(state, 'uniform_KV_bucket409_enabled'):
        raise RuntimeError('ROOT409 bind actual target state only once')
    state.uniform_KV_bucket409_enabled = False
    previous = state.prepare
    old_greedy = engine.graph_runner.run_target_greedy
    def prepared(*args, **kwargs):
        result = previous(*args, **kwargs)
        current = state.static_current
        if current is None:
            raise RuntimeError('ROOT409 checked physical projection missing')
        original_lengths = tuple(current['original'][2].sequence_lens)
        selected = extent(original_lengths) if state.uniform_KV_bucket409_enabled else 2560
        x, position, metadata, inverse, mapping = current['physical']
        n = current['real_N']
        segments = 32 if n > 24 else 24
        from geometry409 import query_ends_wide
        wanted = tuple(query_ends_wide()) if n > 24 else tuple(range(5, 121, 5))
        if (x.numel() != 120 or tuple(metadata.sequence_lens) != (2560,) * segments
                or tuple(metadata.actual_seq_lengths_q) != wanted
                or (n > 24 and current['copy_enabled'])):
            raise RuntimeError('ROOT409 explicit real owner scope and actual dual metadata differ')
        current['physical'] = (x, position, replace(metadata, sequence_lens=(selected,) * segments), inverse, mapping)
        current['uniform_KV_bucket409'] = dict(enabled=state.uniform_KV_bucket409_enabled,
            extent=selected, original_sequence_lens=list(original_lengths))
        return result
    def greedy(*args, **kwargs):
        before = len(state.static_graph_calls)
        result = old_greedy(*args, **kwargs)
        added = state.static_graph_calls[before:]
        if added:
            current = state.static_current
            if current is None or 'uniform_KV_bucket409' not in current:
                raise RuntimeError('ROOT409 actual graph dispatch lacks checked extent')
            for call in added:
                call['uniform_KV_bucket409'] = dict(current['uniform_KV_bucket409'])
        return result
    state.prepare = prepared
    engine.graph_runner.run_target_greedy = greedy
    return dict(target_only=True, M=120, narrow_S=24, wide_S=32,
                maximum_extent=2560, actual_uniform_extent_checked=True,
                model_qualification_inherited=False)


def install(engine, native):
    from static409 import install as original
    return dict(original=original(engine, native), bucket=bind(engine))
