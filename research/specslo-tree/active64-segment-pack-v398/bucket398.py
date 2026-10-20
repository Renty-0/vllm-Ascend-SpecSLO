"""Change only best257's uniform target execution extent, before sealing."""
from dataclasses import replace


def extent(lengths):
    if not lengths or any(type(v) is not int or not 0 < v <= 2560 for v in lengths):
        raise ValueError('ROOT398 complete visible real and padding KV lengths')
    return ((max(lengths) + 511) // 512) * 512


def valid_record(record):
    values = record['fixed_KV_extent']
    detail = record['uniform_KV_bucket398']
    selected = extent(tuple(detail['original_sequence_lens']))
    return (detail['enabled'] is True and detail['extent'] == selected
            and values == [selected] * 24)


def bind(engine):
    if engine.is_draft:
        return None
    state = engine._branch_tail_state
    if hasattr(state, 'uniform_KV_bucket398_enabled'):
        raise RuntimeError('ROOT398 bind same target state exactly once')
    state.uniform_KV_bucket398_enabled = False
    previous = state.prepare
    old_greedy = engine.graph_runner.run_target_greedy

    def prepared(*args, **kwargs):
        result = previous(*args, **kwargs)
        current = state.static_current
        if current is None:
            raise RuntimeError('ROOT398 original checked physical projection required')
        original_lengths = tuple(current['original'][2].sequence_lens)
        selected = extent(original_lengths) if state.uniform_KV_bucket398_enabled else 2560
        x, p, metadata, inverse, mapping = current['physical']
        if (x.numel() != 120 or tuple(metadata.sequence_lens) != (2560,) * 24
                or tuple(metadata.actual_seq_lengths_q) != tuple(range(5, 121, 5))):
            raise RuntimeError('ROOT398 unchanged original M120/Q5/S24 preparation')
        current['physical'] = (x, p, replace(metadata, sequence_lens=(selected,) * 24), inverse, mapping)
        current['uniform_KV_bucket398'] = dict(enabled=state.uniform_KV_bucket398_enabled,
            extent=selected, original_sequence_lens=list(original_lengths))
        return result

    def greedy(*args, **kwargs):
        before = len(state.static_graph_calls)
        output = old_greedy(*args, **kwargs)
        added = state.static_graph_calls[before:]
        if added:
            current = state.static_current
            if current is None or 'uniform_KV_bucket398' not in current:
                raise RuntimeError('ROOT398 actual dispatch lacks checked extent')
            for call in added:
                call['uniform_KV_bucket398'] = dict(current['uniform_KV_bucket398'])
        return output

    state.prepare = prepared
    engine.graph_runner.run_target_greedy = greedy
    return dict(target_only=True, original_best257_copy_attention_and_draft_unchanged=True,
        M=120, Q=5, S=24, maximum_extent=2560, model_qualified=False)


def install(engine, native):
    from static_state257 import install as original
    return dict(original=original(engine, native), bucket=bind(engine))
