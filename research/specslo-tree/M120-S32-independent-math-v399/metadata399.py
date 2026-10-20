"""Independent stepwise S24 reference, preserving M120 and the current KV bucket.

This does not invoke projection397 or reuse candidate inputs/hidden/KV. Each
reference owner's first q5 query is consumed; its future canaries stay masked.
"""
from dataclasses import replace


def pack_q5(engine, frame, *, full):
    from metadata255 import pack_q5 as original
    result = original(engine, frame, full=full)
    inputs, positions, metadata = result
    selected = engine._frugal399_reference_extent
    if selected not in (512, 1024, 1536, 2048, 2560):
        raise RuntimeError('ROOT399 explicit current visible-KV bucket')
    # The original reference consumes only each first query. Future canaries
    # may have larger positions, but their masks expose only the finite key0.
    for row in frame['real_queries']:
        if not row['position']+1 <= selected:
            raise RuntimeError('ROOT399 cannot truncate any own reference prefix')
    return inputs, positions, replace(metadata, sequence_lens=(selected,)*24)


def checked(metadata):
    if (tuple(metadata.tree_attention_mask.shape) != (24, 1, 10, 4096)
            or len(set(metadata.sequence_lens)) != 1
            or metadata.sequence_lens[0] not in (512, 1024, 1536, 2048, 2560)
            or tuple(metadata.actual_seq_lengths_q) != tuple(range(5, 121, 5))):
        raise RuntimeError('ROOT399 unchanged independent S24 q5 reference buffers')
    return metadata
