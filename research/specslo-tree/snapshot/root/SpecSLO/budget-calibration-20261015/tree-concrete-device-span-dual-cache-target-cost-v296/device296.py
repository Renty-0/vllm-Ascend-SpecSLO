"""Resolve an unspecified NPU device without relaxing actual cache checks."""


def resolve_cache_device(requested, actual, current_index):
    if requested.type != 'npu' or actual.type != 'npu':
        raise ValueError('ROOT296 actual cache and requested device must be NPU')
    if actual.index is None or actual.index != current_index:
        raise ValueError('ROOT296 cache must reside on the current concrete NPU')
    if requested.index is not None and requested.index != actual.index:
        raise ValueError('ROOT296 explicitly requested NPU must match cache')
    return actual
