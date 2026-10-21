"""Explicit real-owner dispatch between frozen M120 projectors; installs nothing."""
from projection255 import transform as narrow
from projection397 import transform as wide


def transform(inputs, positions, metadata, real_queries, zero_page, *, extent=2560,
              mask_capacity=10, real_N, copy_enabled):
    if type(real_N) is not int or not 1 <= real_N <= 32 or type(copy_enabled) is not bool:
        raise ValueError('HOST408 explicit real owner count and branch-copy role required')
    if real_N <= 24:
        return narrow(inputs, positions, metadata, real_queries, zero_page,
                      extent=extent, mask_capacity=mask_capacity)
    if copy_enabled or mask_capacity != 10:
        raise ValueError('HOST408 wide linear path cannot execute branch-copy families')
    result = wide(inputs, positions, metadata, real_queries, zero_page,
                  extent=extent, mask_capacity=mask_capacity)
    if result[-1]['real_N'] != real_N:
        raise ValueError('HOST408 original real owner count and wide mapping differ')
    return result
