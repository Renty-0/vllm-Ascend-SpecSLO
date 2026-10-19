"""Explicit48 active cap; immutable original cap/source-adaptation bodies."""
import types
import window160 as original
import probe160
DEFAULT_WINDOWS=(48,)


def window(value):
    if type(value)is not int or value!=48:raise ValueError('this candidate explicitly measures active48')
    return value


def windows(values):
    values=tuple(window(v)for v in values)
    if values!=(48,):raise ValueError('one declared active48 candidate')
    return values


def parse_windows(text):return windows(tuple(int(v)for v in text.split(',')))


adapt_source=types.FunctionType(original.adapt_source.__code__,dict(vars(original),window=window),
    original.adapt_source.__name__,original.adapt_source.__defaults__,original.adapt_source.__closure__)
trace_source=types.FunctionType(probe160.trace_source.__code__,dict(vars(probe160),window=window),
    probe160.trace_source.__name__,probe160.trace_source.__defaults__,probe160.trace_source.__closure__)
