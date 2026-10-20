"""Separate P159-only64 control from optional active32/24/16 combinations."""
from base169 import require
DEFAULT_WINDOWS=(32,)
def window(value):
 require(type(value)is int and value==32,'explicit uncapped64 or active32/24/16')
 return value
def windows(values):
 values=tuple(window(x) for x in values)
 require(bool(values) and len(values)==len(set(values)),'nonempty unique window order')
 return values
def parse_windows(text):
 require(type(text)is str and bool(text),'window list')
 return windows(tuple(int(v) for v in text.split(',')))
def adapt_source(source,active_window):
 value=window(active_window)
 if value==64:return source
 from window160 import adapt_source as capped
 return capped(source,value)
