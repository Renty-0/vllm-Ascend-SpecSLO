"""Read-only config report: Native MC2 public document, then JSON values.

This never deepcopies the immutable profile and never feeds data back into a
model/config. MC2's own strict public schema retains tuple-key measurements.
"""
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
import json
import math

def _value(value,path):
    if value is None or type(value) in (str,bool,int):return value
    if type(value) is float:
        if not math.isfinite(value):raise TypeError(path+': nonfinite config number')
        return value
    # The production class has a purpose-built, read-only schema1 serializer.
    # Generic dataclass recursion would expose tuple-key MappingProxy entries.
    if is_dataclass(value) and not isinstance(value,type) and type(value).__name__=='MC2Profile':
        expected=('metadata','entries','latency_percentile','minimum_samples','minimum_speedup')
        if tuple(field.name for field in fields(value))!=expected or not callable(getattr(value,'_document',None)):
            raise TypeError(path+': unsupported MC2Profile schema')
        document=value._document()
        if not isinstance(document,Mapping) or document.get('schema_version')!=1 or 'measurements' not in document:
            raise TypeError(path+': original MC2Profile public document is invalid')
        return _value(document,path+'._document()')
    if is_dataclass(value) and not isinstance(value,type):
        return {field.name:_value(getattr(value,field.name),path+'.'+field.name) for field in fields(value)}
    if isinstance(value,Mapping):
        out={}
        for key,item in value.items():
            if type(key) is not str:raise TypeError(path+': unsupported non-string config mapping key '+repr(key))
            out[key]=_value(item,path+'['+repr(key)+']')
        return out
    if isinstance(value,(tuple,list)):
        return [_value(item,path+'['+str(index)+']') for index,item in enumerate(value)]
    raise TypeError(path+': unsupported config value '+type(value).__module__+'.'+type(value).__qualname__)

def snapshot_config(config):
    if not is_dataclass(config) or isinstance(config,type):raise TypeError('$config: expected actual dataclass instance')
    result={field.name:_value(getattr(config,field.name),'$config.'+field.name) for field in fields(config)}
    if json.loads(json.dumps(result,allow_nan=False))!=result:raise TypeError('$config: JSON snapshot is not lossless')
    return result
