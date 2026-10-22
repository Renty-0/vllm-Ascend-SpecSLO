"""Retain all425/41975 cases, change only the independent metadata selector."""
from types import FunctionType
import actor419
import bounded425 as original
import target428


def actor():
    namespace=dict(vars(actor419),target=target428)
    fn=actor419.run_case
    case=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    case.__kwdefaults__=fn.__kwdefaults__
    namespace['run_case']=case
    template=original.actor()
    # The template already retains425's read-only diagnostic binding. Its
    # original run body resolves the exact419 case through this own namespace.
    namespace.update({k:v for k,v in template.__globals__.items()if k not in ('target','run_case')})
    return FunctionType(template.__code__,namespace,template.__name__,template.__defaults__,template.__closure__)


def run(engine,native,stage,output):
    namespace=dict(vars(original),actor=actor)
    fn=original.run
    actual=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    value=actual(engine,native,stage,output)
    from metadata428 import descriptor
    engine._root428_reference_metadata_binding=dict(source=descriptor(),
        actual_target_metadata_matches=engine._parent4_target391_binding['actual_metadata_matches'],
        current_cases=75,original425_backend_unchanged=True,Native_flags_assigned=False)
    return value
