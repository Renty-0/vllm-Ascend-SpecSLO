"""Capture original commit identity before installation, as worker199 does.

No model operation, Native flag, qualification result or numerical permission
is created here. The original419 protocol still checks and executes its own
original commit function.
"""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT430 original commit binding: ' + message)


@dataclass(frozen=True)
class Captured:
    protocol: object
    native_class: object
    commit_states: object
    source_sha256: str


def capture(protocol, native):
    from original_tree_cycle_v75 import _require_compiled
    import original_native_protocol_v75 as original
    from base199 import BASE
    from poisson_coverage_probe import _expected_code
    from types import FunctionType
    _require_compiled(protocol)
    path = BASE / 'original_native_protocol_v75.py'
    need(hashlib.sha256(path.read_bytes()).hexdigest() ==
         '74d1a745eb8841afefc2b5763a5802e1229b63e2285878daaa36c91e0a66ab49',
         'exact original protocol file already pinned by P148 parent source')
    function = original.commit_states
    need(isinstance(function, FunctionType)
         and Path(original.__file__).resolve() == path.resolve()
         and Path(function.__globals__['__file__']).resolve() == path.resolve()
         and function.__code__ == _expected_code(path, 'commit_states'),
         'actual original source and complete code identity before installer')
    return Captured(protocol, native.NativePearlEngine, function,
                    hashlib.sha256(inspect.getsource(function).encode()).hexdigest())


def bind(engine, captured):
    import original_native_protocol_v75 as original
    need(type(captured) is Captured and isinstance(engine, captured.native_class)
         and engine._v75_original_protocol is captured.protocol
         and original.commit_states is captured.commit_states
         and not hasattr(engine, '_v199_commit_states')
         and not hasattr(engine, '_root430_commit_binding'),
         'same actual engine, pre-startup protocol and unchanged original function')
    engine._v199_commit_states = captured.commit_states
    engine._root430_commit_binding = captured
    return snapshot(engine)


def snapshot(engine):
    import original_native_protocol_v75 as original
    from original_tree_cycle_v75 import _require_compiled
    value = engine._root430_commit_binding
    _require_compiled(value.protocol)
    need(type(value) is Captured and isinstance(engine, value.native_class)
         and engine._v75_original_protocol is value.protocol
         and engine._v199_commit_states is value.commit_states is original.commit_states,
         'same pre-startup compiled object and original commit identity until audit')
    return dict(original_commit_source_sha256=value.source_sha256,
        same_prestartup_protocol_and_original_commit_identity=True,
        exactly_original_runtime199_capture_and_engine_binding=True,
        Native_flags_assigned=False, NN_permission_granted=False)
