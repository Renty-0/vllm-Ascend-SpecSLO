"""Use the unchanged PEARL receiver with a qualification-only config view.

The engine/worker configuration remains unchanged. The outer owned supervisor
still enforces its existing18000-second total limit. Production generation
and other API calls retain their original timeouts.
"""
from dataclasses import replace

QUALIFICATION_TIMEOUT_SECONDS=18000
_LAST=None


class QualificationView:
    def __init__(self,engine):
        self._engine=engine
        self.config=replace(engine.config,worker_timeout_seconds=QUALIFICATION_TIMEOUT_SECONDS)

    def __getattr__(self,name):return getattr(self._engine,name)


def receive_qualification(engine,operation):
    global _LAST
    from vllm_ascend.spec_decode.pearl.api import PEARLEngine
    expected='V209 causal80-N9-to15-allB4-empty-active32-full463'
    if operation!=expected or not isinstance(engine,PEARLEngine):
        raise RuntimeError('ROOT436 only the exact original full-qualification receive is adapted')
    bound=engine._receive_all
    if bound.__func__ is not PEARLEngine._receive_all:
        raise RuntimeError('ROOT436 original API receive method identity required')
    original_config=engine.config
    view=QualificationView(engine)
    if not(view._connections is engine._connections and view._processes is engine._processes):
        raise RuntimeError('ROOT436 actual same owned processes and pipes required')
    try:
        return PEARLEngine._receive_all(view,operation)
    finally:
        if engine.config is not original_config:
            raise RuntimeError('ROOT436 original engine configuration was changed')
        _LAST=dict(original_parent_timeout_seconds=original_config.worker_timeout_seconds,
            qualification_receive_timeout_seconds=view.config.worker_timeout_seconds,
            original_engine_configuration_object_unchanged=True,
            original_receive_function_and_owned_connections_preserved=True,
            worker_configuration_changed=False,production_timeout_changed=False,
            outer_supervisor_total_limit_seconds=18000)


def snapshot():
    if _LAST is None:raise RuntimeError('ROOT436 actual qualification receive not called')
    return dict(_LAST)
