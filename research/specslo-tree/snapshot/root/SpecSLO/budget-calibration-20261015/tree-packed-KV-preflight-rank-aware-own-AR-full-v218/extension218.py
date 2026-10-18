"""Same209 body with one prevalidated mapper module outside the hot loop."""
import types
from check218 import ready,require,P,sha

_STATE=None


def cached_controls(mapper,parent):
    """Rebind the exact two209 function bodies; never mutate canonical modules."""
    state=dict(initial_parent_source_checks=1,cached_parent_returns=0,
               canonical_mapper_module=str(mapper.__file__),canonical_parent_module=str(parent.__file__))
    def cached(name):
        require(name=='mapper173','only the once-validated original mapper module')
        state['cached_parent_returns']+=1
        return parent
    namespace=dict(vars(mapper),parent_module=cached,__name__='ROOT218.cached_mapper209')
    dispatch=types.FunctionType(mapper.dispatch.__code__,namespace,'dispatch',mapper.dispatch.__defaults__,mapper.dispatch.__closure__)
    namespace['dispatch']=dispatch
    factory=types.FunctionType(mapper.controls_class.__code__,namespace,'controls_class',mapper.controls_class.__defaults__,mapper.controls_class.__closure__)
    controls=factory()
    return controls,state,dispatch


def prepare_install(output):
    global _STATE
    ready()
    import extension209,mapper209
    from base209 import parent_module
    require(_STATE is None,'one fresh private cache per worker')
    # Keep original prepare side effects. Its original ControlsClass is unused.
    entry,_=extension209.prepare_install(output)
    parent=parent_module('mapper173') # Actual source identity/SHA check, once before model work.
    Controls,_STATE,_=cached_controls(mapper209,parent)
    from cost218 import budget_controls
    return entry,budget_controls(Controls,_STATE)


def after_install(native,graph,live):
    import extension209
    import os
    require(os.environ.get("VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION")=="1", "explicit original SLO admission1")
    result=extension209.after_install(native,graph,live)
    from slot_preflight218 import install
    install(native)
    return dict(result,ROOT218_overlay='cached-parent-SLO-admission1-cost-aware-N9to16',SLO_AWARE_ADMISSION=True,cached_source_precheck=True,cost_aware_N9to16_budget=True,
                new_model_or_graph=False,budget_grants_may_change=True,packed_target_KV_preflight=True,source_epoch209_immutable=True)


def loaded_origins():
    import extension209
    frozen=ready()
    return dict(extension209.loaded_origins(),ROOT218_explicit_overlay_sources={
        name:digest for name,digest in frozen['files'].items()if Path(name).parent==P and Path(name).suffix=='.py'})


def qualify_component(engine,native,stage,policy,output):
    import extension209
    value=extension209.qualify_component(engine,native,stage,policy,output)
    from slot_preflight218 import end_shadow
    stage('ROOT218/complete-actual-preflight-shadow',lambda:end_shadow(engine))
    return value


def cache_snapshot():
    require(_STATE is not None and _STATE['cached_parent_returns']>0,'actual bound cached mapper has executed')
    return dict(_STATE,cache_lifetime='fresh worker; frozen pre/post guard remains',
                canonical_mapper_or_base_module_mutated=False)


from pathlib import Path


def postfull_AR(engine,native,stage,windows,output):
    from reference218 import run
    return run(engine,native,stage,windows,output)
