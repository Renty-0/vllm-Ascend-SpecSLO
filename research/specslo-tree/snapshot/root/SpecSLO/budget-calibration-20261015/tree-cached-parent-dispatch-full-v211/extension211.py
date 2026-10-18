"""Same209 body with one prevalidated mapper module outside the hot loop."""
import types
from check211 import ready,require,P,sha

_STATE=None


def cached_controls(mapper,parent):
    """Rebind the exact two209 function bodies; never mutate canonical modules."""
    state=dict(initial_parent_source_checks=1,cached_parent_returns=0,
               canonical_mapper_module=str(mapper.__file__),canonical_parent_module=str(parent.__file__))
    def cached(name):
        require(name=='mapper173','only the once-validated original mapper module')
        state['cached_parent_returns']+=1
        return parent
    namespace=dict(vars(mapper),parent_module=cached,__name__='ROOT211.cached_mapper209')
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
    return entry,Controls


def after_install(native,graph,live):
    import extension209
    result=extension209.after_install(native,graph,live)
    return dict(result,ROOT211_overlay='cached-parent-mapper-only',cached_source_precheck=True,
                new_model_or_graph_or_budget=False,source_epoch209_immutable=True)


def loaded_origins():
    import extension209
    frozen=ready()
    return dict(extension209.loaded_origins(),ROOT211_explicit_overlay_sources={
        name:digest for name,digest in frozen['files'].items()if Path(name).parent==P and Path(name).suffix=='.py'})


def qualify_component(engine,native,stage,policy,output):
    import extension209
    return extension209.qualify_component(engine,native,stage,policy,output)


def cache_snapshot():
    require(_STATE is not None and _STATE['cached_parent_returns']>0,'actual bound cached mapper has executed')
    return dict(_STATE,cache_lifetime='fresh worker; frozen pre/post guard remains',
                canonical_mapper_or_base_module_mutated=False)


from pathlib import Path
