"""Pre-install FastV5 dispatcher; original RootSeed remains the outer callback.

Install the original runner/cache hooks exactly once. Construct the419 E4
callback from its complete retained body without calling its installer again.
The E4 path is gated by live425; no historical receipt issues permission.
"""
import ast
import hashlib
import importlib.util
import inspect
import linecache
from pathlib import Path
import sys
import types

P=Path(__file__).resolve().parent
B=P.parent
BASE=B/'tree-high-normal-cap4-native-families-v148'
E4=B/'tree-parent4-normal24-SLO-mixed-protocol-v419/fast419.generated.py'
_FACTORY=None


def _load(name,path):
    if name in sys.modules:raise RuntimeError('ROOT425 fresh private source module required '+name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module


def body_sources():
    text=E4.read_text()
    install=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='install')
    body=next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name=='draft')
    before=ast.get_source_segment(text,body)
    old='from live419 import fused_prelude,validate'
    new='from live425 import fused_prelude,validate'
    if before.count(old)!=1:raise RuntimeError('ROOT425 exact private E4 gate import')
    after=before.replace(old,new)
    if after.replace(new,old)!=before:raise RuntimeError('ROOT425 entire E4 neural callback source inverse')
    return before,after


def e4_body(native,module,original_native):
    import torch
    source=body_sources()[1]
    filename=str(P/'factory425.py')+'::retained419-E4-body-new-explicit-gate'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    namespace=dict(vars(module),native=native,runner_cls=native.NativeACLGraphRunner,
                   original=original_native,torch=torch)
    exec(compile(source,filename,'exec',dont_inherit=True),namespace)
    return namespace['draft']


def prepare_helpers():
    global _FACTORY
    if _FACTORY is not None or 'fast_tree_draft_v5' in sys.modules:
        raise RuntimeError('ROOT425 prepare once before original RootSeed imports canonical FastV5')
    base=_load('_ROOT425_original_fast',BASE/'fast_tree_draft_v5.py')
    future=_load('_ROOT425_retained_E4_helpers',E4)
    for name in ('execute_top2','reserve_query_pages','missing_committed_suffix'):
        a=ast.parse(inspect.getsource(getattr(base,name)))
        b=ast.parse(inspect.getsource(getattr(future,name)))
        if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False):
            raise RuntimeError('ROOT425 shared unchanged original helper '+name)
    # Fusion helper loading is explicit and identical to the reviewed419 path.
    from base199 import V159  # installs the exact retained151/159 helper paths
    from source159 import module,FUSION
    fusion=module('repair_frontier_fusion',FUSION)
    canonical=types.ModuleType('fast_tree_draft_v5')
    # The original observer replaces canonical.execute_top2 after install.
    # Compile the exact original source into that same module namespace so
    # its retained install/runner callbacks observe the replacement. Copying
    # vars(base) would leave those functions bound to a different dictionary.
    original_path=BASE/'fast_tree_draft_v5.py'
    exec(compile(original_path.read_text(),str(original_path),'exec',dont_inherit=True),
         vars(canonical))
    original_install=canonical.install
    canonical.__file__=str(P/'factory425.py')
    canonical.__package__=''

    def install(native):
        cls=native.NativePearlEngine
        if getattr(cls,'_ROOT425_dispatcher',None) is not None:
            raise RuntimeError('ROOT425 one original FastV5 runner/cache installation required')
        original_native=cls.draft_tree_forward
        original_install(native)
        fallback=cls.draft_tree_forward
        extra=e4_body(native,future,original_native)

        def draft(engine,plans,roots,ids,eager_parent_sources=None,**kwargs):
            if not engine.is_draft:return None
            sources=eager_parent_sources or {}
            if sources and any(p.candidate_budget==4 for p,_ in sources.values()):
                if any(p.candidate_budget!=4 for p,_ in sources.values()):
                    raise RuntimeError('ROOT425 no mixed oldB3/newB4 parent mode')
                from live425 import precheck
                precheck(engine,plans,ids,sources)
                return extra(engine,plans,roots,ids,eager_parent_sources=eager_parent_sources,**kwargs)
            return fallback(engine,plans,roots,ids,eager_parent_sources=eager_parent_sources,**kwargs)

        cls.draft_tree_forward=draft
        cls._ROOT425_dispatcher=dict(dispatch=draft,fallback=fallback,e4=extra,
            original_native=original_native,base_module=base,canonical=canonical,
            original_install=original_install,private_helpers=future,
            runner_execute=native.NativeACLGraphRunner._execute_draft,
            cache_allocate=cls._allocate_cache)

    canonical.install=install
    sys.modules['fast_tree_draft_v5']=canonical
    _FACTORY=dict(canonical=canonical,base=base,e4_helpers=future,fusion=fusion)
    return dict(canonical_path=canonical.__file__,original_runner_hook_installs=1,
        E4_neural_body_only_gate_import_changed=True,mode='unissued',Native_flags_assigned=False)


def current(native):
    value=getattr(native.NativePearlEngine,'_ROOT425_dispatcher',None)
    if value is None:raise RuntimeError('ROOT425 actual pre-model dispatcher installation required')
    outer=native.NativePearlEngine.draft_tree_forward
    if (outer.__code__.co_filename!=str(BASE/'root_seed.py')
            or inspect.getclosurevars(outer).nonlocals.get('original') is not value['dispatch']
            or value['fallback'].__code__.co_filename!=str(BASE/'fast_tree_draft_v5.py')
            or value['fallback'].__globals__ is not vars(value['canonical'])
            or value['original_install'].__globals__ is not vars(value['canonical'])
            or native.NativeACLGraphRunner._execute_draft is not value['runner_execute']
            or native.NativePearlEngine._allocate_cache is not value['cache_allocate']):
        raise RuntimeError('ROOT425 exact RootSeed, dispatcher, original fallback and runner/cache bindings')
    return value


def descriptor():
    before,after=body_sources()
    return dict(base_source_sha256=hashlib.sha256((BASE/'fast_tree_draft_v5.py').read_bytes()).hexdigest(),
        retained419_source_sha256=hashlib.sha256(E4.read_bytes()).hexdigest(),
        original_E4_body_sha256=hashlib.sha256(before.encode()).hexdigest(),
        new_gated_E4_body_sha256=hashlib.sha256(after.encode()).hexdigest(),
        entire_E4_body_inverse=True,original_top2_and_KV_helpers_unmodified=True,
        no_after_install_closure_mutation=True,
        original_install_and_runner_use_canonical_observer_globals=True,
        Native_flags_assigned=False)
