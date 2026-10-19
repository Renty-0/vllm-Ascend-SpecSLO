"""Keep Native frontend checks; choose FULL bucket or original causal frames."""
from dataclasses import replace
from types import MethodType

from metadata336 import canonical, ensure_zero
from metadata339 import bind, HOST_KEY, functions


def install(engine):
    if engine.is_draft:
        return dict(target_only=False)
    root = engine.target_tree_forward.__func__
    targets = [f for f in functions(root) if HOST_KEY in f.__code__.co_names
               and callable(f.__globals__.get(HOST_KEY))]
    if len(targets)!=1:
        raise RuntimeError('one complete installed Native HOST metadata frontend required')
    counted = targets[0].__globals__[HOST_KEY]
    model=engine.model
    if 120 not in model._mc2_chained_row_counts or 120 in model._mc2_deferred_row_counts:
        raise RuntimeError('original qualified flushed MC2 for actual M120 required')
    engine._shape341=None
    engine._calls341=[]
    engine._inputs341=None

    def builder(this, plans, roots, candidates, *args, **kwargs):
        if this is not engine or engine._shape341 is None:
            raise RuntimeError('same engine and explicit diagnostic frame')
        family, extent = engine._shape341
        kwargs['allow_causal_fast_path'] = family=='causal-inner1'
        x,p,m=counted(this,plans,roots,candidates,*args,**kwargs)
        if len(plans)!=24 or x.numel()!=120 or tuple(m.actual_seq_lengths_q)!=tuple(range(5,121,5)):
            raise RuntimeError('all24 uniform Q5 groups and exact M120 required')
        if family!='causal-inner1':
            m=canonical(engine,m,width=5,groups=24)
            m=replace(m,sequence_lens=(extent,)*24)
        elif m.tree_attention:
            raise RuntimeError('actual Native causal metadata required')
        return x,p,m

    private,_,graph=bind(root,builder)
    engine.target_tree_forward=MethodType(private,engine)
    ordinary=engine.graph_runner.run_target_greedy

    def greedy(xs,ps,ms,vocabulary_size):
        family,extent=engine._shape341
        if len(xs)!=len(ps) or len(xs)!=len(ms) or len(xs)!=1 or xs[0].numel()!=120:
            raise RuntimeError('one actual original target model step')
        m=ms[0]
        if bool(m.tree_attention)!=(family!='causal-inner1'):
            raise RuntimeError('actual target family differs from declared experiment')
        if family!='causal-inner1' and (tuple(m.tree_attention_mask.shape)!=(24,1,10,4096)
                                      or tuple(m.sequence_lens)!=(extent,)*24):
            raise RuntimeError('uniform FULL bucket differs')
        result=ordinary(xs,ps,ms,vocabulary_size)
        execution=engine.graph_runner.last_target_execution
        engine._inputs341=(xs[0],ps[0],m)
        engine._calls341.append(dict(family=family,extent=extent,M=120,groups=24,
            tree_attention=bool(m.tree_attention), sequence_lens=list(m.sequence_lens),
            actual_seq_lengths_q=list(m.actual_seq_lengths_q),
            mask_shape=list(m.tree_attention_mask.shape) if m.tree_attention else None,
            used_aclgraph=execution.used_aclgraph,mode=execution.mode,
            fallback_reason=execution.fallback_reason))
        return result
    engine.graph_runner.run_target_greedy=greedy
    return dict(target_only=True, actual_M=120, groups=24,
        all_original_wrapper_code_and_signatures_preserved=True,
        original_model_forward_unchanged=True, Native_flags_assigned=False,
        deferred_read_done=False, actual_Native_HOST_builder_bound=True,
        original_inference_decorators_preserved=True,
        wrappers=[f.__qualname__ for f in graph])


def reference_frame(engine,case,*,owners,positions,tokens,future,extent):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from host_target_metadata import make_host_tree_attention_metadata
    ids=list(owners)+[463+j for j in range(24-len(owners))]
    frontiers=list(positions)+[0]*(24-len(owners))
    inputs=list(tokens)+[0]*(24-len(owners))
    plans=[cached_cpu_tree_speculation_plan(1,4,p,4096,candidate_budget=4) for p in frontiers]
    for owner,p in zip(ids,frontiers,strict=True):
        engine._ensure_cache_capacity([owner],[p+4])
    causal=case['family']=='causal-inner1'
    x,p,m=make_host_tree_attention_metadata(engine,plans,inputs,[[future]*4 for _ in ids],
        sequence_ids=ids,allow_causal_fast_path=causal)
    if not causal:
        m=canonical(engine,m,width=5,groups=24,reference=True)
        m=replace(m,sequence_lens=(extent,)*24)
    elif m.tree_attention:
        raise RuntimeError('independent reference uses actual causal family')
    return x,p,m
