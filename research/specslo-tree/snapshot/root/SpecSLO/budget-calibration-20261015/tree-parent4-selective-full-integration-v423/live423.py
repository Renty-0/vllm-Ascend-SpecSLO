"""Exact E4 execution gate; no permission exists on import.

Diagnostic calls retain419's finally-scoped gate. The full candidate must
issue an engine-owned production record from fresh current-process proofs.
"""
from types import FunctionType

import live419
from scope159 import Capability,native_ready


def need(value,message):
    if value is not True:raise RuntimeError('ROOT423 E4 gate: '+message)


def diagnostic(engine):
    value=getattr(engine,'_v199_diagnostic',None)
    return (type(value) is tuple and len(value)==2 and value[0] is live419._TOKEN
            and getattr(engine,'_v173_qualified_component',None) is not None)


def permission(engine):
    value=getattr(engine,'_root423_E4_permission',None)
    need(value is not None and value.owner is engine and value.model is engine.model,
         'current engine/model proof record required')
    from proof423 import verify_record
    verify_record(engine, value)
    need(value.bounded is getattr(engine,'_root423_bounded_actual',None),
         'same current completed75-case protocol proof object')
    runner=engine.graph_runner
    if engine.is_draft:
        for kind,key,entry in value.entries:
            actual=runner.draft_entries if kind=='draft' else runner.target_entries
            need(actual.get(key) is entry and entry.runtime_validated is True
                 and key not in runner.disabled_entry_keys,'actual naturally validated entry unchanged')
    return value


def precheck(engine,plans,ids,sources):
    if diagnostic(engine):return
    value=permission(engine)
    work=getattr(engine,'_root423_current_work',None)
    need(work is not None and work.owner is engine
         and work.generation==engine._dynamic_tree_generation,
         'current actual shaped work binding required')
    need(tuple(ids)==work.ids and tuple(sources)==work.eager_ids
         and len(ids)==28 and len(sources)==4
         and all(p.candidate_budget==4 for p in plans),
         'exact qualified normal24/B4 plus future4/B4 work')
    for owner,plan,row in work.parents:
        actual=sources[owner]
        need(actual[0] is plan and tuple(actual[1])==row,
             'actual parent plan and all four dependency tokens unchanged')
    need(value.capability.mode=='production','explicit current production capability')


def validate(engine,native,plans,ids,sources,repairs,frontiers,steps):
    if not sources:return
    if diagnostic(engine):
        return live419.validate(engine,native,plans,ids,sources,repairs,frontiers,steps)
    precheck(engine,plans,ids,sources)
    current=permission(engine)
    normal=[p for p,i in zip(plans,ids,strict=True) if i not in sources]
    need(steps==4 and len(normal)==24 and len(frontiers)==4 and 0<=len(repairs)<=24
         and all(len(row[2])==1 for row in repairs)
         and all(p.candidate_budget==4 for p in plans)
         and all(p.candidate_budget==4 for p,_ in sources.values()),
         'exact current N24/E4/R0..24 one-token repair domain')
    runner=engine.graph_runner
    if not repairs:
        key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|paged',(4,))
        entry=runner.target_entries.get(key)
        need(entry is not None and entry.runtime_validated is True
             and key not in runner.disabled_entry_keys,'closed R0 frontier graph')
    need(native_ready(engine,current.capability,chain_M=28,real_rows=28,
         union_M=native._next_linear_draft_graph_bucket(4+len(repairs),64) if repairs else None),
         'closed shared4 and actual union graph objects')


_FUSED_GLOBALS=dict(live419.fused_prelude.__globals__)
fused_prelude=FunctionType(live419.fused_prelude.__code__,_FUSED_GLOBALS,
    live419.fused_prelude.__name__,live419.fused_prelude.__defaults__,live419.fused_prelude.__closure__)


def activate(engine,record):
    """Accept only the current record created by the explicit proof issuer."""
    from proof423 import verify_record
    need(verify_record(engine,record) is True,'complete current proof issuer and exact record required')
    need(engine.cache_allocation is None and engine.cache_block_tables is None
         and not engine.graph_runner.graph_cache_sealed,'released fixture cache before production warm')
    need(not hasattr(engine,'_root423_E4_permission'),'one engine-owned capability issuance')
    _FUSED_GLOBALS['_capability']=record.capability
    engine._root423_E4_permission=record
    permission(engine)
    return dict(mode='production',same_engine_protocol_cases=75,
        ordinary_parent4_only=True,Native_flags_assigned=False,full463_qualified=False)
