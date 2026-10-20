"""Same-process structural capability issued after actual independent actors.

The capability describes representative Native layout classes. It never claims
arbitrary-token/path exact equality, own AR, or unchanged online KV history.
"""
from dataclasses import dataclass,field
from pathlib import Path

_ISSUER=object()


def target_key(segments):
    return ('target-hidden:0|steps:1|tree-fia-full:exact-length-task-update|buffers:'
        +repr((((segments,1,128,4096),(segments,32)),)),(197,))


def validate_inventory(before,after,*,is_draft):
    """Exact real inventory delta, reused by the independent HOST auditor."""
    names={'entries','draft_entries','target_entries'}
    if set(before)!=names or set(after)!=names:
        raise RuntimeError('V101 actual Native inventory families differ')
    for name in names:
        if (type(before[name]) is not list or type(after[name]) is not list
            or before[name]!=sorted(set(before[name])) or after[name]!=sorted(set(after[name]))):
            raise RuntimeError('V101 actual Native inventory must be complete and unique')
        additions=set(after[name])-set(before[name])
        wanted={repr(target_key(19)),repr(target_key(35))} if not is_draft and name=='target_entries' else set()
        if additions!=wanted or not set(before[name])<=set(after[name]):
            raise RuntimeError('V101 independent actors changed an unexpected Native key')


@dataclass(frozen=True)
class StructuralScope:
    counts:frozenset[int]
    _engine:object=field(repr=False,compare=False)
    _entry:object=field(repr=False,compare=False)
    _issuer:object=field(repr=False,compare=False)

    def require(self,engine):
        if (self._issuer is not _ISSUER or engine is not self._engine
            or self.counts!=frozenset(range(17,33))):
            raise RuntimeError('V101 lacks the current actual structural scope')
        if not engine.is_draft:
            entry=engine.graph_runner.target_entries.get(target_key(35))
            if (entry is not self._entry or entry.runtime_validated is not True
                or target_key(35) in engine.graph_runner.disabled_entry_keys):
                raise RuntimeError('V101 original S35 Native entry was replaced/disabled/unqualified')
        return self.counts


def qualify_structures(engine,*,stage_all4):
    """Only this actual actor producer may issue the engine-local capability."""
    from diagnostic_authorization import require_bounded_scope
    require_bounded_scope()
    from local_mixed_qualification import run_prepared_actor_chain
    from bounded_report_audit_v78 import chain_record
    from local_qualification_scope import SHORT,LONG,OLD
    if getattr(engine,'_v101_structural_scope',None) is not None:
        raise RuntimeError('V101 structural actor cannot be substituted or repeated')
    before=stage_all4('V101-actual-inventory-before-structures',lambda:
        {name:sorted(repr(key) for key in getattr(engine.graph_runner,name))
         for name in ('entries','draft_entries','target_entries')})
    chain=run_prepared_actor_chain(engine)
    after=stage_all4('V101-actual-inventory-after-structures',lambda:
        {name:sorted(repr(key) for key in getattr(engine.graph_runner,name))
         for name in ('entries','draft_entries','target_entries')})
    chain['actual_inventory']=dict(before=before,after=after)
    from serialized_structure_evidence_v129 import retain_structure_evidence
    chain,evidence=retain_structure_evidence(engine,chain,stage_all4=stage_all4)
    engine._v129_structure_evidence=evidence
    def validate():
        if engine.device.type!='npu' or engine.graph_runner.graph_cache_sealed:
            raise RuntimeError('V101 actual structure qualification must precede seal')
        role='draft' if engine.is_draft else 'target'
        chain_record(chain,role,rank=engine.rank)
        if (len(SHORT)!=16 or {s.target_requests for s in SHORT}!=set(range(17,33))
            or len(LONG)!=3 or len(OLD)!=2):
            raise RuntimeError('V101 full representative coverage changed')
        entry=None
        if not engine.is_draft:
            entry=engine.graph_runner.target_entries.get(target_key(35))
            old=engine.graph_runner.target_entries.get(target_key(19))
            if (entry is None or old is None or entry.runtime_validated is not True
                or old.runtime_validated is not True):
                raise RuntimeError('V101 both actual Native S19/S35 gates must be closed')
        validate_inventory(before,after,is_draft=engine.is_draft)
        return StructuralScope(frozenset(range(17,33)),engine,entry,_ISSUER)
    scope=stage_all4('V101-complete-actual-structure-and-original-protocol-gate',validate)
    engine._v101_structural_scope=scope
    return chain,scope


def scope_report(scope,engine):
    counts=scope.require(engine)
    return dict(schema='V101.actual-structure-representatives.v1',N=sorted(counts),
        physical_M=197,physical_S=35,mask=[35,1,128,4096],C=96,Q=128,
        qualification_scope='one genuine short-prefix heterogeneous representative per N plus three original long-prefix S35 controls; source HOST domain checks every call',
        arbitrary_token_or_path_exact_equivalence_claimed=False,
        exact_Btuple_device_enumeration_claimed=False,actual_workload_KV_reconstructed=False,
        independent_own_AR_qualified=False,online_qualified=False,formal_Goodput=False)
