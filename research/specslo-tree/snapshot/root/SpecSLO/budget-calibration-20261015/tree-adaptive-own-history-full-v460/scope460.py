"""Original432 permission plus a fresh same-model693 adaptive target proof."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import scope432 as original
import scope425

_ISSUER=object()
def need(value,message):
    if value is not True:raise RuntimeError('ROOT460 scope: '+message)

def validate_proof(engine):
    proof=getattr(engine,'_adaptive460_proof',None)
    need(proof is not None and proof['cases']==693 and proof['same_model_instance']is True
         and proof['global_active_ceiling']==64 and proof['Native_flags_assigned']is False,
         'actual current693 proof and original64 configuration')
    need(proof['parameter_SHA_before']==proof['parameter_SHA_after']and bool(proof['parameter_SHA_before']),
         'all original model parameter bytes unchanged')
    need([bank['name']for bank in proof['banks']]==['mixed','mid','short'],'all3 fresh prefix banks')
    for bank in proof['banks']:
        path=Path(bank['path']);need(hashlib.sha256(path.read_bytes()).hexdigest()==bank['sha256'],'actual original bank proof SHA')
        value=json.loads(path.read_text());rows=value['math']['cases']
        need(value['rank']==engine.rank and len(rows)==231 and bank['cases']==231,'all actual bank cases')
        if not engine.is_draft:
            need(value['bank']['independent_neural_prefills']==72
                 and value['bank']['all_triplet_roots_equal']is True and value['bank']['all_triplet_prefix_KV_equal']is True
                 and value['math']['all_independent_prefixes_unchanged']is True,'72 independent own prefixes unchanged')
            for row in rows:
                need(row['exact_greedy']is True and row['hidden_bitwise_equal']is True and row['KV_bitwise_equal']is True
                     and row['graph_eager_KV_bitwise_equal']is True and row['candidate_KV_copied']is False
                     and row['candidate_tokens']==row['reference_tokens'],'every actual independent node vector and token exact')
                need(all(frame['physical_queries']==row['case']['M']for frame in row['independent_reference_frames']),
                     'reference computed at the same physical geometry')
        else:need(value['bank']is None and all(row is None for row in rows),'draft claims no target mathematical computation')
    return proof

def current(engine):
    inherited=scope425.current(engine)
    permission=getattr(engine,'_adaptive460_permission',None)
    need(permission is not None and permission['issuer']is _ISSUER and permission['engine']is engine
         and permission['proof']is getattr(engine,'_adaptive460_proof',None),'same actual adaptive proof and model instance')
    state=engine._branch_tail_state
    need(state.adaptive377_enabled is True and state.prepare is permission['prepare']
         and engine.graph_runner.run_target_greedy is permission['greedy'],'same declared adaptive dispatcher')
    if not engine.is_draft:
        runner=engine.graph_runner
        for key,entry in permission['entries'].items():
            need(runner.target_entries.get(key)is entry and entry.runtime_validated is True
                 and key not in runner.disabled_entry_keys,'same naturally validated additional80 graph entries')
    return inherited

@dataclass(frozen=True)
class Scope:
    engine:object
    original:object
    permission:object
    issuer:object
    def require(self,engine):
        need(self.issuer is _ISSUER and self.engine is engine and self.original.require(engine)==frozenset(),
             'all original432/425/257/209 obligations')
        current(engine);return frozenset()

def issue(engine,*,component_proof):
    proof=validate_proof(engine)
    need(engine.cache_allocation is None and engine.cache_block_tables is None
         and not engine.graph_runner.graph_cache_sealed,'all current proofs released before original warm and seal')
    need(not hasattr(engine,'_adaptive460_permission'),'issue actual adaptive scope once')
    inherited=original.issue(engine,component_proof=component_proof)
    entries={}
    if not engine.is_draft:
        for capacity in(10,11):
            shape=(((16,1,capacity,4096),(16,32)),)
            key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{shape}',(80,))
            entry=engine.graph_runner.target_entries.get(key)
            need(entry is not None and entry.runtime_validated is True and key not in engine.graph_runner.disabled_entry_keys,
                 'both actual additional80 causal and branch graph families naturally validated')
            entries[key]=entry
    state=engine._branch_tail_state;state.adaptive377_enabled=True
    permission=dict(issuer=_ISSUER,engine=engine,proof=proof,entries=entries,prepare=state.prepare,
                    greedy=engine.graph_runner.run_target_greedy)
    engine._adaptive460_permission=permission
    result=Scope(engine,inherited,permission,_ISSUER);result.require(engine);return result

def report(scope,engine):
    scope.require(engine)
    value=original.report(scope.original,engine)
    return dict(value,schema='ROOT460.original432-plus-current-own-physical693.v1',
                adaptive_physical_M=[80,120],adaptive_physical_S=[16,24],current_additional_own_physical_cases=693,
                all_original227_681_75_6_obligations_retained=True,global_active_limit=64,
                target_logical_limit=24,original_MC2_profile_unchanged=True,
                no_Native_qualification_flag_assignment=True,whole_workload_AR_qualified=False,
                formal_Goodput=False,adopted=False)
