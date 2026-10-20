"""Explicit pre-install factory and cheap union execution for future full run.

Not installed on import. No receipt transfers numerical qualification.
"""
import sys,types
from pathlib import Path
from scope159 import Capability,require,bucket,native_ready
from lease159 import guarded
from source159 import P,BASE,FUSION,checked,module,fast_source,shape_namespace,controls_namespace,tree_entry_namespace,require_artifacts
_capability=Capability()
_factory=None

def prepare_helpers(capability):
    """Call once BEFORE original root_seed/tree_entry installation.

    root_seed imports canonical FastV5 and captures it as its fallback, so an
    after-install monkeypatch would incorrectly bypass normal-only root_seed.
    Runtime source/origin closure must explicitly bind this new module factory.
    """
    global _capability,_factory
    require(type(capability)is Capability,'explicit current capability')
    require('fast_tree_draft_v5' not in sys.modules,'prepare before canonical FastV5 install')
    require_artifacts()
    _capability=capability
    fusion=module('repair_frontier_fusion',FUSION)
    source,proof=fast_source();fast=types.ModuleType('fast_tree_draft_v5')
    fast.__file__=str(P/'fast159.generated.py');fast.__package__=''
    exec(compile(source,fast.__file__,'exec'),fast.__dict__)
    fast._v159_source_contract=proof;sys.modules['fast_tree_draft_v5']=fast
    shape,_=shape_namespace(capability)
    _factory=dict(fast=fast,fusion=fusion,shape=shape,controls=controls_namespace(shape),
        tree_entry=tree_entry_namespace(P),capability=capability)
    return _factory

def finish_install(native,live_decode_patch):
    """After returned tree_entry.install(), BEFORE CandidateControls/live install.

    This adds exactly one common status reduce to each original draft stage.
    No new collective is executed inside the asymmetric draft/target forwards.
    """
    require(_factory is not None,'prepare helpers before original installation')
    from status159 import install
    return install(native,live_decode_patch)

def activate_after_component(capability,*,engine,verify_actual_component):
    """Explicit common untimed boundary; a future actor supplies its real gate.

    No fallback to a retained receipt, old flags or a constant-success default.
    Caller must run the gate on all ranks in one original common stage before
    using this function, then run new warm/two-zero/seal/full in this process.
    """
    global _capability
    require(_factory is not None and type(capability)is Capability and capability.mode=='production','explicit production domain')
    require(engine.cache_allocation is None and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,
        'released fixture cache before production warm/seal')
    require(callable(verify_actual_component) and verify_actual_component(engine,capability) is True,'actual own-process changed-component gate')
    _capability=capability;_factory['shape']['_v159_capability']=capability;_factory['capability']=capability
    return dict(mode=capability.mode,shared4_pairs=sorted(capability.shared4_pairs),union_pairs=sorted(capability.union_pairs),
        Native_flags_assigned=False,formal_Goodput_measured=False)

def fused_prelude(engine,native,runner_cls,plans,ids,sources,repairs,frontiers,steps):
    """Return (None,None) only BEFORE touching new union model/ledger state.

    Unexpected lease/Native errors propagate; original fatal cleanup remains
    mandatory. No catch-and-continue after model writes, fabricated KV or flags.
    """
    if not (steps==4 and sources and repairs and frontiers and all(len(x[2])==1 for x in repairs)):
        return None,None
    normal=[p for p,i in zip(plans,ids) if i not in sources]
    if not normal or max(int(p.candidate_budget) for p in normal)!=4:return None,None
    if any(int(p.candidate_budget)!=3 for p,i in zip(plans,ids) if i in sources):return None,None
    if any(int(p.candidate_budget)!=3 for p,row in sources.values()):return None,None
    N,E=len(normal),len(frontiers);R=len(repairs)
    if not (_capability.allows_shared4(N,E) and _capability.allows_union(E,R)):return None,None
    M=native._next_linear_draft_graph_bucket(len(ids),engine.config.max_num_seqs)
    F=native._next_linear_draft_graph_bucket(E,engine.config.max_num_seqs)
    U=native._next_linear_draft_graph_bucket(F+R,engine.config.max_num_seqs)
    if not native_ready(engine,_capability,chain_M=M,real_rows=len(ids),union_M=U):return None,None
    require(engine.config.max_num_seqs==64 and not engine.config.enable_prefix_caching,'unchanged capacity64, independently owned pages')
    from repair_frontier_fusion import AllocationWitness,plan_frame,build_union,run_union
    import torch
    # FastV5 already reserved all main-chain pages. Complete all prelude capacity
    # preparation before taking the synchronous ownership/ledger lease.
    engine._ensure_cache_capacity([x[0] for x in frontiers]+[x[0] for x in repairs],
        [x[1] for x in frontiers]+[x[1] for x in repairs])
    owners=[x[0] for x in frontiers]+[x[0] for x in repairs]
    with guarded(engine,owners,AllocationWitness) as lease:
        plan=plan_frame(repairs,frontiers,next_bucket=native._next_linear_draft_graph_bucket,witness=lease.check())
        packed=build_union(plan,tensor=lambda values:torch.tensor(values,dtype=torch.long,device=engine.device),
            prepare=engine._prepare_attention_metadata,pad=runner_cls._pad_inputs,cat=torch.cat,lease=lease.check)
        predicted=run_union(plan,packed,runner=engine.graph_runner,model=engine.model,
            vocabulary_size=engine.draft_vocab_size,lease=lease.check).cpu().tolist()
    return plan,predicted
