"""Additional fresh693 own-physical proof on the already installed model."""
from pathlib import Path
import hashlib,json

def save(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')

def qualify(engine,native,stage,output):
    from bank257 import build
    from oracle459 import run,cases
    from dispatch377 import audit_calls
    from prefixes425 import mid_prefixes,short_prefixes
    from probe459 import fingerprints
    target=not engine.is_draft;state=engine._branch_tail_state
    if engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed or state.adaptive377_enabled:
        raise RuntimeError('ROOT460 old227/681 finished before new proof; one existing unsealed state')
    if engine.config.max_num_seqs!=64:raise RuntimeError('ROOT460 same actual global64')
    fixture=json.loads((Path(__file__).parent.parent/'tree-static-q5-M120-SLO-first-full463-own-AR-v257/FIXTURE257.json').read_text())
    before=stage('ROOT460/additional693-parameter-SHA-before',lambda:fingerprints(engine))
    pointers=[(name,value.data_ptr())for name,value in engine.model.named_parameters()]
    state.adaptive377_enabled=True
    root=Path(output)/'adaptive-own-physical-node-math460';summaries=[]
    stage('ROOT460/create-additional-output',lambda:root.mkdir(exist_ok=False)if engine.rank==0 else None)
    try:
        for name,prefixes in [('mixed',fixture['prefixes']),('mid',mid_prefixes(fixture['prefixes'])),('short',short_prefixes(fixture['prefixes']))]:
            lane=lambda label,fn:stage('ROOT460/'+name+'/'+label,fn)
            folder=root/name;lane('create-bank-output',lambda:folder.mkdir(exist_ok=False)if engine.rank==0 else None)
            bank=build(engine,native,lane,prefixes);start=len(state.static_graph_calls)
            math=run(engine,native,lane,bank,folder)
            calls=list(state.static_graph_calls[start:])
            dispatch=lane('audit-all-real-dispatches',lambda:audit_calls(cases(),calls,prefixes)if target else None)
            if target and not(all(r['exact_greedy']and r['hidden_bitwise_equal']and r['KV_bitwise_equal']and r['graph_eager_KV_bitwise_equal']for r in math['cases'])and math['all_independent_prefixes_unchanged']is True):
                raise RuntimeError('ROOT460 every actual same-M node and independent prefix must be exact')
            path=folder/f'rank{engine.rank}-proof460.json'
            lane('retain-same-instance-math',lambda:save(path,dict(rank=engine.rank,bank=bank['evidence'],prefix_lengths=[len(p)for p in prefixes],math=math,actual_calls=calls,dispatch=dispatch)))
            summaries.append(dict(name=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=231))
            lane('release-independent-bank',lambda:engine._release_cache()if target else None)
    finally:
        state.adaptive377_enabled=False
    after=stage('ROOT460/additional693-parameter-SHA-after',lambda:fingerprints(engine))
    if before!=after or pointers!=[(name,value.data_ptr())for name,value in engine.model.named_parameters()]:
        raise RuntimeError('ROOT460 original model parameter bytes and addresses retained')
    proof=dict(cases=693,banks=summaries,same_model_instance=True,original227_plus681_retained=True,
               parameter_SHA_before=before,parameter_SHA_after=after,global_active_ceiling=64,
               original_branch_copy_state_installed_once=True,Native_flags_assigned=False,
               old75_plus6_still_required=True,production_permission=False,full463=False,Raw=None,Goodput=None)
    stage('ROOT460/retain-additional693-complete',lambda:save(root/f'rank{engine.rank}-proof460.json',proof))
    engine._adaptive460_proof=proof
    return proof
