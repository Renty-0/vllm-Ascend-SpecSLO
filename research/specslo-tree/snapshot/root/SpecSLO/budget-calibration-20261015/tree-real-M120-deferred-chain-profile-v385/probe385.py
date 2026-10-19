"""Capture genuine layer0 M120 payloads, then use the unchanged chain qualifier."""
import json
from pathlib import Path


def run(engine,native,stage,output,fixture):
    target=not engine.is_draft
    from static_state257 import install
    from bank251 import build
    from oracle255 import cases,prepare
    from dual_bank_v146 import prefix_evidence
    from original385 import run as original
    activation=stage('ROOT385/install-unchanged-FULL120-projection',lambda:install(engine,native))
    def safe():
        if target and (120 not in engine.model._mc2_chained_row_counts or 120 in engine.model._mc2_deferred_row_counts):
            raise RuntimeError('ROOT385 actual old fully-flushed120 profile required for capture')
        return True
    stage('ROOT385/check-original-per-route-flush',safe)
    bank=build(engine,native,stage,fixture['prefixes'])
    def capture():
        if not target:return None
        import torch
        kind=type(engine.model.layers[0].input_layernorm);old=kind.forward_mc2;payloads={}
        def observed(norm,x,residual,weight,context,**kwargs):
            route=kwargs.get('static_route')
            if route is not None and route.layer_index==0 and route.projection_kind not in payloads:
                if x.shape[0]!=120:raise RuntimeError('ROOT385 genuine120 rows required')
                payloads[route.projection_kind]=dict(activation=x.detach().clone(),weight=weight,
                    residual=residual.detach().clone(),gamma=norm.weight)
            return old(norm,x,residual,weight,context,**kwargs)
        kind.forward_mc2=observed
        try:
            case=next(c for c in cases()if c['name']=='N24-all4-anchor0')
            plans,rows=prepare(case,bank['prefixes']);roots=bank['roots'][:24]
            pp,rr,dd,ss,real=engine._pad_spec_rhythm_target_tree_graph(plans,roots,rows,list(range(24)),[],[])
            engine.target_tree_forward(pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
            torch.npu.synchronize()
        finally:kind.forward_mc2=old
        if set(payloads)!={'attention','down'}:raise RuntimeError('ROOT385 both actual layer0 routes captured')
        return payloads
    payloads=stage('ROOT385/capture-real-inputs-without-changing-outputs',capture)
    measured=stage('ROOT385/original128-epilogue-graph-qualifier-and-balanced-cost',
        lambda:original(engine,payloads,output)if target else None)
    def protected():
        if not target:return None
        for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
            if prefix_evidence(engine,{owner:bank['prefixes'][owner%24]})!=expected:
                raise RuntimeError('ROOT385 original committed prefix KV unchanged')
        return True
    intact=stage('ROOT385/all72-prefixes-after-independent-operator-sequence',protected)
    stage('ROOT385/release-bank',lambda:engine._release_cache()if target else None)
    return dict(rank=engine.rank,activation=activation,measurement=measured,
        independent_prefills=72 if target else 0,all72_prefixes_protected=intact,
        current_profile_unchanged=True,new_profile_installed=False,
        operator_qualification_only=True,full463=False,adopted=False,Raw=None,Goodput=None)
