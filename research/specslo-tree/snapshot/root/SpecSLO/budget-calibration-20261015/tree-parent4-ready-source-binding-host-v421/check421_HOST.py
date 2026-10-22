"""Real CPU tree plans and controller lifecycle; no Native model or NPU."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace


def main():
    import torch
    import torch_npu
    torch.set_num_threads(1)
    def blocked(*a,**k):raise AssertionError('HOST421 may not initialize an NPU')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    from vllm_ascend.spec_decode.pearl import spec_rhythm as rhythm
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan
    import parents421 as candidate
    rng=random.Random(421);cases=0
    for trial in range(128):
        n=rng.randrange(1,65)
        ctl=rhythm.SpecRhythmPipelineController({i:rhythm.SpecRhythmRuntimeState(i,i%2,max_gamma=5) for i in range(n)})
        payloads={};states={};expected=[]
        for i in range(n):
            b=1+i%5;prefix=128+i
            plan=pack_selected_tree_plan(
                cached_cpu_tree_speculation_plan(2,4,prefix,4096,candidate_budget=8),range(b))
            parent=ctl.new_ticket(i,gamma=b,eager=False);ctl.publish([parent])
            ticket=parent
            if b==4 and trial%2:
                child=ctl.new_ticket(i,gamma=b,eager=True);ctl.publish([child])
                got=ctl.finish_verification(i,fully_accepted=True,proposed_tokens=b,
                    accepted_tokens=b,delivered_tokens=b+1,draft_confidence=.5,ema_alpha=.1)
                assert got is child;ticket=child
            payloads[ticket.proposal_id]=dict(ticket=ticket,plan=plan,row=list(range(b)))
            states[i]=SimpleNamespace(token_ids=[0]*(prefix+1))
            if b==3 or b==4 and not ticket.eager:expected.append(i)
        before=copy.deepcopy(ctl)
        result=candidate.collect(ctl,payloads,range(n),states)
        assert [r.owner for r in result]==expected
        assert ctl.__dict__==before.__dict__
        assert all(r.budget==3 or r.normal_B4 for r in result)
        for r in result:
            assert r.proposal_id==ctl.ready[r.owner].proposal_id
            assert r.prefix_epoch==ctl.request_states[r.owner].prefix_epoch
        cases+=1
    # Corruptions must reject before any device/model operation.
    owner=next(iter(ctl.ready));payload=payloads[ctl.ready[owner].proposal_id]
    original=payload['ticket'];payload['ticket']=copy.copy(original)
    try:candidate.collect(ctl,payloads,[owner],states)
    except RuntimeError:pass
    else:raise AssertionError('copied ticket identity accepted')
    payload['ticket']=original
    states[owner].token_ids.append(1)
    try:candidate.collect(ctl,payloads,[owner],states)
    except RuntimeError:pass
    else:raise AssertionError('stale prefix accepted')
    value=dict(status='PASS_HOST421_ACTUAL_CPU_READY_PARENTS_PREFIX_EPOCH_AND_TICKET_IDENTITY',
        CPU_cases=cases,actual_original_controller_and_CPU_tree_plans=True,
        normal_B4_only=True,promoted_B4_and_branch_B5_excluded=True,
        original_parent3_route_preserved=True,no_controller_mutations=True,
        stale_prefix_and_ticket_identity_rejected=True,NPU_used=False,
        serving_installed=False,NN_permission=False,Raw=None,Goodput=None)
    p=Path(__file__).resolve().parent
    (p/'HOST421_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
