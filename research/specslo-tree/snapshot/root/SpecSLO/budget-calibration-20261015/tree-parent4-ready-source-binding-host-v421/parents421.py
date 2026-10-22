"""Read-only HOST binding of actual ready normal-B4 parents.

Only normal B4 tickets are proposed for the pending finite E4 path. Promoted
eager B4 tickets and B5 branches retain their previous exclusion. No neural
scope is granted here; the production caller still needs same-engine proof.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Parent:
    owner:int
    budget:int
    proposal_id:int
    prefix_epoch:int
    prefix_len:int
    normal_B4:bool


def collect(controller,payloads,indices,local_states):
    indices=tuple(indices)
    if len(indices)!=len(set(indices)):raise ValueError('HOST421 unique eager owners required')
    result=[]
    for owner in indices:
        ticket=controller.validate_verification(owner)
        payload=payloads.get(ticket.proposal_id)
        if payload is None or payload.get('ticket') is not ticket:
            raise RuntimeError('HOST421 exact ready payload/ticket identity required')
        plan=payload['plan'];budget=plan.candidate_budget;row=payload['row']
        if (type(budget) is not int or not 1<=budget<=5 or ticket.gamma!=budget
                or len(row)!=budget or any(type(v) is not int for v in row)):
            raise RuntimeError('HOST421 complete original ready candidate rows required')
        if plan.parent_indices.device.type!='cpu' or plan.positions.device.type!='cpu':
            raise RuntimeError('HOST421 parent metadata must remain HOST-only')
        prefix=len(local_states[owner].token_ids)-1
        if plan.prefix_len!=prefix:
            raise RuntimeError('HOST421 stale actual committed prefix')
        normal4=(budget==4 and not ticket.eager)
        if normal4 and (plan.width!=2 or plan.depth!=4
                or tuple(plan.parent_indices.tolist())!=(-1,0,1,2)
                or tuple(plan.positions.tolist())!=tuple(range(prefix,prefix+5))):
            raise RuntimeError('HOST421 normal B4 requires the complete primary spine')
        if budget==3 or normal4:
            result.append(Parent(owner,budget,ticket.proposal_id,
                controller.request_states[owner].prefix_epoch,prefix,normal4))
    return tuple(result)
