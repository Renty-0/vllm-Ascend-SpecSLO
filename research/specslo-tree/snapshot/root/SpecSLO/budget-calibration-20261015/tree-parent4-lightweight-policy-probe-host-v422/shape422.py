"""HOST-only allocation probe that avoids copying the growing audit history."""
from collections import Counter
import copy
import hashlib
import inspect

import shape420 as original


def probe_policy(policy):
    from budget_policy import TreeBudgetPolicy
    if (type(policy) is not TreeBudgetPolicy
            or policy.gain.__func__ is not TreeBudgetPolicy.gain):
        raise TypeError('HOST422 exact retained gain policy required')
    result=copy.copy(policy)
    result.counts=Counter(policy.counts)
    # The exact retained gain function reads base/observed; the exact retained
    # shape_draft_work writes only counts on this policy object. Admission
    # selection/finish happen outside the probe. The full diagnostic history
    # can be shared without a deep copy or any writes through this object.
    return result


def sources():
    before=inspect.getsource(original.propose)
    old='probe_policy=copy.deepcopy(policy)'
    new='probe_policy=_probe_policy422(policy)'
    if before.count(old)!=1:raise RuntimeError('HOST422 exact retained allocation probe anchor')
    after=before.replace(old,new)
    if after.replace(new,old)!=before:raise RuntimeError('HOST422 whole proposal source inverse')
    return before,after


def construct():
    namespace=dict(vars(original),_probe_policy422=probe_policy)
    exec(compile(sources()[1],__file__+'::same-allocation-light-probe','exec',dont_inherit=True),namespace)
    return namespace['propose']


propose=construct()


def descriptor():
    before,after=sources()
    return dict(original_proposal_sha256=hashlib.sha256(before.encode()).hexdigest(),
        projected_proposal_sha256=hashlib.sha256(after.encode()).hexdigest(),
        entire_proposal_source_inverse=True,only_probe_counter_is_copied=True,
        original_gain_and_budget_allocation_unchanged=True,
        NPU_used=False,serving_installed=False,Raw=None,Goodput=None)
