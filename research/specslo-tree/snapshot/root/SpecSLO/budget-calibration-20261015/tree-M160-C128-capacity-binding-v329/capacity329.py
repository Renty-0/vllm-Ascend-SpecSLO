"""Bind the actual installed policy/accounting instance to C128/Q160/N32."""
import inspect


def bind(policy,native):
    before=(policy.candidate_roof,policy.target_queries,policy.max_target_requests)
    if before!=(96,128,32):
        raise RuntimeError('expected original96/128/32 private policy')
    account=native.NativePearlEngine._specslo_account_real_target_plans
    cells=inspect.getclosurevars(account).nonlocals
    if cells.get('policy')is not policy:
        raise RuntimeError('actual installed accounting callback must own this policy')
    original=policy.check_target
    policy.target_queries=160
    policy.candidate_roof=128

    def checked(budgets):
        values=tuple(budgets)
        if any(type(b)is not int or not 1<=b<=8 for b in values):
            raise ValueError('only declared positive B1..8')
        if not values or len(values)>32 or sum(values)>128 or len(values)+sum(b>4 for b in values)>32 or sum(b>4 for b in values)>12:
            raise RuntimeError('C128/N32 and32 primary-plus-side groups')
        return original(values)

    policy.check_target=checked
    # Closure identity proves all installed accounting callers see this same
    # instance; the original check still enforces root-inclusive160 itself.
    if cells['policy'].check_target is not checked:
        raise RuntimeError('private capacity binding did not reach actual callback')
    return dict(old=dict(C=96,Q=128,N=32),new=dict(C=128,Q=160,N=32,groups=32,branches=12),
        same_actual_accounting_policy_instance=True,original_check_target_code_retained=True,
        positive_per_request_B1to8_and_whole_group_bound_checked=True,
        original_policy_class_unmodified=True,model_loaded=False)
