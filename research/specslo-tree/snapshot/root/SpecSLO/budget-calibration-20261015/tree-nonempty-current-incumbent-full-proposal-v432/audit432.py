"""All original425/428/430 audits with an explicit stronger admission audit."""
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace


def constructed():
    import audit_base425, audit425, audit428, audit430, audit433
    from projection425 import project_function
    from bucket425 import valid_record
    def base_audit():
        return project_function(audit_base425.audit, (
            ("'from empty425 import audit_trace'", "'from admission_audit432 import audit_trace'"),
            ("    if projected==audit_source:",
             "    projected=projected.replace('actual_empty_cohorts=', 'actual_admission_cohorts=').replace('empty_active_cohorts=', 'actual_admission_cohorts=')\n    if projected==audit_source:"),
            ("and row['fixed_KV_extent']==[2560]*24", 'and _ROOT425_valid_record(row)'),
        ), {'_ROOT425_valid_record':valid_record})
    def bind(function, values):
        ns = dict(function.__globals__, **values)
        return FunctionType(function.__code__,ns,function.__name__,function.__defaults__,function.__closure__)
    current425 = bind(audit425.audit, dict(projected_original_audit=base_audit))
    current428 = bind(audit428.audit, dict(original=SimpleNamespace(audit=current425)))
    current430 = bind(audit430.audit, dict(original=SimpleNamespace(audit=current428)))
    current433 = bind(audit433.audit, dict(original=SimpleNamespace(audit=current430)))
    return current433, base_audit


def audit(directory, selected_windows):
    actual, _ = constructed()
    value = actual(directory, selected_windows)
    from incumbents427 import audit as audit_incumbents
    extra = audit_incumbents(Path(directory)/'multiincumbent32-prefill427-own-current-process')
    if extra['passed'] is not True:
        raise RuntimeError('ROOT432 actual same-process original427 cases required')
    for rank in range(4):
        report = json.loads((Path(directory)/f'rank{rank}-local-diagnostic.json').read_text())
        proof = report['ROOT432_current_incumbent_refill']
        if not (proof['current_same_model_proof'] is True and proof['actual_NN_cases']==6
                and proof['proof']['audit']==extra and proof['Native_flags_assigned'] is False):
            raise RuntimeError('ROOT432 actual current427 model proof and original flags retained')
    return dict(value, ROOT432_current_original427_NN_proof=extra,
        ROOT432_exact_original_incumbent_refill_time_charge_audited=True,
        ROOT432_all_previous430_math_Native_full_ownAR_checks_retained=True)
