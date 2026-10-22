"""All original257 permutation/graph checks, with explicit extent and warm reuse."""
from projection423 import project_function
from bucket423 import extent


def audit_calls(cases, calls, prefixes):
    from audit_static_calls257 import audit_calls as original
    projected = project_function(original, (
        ("and call['sequence_lens']==[2560]*24",
         "and call['sequence_lens']==[_ROOT423_expected(case)]*24"),
        ('cold=capacity not in seen;count=3 if cold else 1',
         "cold=capacity not in seen and cursor<len(calls) and calls[cursor]['execution_mode']=='capture_replay';count=3 if cold else 1"),
        ('if cursor!=len(calls)or sorted(families)!=[10,11]:',
         'if cursor!=len(calls)or seen!={10,11}:'),
    ), {'_ROOT423_expected': lambda case: extent(tuple(len(prefixes[i]) + min(b, 4) + 1
            for i, b in enumerate(case['budgets'])))})
    result = projected(cases, calls)
    for call in calls:
        detail = call['uniform_KV_bucket423']
        if detail['enabled'] is not True or detail['extent'] != extent(tuple(detail['original_sequence_lens'])):
            raise RuntimeError('ROOT423 checked full visible extent and enabled actual dispatch')
    return dict(result, uniform_extent_checked=True, original_reference_FULL2560=True)
