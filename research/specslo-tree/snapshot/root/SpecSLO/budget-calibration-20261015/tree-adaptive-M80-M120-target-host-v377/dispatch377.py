"""Independent complete geometry/capture reconstruction for all four families."""
from source377 import function
from bucket377 import extent


def audit_calls(cases,calls,prefixes):
    from audit_static_calls257 import audit_calls as original
    changes = (
        ('from projection255 import plan','from physical377 import plan as _plan'),
        ("physical=plan(ends,case['query_count'])", "physical=_plan(ends,case['query_count'],case['M'])"),
        ('cold=capacity not in seen;count=3 if cold else 1',
         "family=(case['M'],capacity);cold=family not in seen and cursor<len(calls) and calls[cursor]['execution_mode']=='capture_replay';count=3 if cold else 1"),
        ("call['M']==120 and call['attention_segments']==24", "call['M']==case['M'] and call['attention_segments']==case['M']//5"),
        ("call['actual_seq_lengths_q']==list(range(5,121,5))", "call['actual_seq_lengths_q']==list(range(5,case['M']+1,5))"),
        ("call['sequence_lens']==[2560]*24", "call['sequence_lens']==[_ROOT377_extent(case)]*(case['M']//5)"),
        ("call['mask_shape']==[24,1,capacity,4096]", "call['mask_shape']==[case['M']//5,1,capacity,4096]"),
        ('if cold:families.append(capacity);extra+=2','if cold:families.append(family);extra+=2'),
        ('seen.add(capacity);cursor+=count','seen.add(family);cursor+=count'),
        ('if cursor!=len(calls)or sorted(families)!=[10,11]:',
         'if cursor!=len(calls)or seen!={(80,10),(80,11),(120,10),(120,11)}:'),
    )
    projected = function(original,changes,dict(_ROOT377_extent=lambda case:extent(tuple(
        len(prefixes[i])+min(budget,4)+1 for i,budget in enumerate(case['budgets'])))))
    result = projected(cases,calls)
    for call in calls:
        details = call['uniform_KV_bucket377']
        if (call['adaptive377_enabled'] is not True or details['enabled'] is not True
                or call['sequence_lens'] != [details['extent']]*(call['M']//5)
                or details['extent'] != extent(tuple(details['original_sequence_lens']))):
            raise RuntimeError('ROOT377 actual selected M and complete visible KV extent differ')
    return dict(result, all_four_physical_families_reconstructed=True,
        independent_reference_FULL120=True, Native_flags_assigned=False)
