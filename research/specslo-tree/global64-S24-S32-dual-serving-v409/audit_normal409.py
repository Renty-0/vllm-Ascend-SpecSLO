"""Retained control-plane proof; does not reconstruct model/KV arithmetic."""
import hashlib
import inspect
import textwrap


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT409 normal admission audit: ' + message)


def verify_snapshot(value, generation):
    need(value['available'] is True and value['global_active_limit'] == 64
         and value['normal_work_limit'] == 32 and value['wide_normal_C'] == 88
         and value['wide_normal_B3_owner_limit'] == 24
         and value['actual_policy_object_preserved'] is True,
         'actual engine-owned typed32 C88 resource scope')
    from wide409 import source_descriptor
    need(value['policy_projection'] == source_descriptor(), 'exact whole typed high-load projection source')
    state = value['actual_original_normal_admission']
    need(state['available'] is True and state['generation'] == generation
         and state['rule'] == 'V140.normal-class-priority-defer-age-owner-order.v1',
         'actual original normal32 rule and issued generation')
    histograms = [state[key] for key in ['raw_N_histogram', 'admitted_N_histogram', 'positive_grant_N_histogram']]
    need([len(hist) for hist in histograms] == [65, 33, 33]
         and all(type(n) is int and n >= 0 for hist in histograms for n in hist)
         and all(sum(hist) == state['call_count'] for hist in histograms), 'all actual call count histograms')
    priority = {'tight': 3, 'normal': 2, 'loose': 1}
    calls = []
    overflow = 0
    for row in state['calls']:
        raw, admitted = row['raw_normal'], row['admitted_normal']
        granted = {r['owner']: r['B'] for r in row['positive_normal_grants']}
        need(type(row['call']) is int and 0 <= row['call'] < state['call_count']
             and row['call'] not in calls, 'retained unique original call ordinal')
        calls.append(row['call'])
        need(row['raw_N'] == len(raw) <= 64 and len(set(raw)) == len(raw)
             and row['admitted_N'] == len(admitted) <= 32
             and row['positive_grant_N'] == len(granted)
             and set(granted) <= set(admitted) <= set(raw)
             and all(type(b) is int and 1 <= b <= 5 for b in granted.values()), 'whole normal owners and grants')
        if len(admitted) > 24 or row['explicit_active_batch_size'] > 48:
            need(all(b <= 3 for b in granted.values())
                 and sum(b == 3 for b in granted.values()) <= 24
                 and sum(granted.values()) <= 88, 'actual retained wide C88/B3 quota')
        need(row['actual_work_normal_row_limit'] == 32 and row['actual_work_combined_row_limit'] == 64
             and type(row['explicit_active_batch_size']) is int
             and len(raw) + row['eager_N'] <= row['explicit_active_batch_size'] <= 64,
             'global bound and declared actual policy/hardware row bounds')
        need(row['cap_deferred_normal'] == [i for i in raw if i not in admitted]
             and row['zero_grant_deferred_normal'] == [i for i in admitted if i not in granted]
             and row['all_deferred_normal'] == [i for i in raw if i not in granted],
             'all retained deferred requests accounted without removal')
        need(len(row['classes']) == len(row['ages_before']) == len(row['ages_after']) == len(raw)
             and all(c in priority for c in row['classes'])
             and all(type(age) is int and age >= 0 for age in row['ages_before']), 'full class and age inputs')
        need(row['ages_after'] == [0 if i in granted else age + 1
                                  for i, age in zip(raw, row['ages_before'], strict=True)],
             'exact original successful-grant age reset and deferred age increment')
        is_overflow = len(raw) > 32
        need(row['overflow'] is is_overflow, 'truthful policy overflow')
        if is_overflow:
            ranked = sorted(range(len(raw)), key=lambda j: (
                -priority[row['classes'][j]], -row['ages_before'][j], raw[j]))
            chosen = {raw[j] for j in ranked[:32]}
            need(admitted == [i for i in raw if i in chosen], 'actual original class-age-owner policy at32')
            overflow += 1
        else:
            need(admitted == raw, 'narrow raw owner set retained')
    need(calls == sorted(calls) and overflow == state['overflow_calls'], 'all overflow records retained')
    return dict(generation=generation, calls=state['call_count'], overflow_calls=overflow,
                retained_interesting_calls=len(calls), global_limit=64, normal_work_limit=32,
                every_deferred_owner_retained=True, exact_class_age_policy=True,
                neural_math_recomputed=False)
