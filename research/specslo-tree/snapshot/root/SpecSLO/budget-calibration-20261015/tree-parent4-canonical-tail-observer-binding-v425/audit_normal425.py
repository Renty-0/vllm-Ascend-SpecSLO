"""Retained control-plane proof; does not reconstruct model/KV arithmetic."""
import hashlib
import inspect
import textwrap


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT425 normal admission audit: ' + message)


def verify_snapshot(value, generation):
    import split_shape_v132
    before = textwrap.dedent(inspect.getsource(split_shape_v132.shape))
    after = before.replace('if len(raw_normal)>32:', 'if len(raw_normal)>24:')
    need(value['available'] is True and value['global_active_limit'] == 64
         and value['normal_work_limit'] == 24, 'actual global64 and policy24 scope')
    need(value['original_typed_shape_sha256'] == hashlib.sha256(before.encode()).hexdigest()
         and value['projected_typed_shape_sha256'] == hashlib.sha256(after.encode()).hexdigest(),
         'exact original typed gain and allocation source, one threshold change')
    need(value['original_gain_model_unchanged'] is True
         and value['original_gain_model_and_paid_role_limits_unchanged'] is False
         and value['legacy_paid_limits_preserved'] is True
         and value['new_exact_E4_paid_work_limit'] == dict(normal_rows=24, eager_rows=4,
                    normal_per_request=4, eager_per_request=4, total_candidates=112)
         and value['same_original_policy_object'] is True
         and value['Native_flags_assigned'] is False, 'actual callback ownership')
    state = value['normal_admission']
    need(state['available'] is True and state['generation'] == generation
         and state['rule'] == 'ROOT405.normal-class-priority-defer-age-owner-order.v1',
         'real current phase generation and new rule')
    histograms = [state[key] for key in ['raw_N_histogram', 'admitted_N_histogram', 'positive_grant_N_histogram']]
    need([len(hist) for hist in histograms] == [65, 25, 25]
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
             and row['admitted_N'] == len(admitted) <= 24
             and row['positive_grant_N'] == len(granted)
             and set(granted) <= set(admitted) <= set(raw)
             and all(type(b) is int and 1 <= b <= 5 for b in granted.values()), 'whole normal owners and grants')
        need(row['actual_work_normal_row_limit'] == 24 and row['actual_work_combined_row_limit'] == 64
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
        is_overflow = len(raw) > 24
        need(row['overflow'] is is_overflow, 'truthful policy overflow')
        if is_overflow:
            ranked = sorted(range(len(raw)), key=lambda j: (
                -priority[row['classes'][j]], -row['ages_before'][j], raw[j]))
            chosen = {raw[j] for j in ranked[:24]}
            need(admitted == [i for i in raw if i in chosen], 'actual original class-age-owner policy at24')
            overflow += 1
        else:
            need(admitted == raw, 'narrow raw owner set retained')
    need(calls == sorted(calls) and overflow == state['overflow_calls'], 'all overflow records retained')
    return dict(generation=generation, calls=state['call_count'], overflow_calls=overflow,
                retained_interesting_calls=len(calls), global_limit=64, normal_work_limit=24,
                every_deferred_owner_retained=True, exact_class_age_policy=True,
                neural_math_recomputed=False)
