"""Independently reconstruct both complete physical target geometries."""


def query_ends_wide():
    total = 0
    result = []
    for value in [4] * 24 + [3] * 8:
        total += value
        result.append(total)
    return result


def valid(record):
    from bucket409 import valid_record
    budgets = record.get('budgets')
    n = record.get('N')
    if (type(n) is not int or not 1 <= n <= 32 or not isinstance(budgets, list)
            or len(budgets) != n or any(type(b) is not int or not 1 <= b <= 5 for b in budgets)
            or record['C'] != sum(budgets) or record['M'] != 120
            or record['logical_queries'] != n + sum(budgets)
            or record['branches'] != budgets.count(5)
            or type(record['actual_copy_slots']) is not int
            or not 0 <= record['actual_copy_slots'] <= 1792
            or not valid_record(record)):
        return False
    if n > 24:
        return (all(b <= 3 for b in budgets) and budgets.count(3) <= 24
                and record['C'] <= 88 and record['branches'] == 0
                and record['copied'] is False and record['actual_copy_slots'] == 0
                and record['mask_capacity'] == 10 and record['attention_segments'] == 32
                and record['primary_side_groups'] == n
                and record['actual_q_partition'] == query_ends_wide())
    branches = budgets.count(5)
    return (record['C'] <= 96 and n + branches <= 24
            and record['primary_side_groups'] == n + branches
            and record['copied'] is bool(branches)
            and (branches > 0 or record['actual_copy_slots'] == 0)
            and record['mask_capacity'] == (11 if branches else 10)
            and record['attention_segments'] == 24
            and record['actual_q_partition'] == list(range(5, 121, 5)))


def audit_pair(record, call):
    if not valid(record):
        raise RuntimeError('ROOT409 actual target geometry or paid resources invalid')
    if (record['budgets'] != call['budgets'] or record['N'] != call['N']
            or record['M'] != call['physical_M']
            or record['logical_queries'] != call['real_query_count']):
        raise RuntimeError('ROOT409 actual physical witness differs from real target call')
    ends = []
    total = 0
    for i, count in enumerate(call['physical_query_partition']):
        for size in ([5, 1] if i < call['N'] and call['budgets'][i] == 5 else [count]):
            total += size
            ends.append(total)
    if call['N'] > 24:
        from projection397 import plan
    else:
        from projection255 import plan
    expected = plan(ends, call['real_query_count'])
    if (record['physical_to_original'] != expected['physical_to_original']
            or record['original_to_physical'] != expected['original_to_physical']):
        raise RuntimeError('ROOT409 complete physical permutation differs from independent plan')
    from bucket409 import extent
    wanted = extent(tuple(context + min(budget, 4) + 1
                          for context, budget in zip(call['contexts'], call['budgets'], strict=True)))
    if record['fixed_KV_extent'] != [wanted] * record['attention_segments']:
        raise RuntimeError('ROOT409 visible extent differs from independently retained real contexts')
    return True
