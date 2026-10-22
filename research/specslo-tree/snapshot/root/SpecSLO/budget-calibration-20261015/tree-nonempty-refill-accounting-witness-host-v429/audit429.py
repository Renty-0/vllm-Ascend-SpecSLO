"""Proposed replacement for the empty-only admission trace predicate.

This adds checks for actual timing witnesses; it does not audit NN values or
grant serving permission. The original whole-prompt, Native, output, own-AR,
and all same-model component audits remain separately required.
"""
import math
from admission429 import need


def audit_trace(trace, witnesses):
    groups = [v['sequence_ids'] for v in trace['standalone_prefill_calls']]
    need(len(groups) == len(witnesses), 'one actual original prefill per witness')
    active, seen, part, group_index, cohorts = [], set(), [], 0, []
    for event in trace['stream_events']:
        i = event['request_index']
        if i not in seen:
            if not part:
                need(group_index < len(groups), 'every first token belongs to a genuine helper')
                row = witnesses[group_index]
                owners = groups[group_index]
                need(row['incumbents'] == active and row['admitted'] == owners
                     and len(owners) == len(set(owners)) and not set(owners).intersection(seen)
                     and len(active) + len(owners) <= 64,
                     'independently reconstructed active owners, helper order and global64')
                if active:
                    need(len(active) <= 32 and 8 <= len(owners) <= 32,
                         'actual nonempty low-water32/min8/max32')
                else:
                    need(0 < len(owners) <= 64, 'original empty refill capacity')
                now, elapsed = row['original_arrival_clock'], row['prefill_ms']
                need(type(now) is float and math.isfinite(now)
                     and type(elapsed) is float and math.isfinite(elapsed) and elapsed >= 0.,
                     'genuine finite existing arrival and prefill clocks')
                arrivals = trace['absolute_arrivals']
                need(all(arrivals[j] <= now for j in owners), 'every request has arrived')
                pages = 64 + sum((len(trace['tokenized_inputs'][j]) + 256 + 5 + 127) // 128
                                 for j in (*active, *owners))
                need(row['total_reserved_pages'] == pages <= 544,
                     'independent unchanged total full-output and scratch reservation')
                before, after = row['incumbent_decode_before_ms'], row['incumbent_decode_after_ms']
                need(len(before) == len(after) == len(active)
                     and all(type(v) is float and math.isfinite(v) and v >= 0. for v in (*before, *after))
                     and after == [v + elapsed for v in before],
                     'all actual incumbents charged the full original prefill duration exactly once')
                need(row['original_incumbent_tokens_epochs_homes_and_ready_tickets_unchanged'] is True
                     and len(row['new_root_tokens']) == len(owners),
                     'actual old state and complete new roots retained')
                cohorts.append(dict(ordinal=group_index, owners=list(owners),
                    incumbent_count=len(active), reserved_pages=pages, prefill_ms=elapsed))
            owner_offset = len(part)
            need(i == groups[group_index][owner_offset]
                 and event['token_ids'] == [witnesses[group_index]['new_root_tokens'][owner_offset]],
                 'unchanged original immediate single root callback in helper order')
            seen.add(i); active.append(i); part.append(i)
            if len(part) == len(groups[group_index]):
                part = []; group_index += 1
        else:
            need(not part, 'no incumbent callback interrupts the original first-token group')
        if event['finished']:
            need(i in active, 'finished owner actually active')
            active.remove(i)
    need(not active and not part and group_index == len(groups)
         and seen == set(range(trace['scope_count'])),
         'all original finite requests fully drained without a drop')
    return dict(cohorts=cohorts, cohort_count=len(cohorts), all_arrivals_retained=True,
        nonempty_cohorts=sum(v['incumbent_count'] > 0 for v in cohorts),
        source_clocks_unchanged=True, exact_original_incumbent_prefill_charge=True,
        no_extra_root_hold_or_callback=True, no_per_prefill_duration_inference=True,
        NN_qualification_inferred=False, serving_permission=False)
