"""Retained streaming evidence detects unsafe or incomplete live refills."""
import copy
import json
from pathlib import Path

import runtime346
from empty346 import audit_trace


def main():
    trace = dict(scope_count=6, requests=[dict(max_tokens=256) for _ in range(6)],
        tokenized_inputs=[[0]*128 for _ in range(6)], absolute_arrivals=[90.]*6,
        standalone_prefill_calls=[dict(sequence_ids=[0,1]), dict(sequence_ids=[2,3,4,5])],
        stream_events=[dict(request_index=i, finished=False, first_token_wall_seconds160=100.+i)
                       for i in range(6)]+[dict(request_index=i, finished=True) for i in range(6)])
    value = audit_trace(trace)
    assert value['actual_inflight_refill_groups'] == 1 and value['actual_peak'] == 6
    assert value['cohorts'][1]['incumbent_owners'] == [0,1]
    assert value['cohorts'][1]['conservative_reserved_pages'] == 88
    assert value['no_active_refill'] is False
    def duplicate(v): v['standalone_prefill_calls'][1]['sequence_ids'][0] = 0
    def reorder(v): v['stream_events'][2]['request_index'] = 3
    def early(v): v['stream_events'][0]['first_token_wall_seconds160'] = 89.
    def missing(v): v['stream_events'].pop()
    def interleave(v): v['stream_events'].insert(3, dict(request_index=0, finished=False))
    def pages(v): v['tokenized_inputs'][0] = [0]*(128*545)
    for mutate in (duplicate,reorder,early,missing,interleave,pages):
        bad = copy.deepcopy(trace)
        mutate(bad)
        try: audit_trace(bad)
        except RuntimeError: pass
        else: raise AssertionError('corrupt refill trace accepted: '+mutate.__name__)
    result = dict(status='passed_live_refill_auditor_HOST', NPU_used=False,
        fabricated_trace_only=True, actual_incumbent_tracking=True,
        duplicate_reordered_early_missing_interleaved_and_overcommitted_rejected=6,
        no_model_qualification=True)
    (Path(__file__).resolve().parent/'HOST_REFILL_AUDIT346_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
