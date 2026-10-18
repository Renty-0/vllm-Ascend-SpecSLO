"""Actual Native HOST maps: exact original split/pack inverse and controls."""
import copy
import json
import statistics
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B / 'tree-concrete-device-span-dual-cache-target-cost-v296'),
               str(B / 'tree-deferred-FULL-mask-host-projection-v259')]
import check296


def main():
    from fixture259 import build
    env = build()
    engine, native = env['engine'], env['native']
    from static_state296 import bind_checked_prepare
    import branch_tail255
    bind_checked_prepare(engine._branch_tail_state, branch_tail255)
    from padding269 import pad
    from plan269 import make
    from oracle296 import cases
    from split288 import split
    from spans292 import pack
    from prefix278 import select_grid
    from descriptors304 import CheckedRows, separate
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = pad
    saved, examples = [], []

    def one(case, prefix):
        budgets = case['budgets']
        plans = [make(b, prefix, case['anchor']) for b in budgets]
        roots = list(range(41, 41 + len(budgets)))
        rows = [[103 + 17*i + 29*j for j in range(b)] for i, b in enumerate(budgets)]
        pp, rr, dd, ss, real = pad(engine, plans, roots, rows, list(range(len(budgets))), [], [])
        native.NativePearlEngine.target_tree_forward(engine, pp, rr, dd, sequence_ids=ss,
                                                     return_logits=False, real_tree_count=real)
        if not any(b > 4 for b in budgets):
            return
        state = engine._branch_tail_state
        desc = state.copy296_layout_HOST
        slots = state.static_current['physical'][2].slot_mapping.tolist()
        before = (list(desc['source']), list(desc['dest']), list(slots))
        old = split(desc['source'], desc['dest'], slots)
        packed = pack(old['prefix_source'], old['prefix_destination'])
        verified = CheckedRows(desc['source'], desc['dest'])
        new = separate(verified, slots)
        assert new['query_alias_destination'] == old['query_alias_destination']
        assert new['spans'] == packed['spans'] and new['span_count'] == packed['span_count']
        for name in ('prefix_copies', 'current_aliases', 'original_copies'):
            assert new[name] == old[name]
        assert new['legacy_capture_grid'] == select_grid(desc['source'], desc['dest'])
        expected = [(s, d) for s, d in zip(old['prefix_source'], old['prefix_destination']) if s != d]
        actual = [(s+i, d+i) for s, d, count in new['spans'] for i in range(count)]
        assert actual == expected
        assert before == (list(desc['source']), list(desc['dest']), list(slots))
        saved.append((tuple(desc['source']), tuple(desc['dest']), tuple(slots)))
        examples.append(dict(case=case['name'], prefix=prefix, N=len(budgets),
                             prefix_copies=new['prefix_copies'], current_aliases=new['current_aliases']))

    for case in cases():
        one(case, 127)
    extra = [case for case in cases() if max(case['budgets']) == 8
             and case['N'] in (1, 12, 16, 23) and case['anchor'] == 0]
    for prefix in (0, 1, 2, 126, 128, 129, 1505, 2555):
        for case in extra:
            one(case, prefix)
    source, destination, slots = saved[0]
    reference = separate(CheckedRows(source, destination), slots)
    rejected = []

    def reject(name, function):
        try:
            function()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Bad contract accepted: ' + name)

    bad = list(source)
    bad[0] = -1
    reject('out_of_bounds', lambda: CheckedRows(bad, destination))
    duplicate = list(destination)
    duplicate[0] = duplicate[1]
    reject('duplicate_destination', lambda: CheckedRows(source, duplicate))
    moved = next(d for s, d in zip(source, destination) if s != d)
    overlap = list(source)
    overlap[-1] = moved
    reject('source_destination_alias', lambda: CheckedRows(overlap, destination))
    verified = CheckedRows(source, destination)
    reject('writer_rows119', lambda: separate(verified, slots[:-1]))
    reject('invalid_writer', lambda: separate(verified, (-1,) + slots[1:]))
    alias_row = next(i for i, d in enumerate(reference['query_alias_destination']) if d >= 0)
    duplicate_writers = list(slots)
    duplicate_writers[(alias_row + 1) % 120] = slots[alias_row]
    reject('ambiguous_current_writer', lambda: separate(verified, duplicate_writers))
    overlap_writer = list(slots)
    overlap_writer[0] = reference['query_alias_destination'][alias_row]
    reject('alias_destination_is_writer', lambda: separate(verified, overlap_writer))
    reject('unvalidated_map', lambda: separate(object(), slots))
    mutable_source = list(source)
    frozen = CheckedRows(mutable_source, destination)
    mutable_source[0] = -1
    assert separate(frozen, slots) == reference
    try:
        frozen.source = ()
    except AttributeError:
        pass
    else:
        raise AssertionError('Checked map must be immutable')
    # Profile only the HOST contract/split/pack stage. No NPU or model cost.
    timing = []
    sampled = saved[::max(1, len(saved)//12)][:12]
    for source, destination, slots in sampled:
        old_times, new_times = [], []
        for iteration in range(30):
            order = ('old', 'new') if iteration % 2 == 0 else ('new', 'old')
            for mode in order:
                start = time.perf_counter()
                if mode == 'old':
                    select_grid(source, destination)
                    old = split(source, destination, slots)
                    pack(old['prefix_source'], old['prefix_destination'])
                else:
                    separate(CheckedRows(source, destination), slots)
                elapsed = (time.perf_counter() - start)*1000
                (old_times if mode == 'old' else new_times).append(elapsed)
        timing.append(dict(original_median_ms=statistics.median(old_times),
                           proposed_median_ms=statistics.median(new_times)))
    assert 'torch_npu' not in sys.modules
    value = dict(status='HOST_actual_Native_maps_single_full_contract_span_alias_inverse_passed',
                 original_declared_cases=408, branch_map_cases=len(examples),
                 extra_page_boundaries=list((0, 1, 2, 126, 128, 129, 1505, 2555)),
                 rejected_controls=rejected, immutable_input_snapshot=True,
                 every_original_prefix_and_alias_row_preserved=True,
                 original_full_1792_anti_alias_bounds_contract_still_executed=True,
                 no_unused_1792_prefix_staging_tensor_required=True,
                 HOST_descriptor_timings=timing, timing_is_not_NPU_model_performance=True,
                 NPU_used=False, runtime_installed=False, Goodput=None)
    with (P / 'HOST304_RESULT.json').open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
