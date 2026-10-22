"""CPU execution of the actual admission body and accounting negatives."""
import ast
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
from dataclasses import replace


def main():
    import torch
    import torch_npu
    def forbidden(*a, **k):
        raise AssertionError('HOST429 cannot construct a Native model or initialize an NPU')
    torch.npu._lazy_init = forbidden
    torch_npu.npu._lazy_init = forbidden
    from vllm_ascend.spec_decode.pearl import native_engine as native, spec_rhythm as rhythm
    native.NativePearlEngine.__init__ = forbidden
    import admission429 as candidate
    inherited = candidate.P.parent / 'tree-nonempty-wholeprompt-refill-resource-host-v426/check426_HOST.py'
    tree = ast.parse(inherited.read_text())
    inherited_main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    nodes = [copy.deepcopy(n) for n in inherited_main.body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ('Clock', 'build')]
    assert len(nodes) == 2
    ns = dict(torch=torch, native=native, rhythm=rhythm, SimpleNamespace=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(inherited) + '::unchanged-explicit-model-IO-doubles', 'exec'), ns)
    build = ns['build']
    rng = random.Random(429)
    positive = deferred = empty = repeated_without_duplicate = 0
    negatives = []
    for trial in range(360):
        old_count = 0 if trial % 6 == 0 else rng.randrange(1, 65)
        lengths = [rng.choice((32, 127, 255, 511, 1023, 1800)) for _ in range(rng.randrange(0, 65))]
        future = {old_count + i for i in range(len(lengths)) if rng.random() < .2}
        namespace, view = build(old_count, lengths, future)
        namespace['controller'] = view['controller']
        function = candidate.construct(namespace)
        assert function() is False
        witnesses = namespace['self']._HOST429_refill_witnesses
        admitted = [i for i, _ in view['publications']]
        assert len(witnesses) == int(bool(admitted))
        if admitted:
            row = witnesses[0]
            assert row['incumbents'] == list(range(old_count))
            assert row['admitted'] == admitted
            assert row['new_root_tokens'] == [100+i for i in admitted]
            assert row['total_reserved_pages'] <= 544
            assert row['prefill_ms'] > 0.
            assert row['incumbent_decode_after_ms'] == [x + row['prefill_ms'] for x in row['incumbent_decode_before_ms']]
            assert set(view['activation']) == set(admitted)
            assert view['prefills'] == [admitted]
            assert len(view['publications']) == len(set(i for i, _ in view['publications']))
            positive += bool(old_count)
            empty += not old_count
            if old_count:
                assert 8 <= len(admitted) <= 32
                assert old_count <= 32
        else:
            deferred += bool(old_count)
        first_published = list(view['publications'])
        assert function(poll_live=False) is False
        second = [i for i, _ in view['publications'][len(first_published):]]
        assert not set(second).intersection(i for i, _ in first_published)
        assert len(view['active']) <= 64
        repeated_without_duplicate += 1

    # Controlled failures prove that records cannot silently forgive missing
    # incumbent latency, stale tickets, or admitting future requests.
    for kind in ('missing_charge', 'double_charge', 'stale_ticket', 'old_token_write', 'future_arrival'):
        namespace, view = build(8, [63]*8, set())
        namespace['controller'] = view['controller']
        engine = namespace['self']
        engine._HOST429_refill_witnesses = []
        old = view['active'][:]
        new = view['pending'][:]
        if kind == 'future_arrival':
            view['params'][new[0]] = native.NativeSamplingParams(temperature=0., max_tokens=256,
                ignore_eos=True, arrival_ts=1001., request_id=str(new[0]))
        try:
            witness = candidate.before(engine, old, new, view['states'], view['target'],
                                       1000., view['params'], view['controller'])
            if kind != 'missing_charge':
                for i in old:
                    view['states'][i].add_decode_time(60. if kind == 'double_charge' else 30.)
            if kind == 'stale_ticket':
                view['controller'].invalidate_request(old[0])
            elif kind == 'old_token_write':
                view['target'][old[0]].token_ids.append(999)
            candidate.charged(witness, view['states'], view['target'], view['controller'], 30.)
        except RuntimeError as error:
            negatives.append(dict(kind=kind, rejected=True, message=str(error)))
        else:
            raise AssertionError('HOST429 negative was incorrectly accepted: ' + kind)
    # Build two real CPU admission groups using the actual function. Only the
    # completion events are synthetic auditor controls; there is no NN or
    # serving result and no historical timing reconstruction.
    namespace, view = build(0, [63]*16, set(range(8, 16)))
    namespace['controller'] = view['controller']
    function = candidate.construct(namespace)
    function()
    for i in range(8, 16):
        view['params'][i] = replace(view['params'][i], arrival_ts=999.)
    function()
    witnesses = namespace['self']._HOST429_refill_witnesses
    assert len(witnesses) == 2 and not witnesses[0]['incumbents'] and len(witnesses[1]['incumbents']) == 8
    trace = dict(scope_count=16,
        tokenized_inputs=[list(range(63)) for _ in range(16)], absolute_arrivals=[999.]*16,
        standalone_prefill_calls=[dict(sequence_ids=row['admitted']) for row in witnesses],
        stream_events=[dict(request_index=i, token_ids=tokens, finished=False) for i,tokens in view['publications']]
            + [dict(request_index=i, token_ids=[7], finished=True) for i in view['active']])
    import audit429
    trace_audit = audit429.audit_trace(trace, witnesses)
    assert trace_audit['nonempty_cohorts'] == 1
    trace_negatives = []
    for kind in ('missing_old_charge', 'wrong_old_membership', 'wrong_root', 'future_arrival'):
        bad = copy.deepcopy(witnesses)
        if kind == 'missing_old_charge':
            bad[1]['incumbent_decode_after_ms'][0] = bad[1]['incumbent_decode_before_ms'][0]
        elif kind == 'wrong_old_membership':
            bad[1]['incumbents'][0] = 999
        elif kind == 'wrong_root':
            bad[1]['new_root_tokens'][0] += 1
        else:
            bad[1]['original_arrival_clock'] = 998.
        try:
            audit429.audit_trace(trace, bad)
        except RuntimeError as error:
            trace_negatives.append(dict(kind=kind, rejected=True, error=str(error)))
        else:
            raise AssertionError('HOST429 raw witness negative incorrectly accepted: '+kind)
    assert positive and deferred and empty and len(negatives) == 5 and len(trace_negatives) == 4
    result = dict(status='PASS_HOST429_ACTUAL_CPU_REFILL_WITNESSES_AND_NEGATIVES',
        actual_admission_CPU_cases=360, nonempty_positive_cases=positive,
        nonempty_deferred_cases=deferred, empty_positive_cases=empty,
        repeated_admission_without_duplicate_publication=repeated_without_duplicate,
        controlled_negatives=negatives, source=candidate.descriptor(),
        actual_CPU_two_admission_groups_with_synthetic_final_drain=trace_audit,
        controlled_trace_negatives=trace_negatives,
        original_CPU_states_controller_and_admission_priority=True,
        original_426_model_IO_clock_broadcast_doubles_explicit=True,
        NPU_used=False, Native_models_constructed=0,
        NN_incumbent_KV_qualified=False, serving_installed=False,
        Raw=None, Goodput=None)
    path = candidate.P / 'HOST429_RESULT.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
