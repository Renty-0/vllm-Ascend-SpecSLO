"""Real227 budget shapes, full real-node bijections and single-state callbacks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P = Path(__file__).parent
B = P.parent
sys.path[:0] = [str(P), str(B / 'tree-static-q5-M120-node-math-v255'),
    str(B / 'tree-canonical-contiguous-full463-own-AR-v235'),
    str(B / 'tree-contiguous-branch-tail-final-metadata-math-v234')]


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT377 HOST: ' + message)


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise RuntimeError('ROOT377 HOST invalid input accepted')


def ends(case):
    sizes = []
    for budget in case['budgets']:
        sizes.extend([5, 1] if budget == 5 else [budget + 1])
    sizes.extend(case['padding'])
    result = []
    for size in sizes:
        result.append((result[-1] if result else 0) + size)
    return result


def main():
    from geometry377 import choose, adjust, pad_function
    from physical377 import plan
    from oracle255 import cases
    import projection255, install377, branch_tail255
    source = cases()
    need(len(source) == 227, 'original complete227 real shape domain')
    count = {80: 0, 120: 0}
    families = set()
    real_queries = 0
    for case in source:
        candidate = adjust(case, enabled=True)
        original = adjust(case, enabled=False)
        need(original == case, 'disabled selection is exact best257 geometry')
        m = candidate['M']
        groups = case['N'] + case['budgets'].count(5)
        need(m == (80 if groups <= 16 and case['query_count'] <= 80 else 120),
             'independent complete real-group capacity selection')
        need(sum(candidate['padding']) + case['query_count'] == m
             and candidate['budgets'] is case['budgets'], 'all real budget objects and exact work accounting')
        selected = plan(ends(candidate), case['query_count'], m)
        full = projection255.plan(ends(case), case['query_count'])
        need(sorted(selected['physical_to_original']) == list(range(m))
             and selected['real_moves'] == full['real_moves']
             and selected['original_to_physical'][:case['query_count']]
                 == full['original_to_physical'][:case['query_count']],
             'all real inputs, positions, KV writer slots and ordering match original120')
        need(selected['supported_primary_side_groups'] == groups
             and selected['actual_seq_lengths_q'] == list(range(5, m + 1, 5)),
             'complete actual Q5 geometry and no missing side query')
        families.add((m, 11 if 5 in case['budgets'] else 10))
        count[m] += 1
        real_queries += case['query_count']
    need(families == {(80,10),(80,11),(120,10),(120,11)}, 'all four physical FULL graph families')
    rejects(lambda: choose([5]*13, enabled=True))
    rejects(lambda: choose([4]*25, enabled=True))
    rejects(lambda: choose([True], enabled=True))
    rejects(lambda: choose([4.0], enabled=True))
    rejects(lambda: choose([6], enabled=True))
    need(choose([5]*8, enabled=True) == 80 and choose([5]*9, enabled=True) == 120,
         'side groups, rather than owner count alone, control boundary')
    need(choose([4]*16, enabled=True) == 80 and choose([4]*17, enabled=True) == 120,
         'complete physical query capacity boundary')
    from oracle377 import cases as neural_cases, candidate_function, specialized_run
    import dis
    actual = neural_cases()
    need(len(actual)==231, 'all original227 plus4 max80 branch cases')
    maximum={(c['M'],11 if 5 in c['budgets'] else 10):c['query_count']for c in actual[:4]}
    need(maximum=={(80,10):80,(80,11):76,(120,10):120,(120,11):116},
         'largest actual real-row count first in every family')
    need(sum(i.opname=='LOAD_CONST'and i.argval=='M'for i in dis.get_instructions(candidate_function()))==3,
         'all three actual candidate physical dimension gates')
    need(specialized_run().__globals__['reference'].__module__=='oracle255',
         'independent reference stays original FULL120')

    # Exercise actual installer callback construction while replacing only the
    # allocation/attention-binding entry with a HOST double. No model executes.
    old = branch_tail255.install
    def host_install(engine, native):
        state = branch_tail255.State.__new__(branch_tail255.State)
        state.engine = engine
        state.enabled = False
        engine._branch_tail_state = state
        return dict(HOST_allocation_double=True)
    branch_tail255.install = host_install
    try:
        class Native:
            def target_tree_forward(*args, **kwargs):
                raise RuntimeError('HOST test must never execute a target')
        native = SimpleNamespace(NativePearlEngine=Native)
        runner = SimpleNamespace(run_target_greedy=lambda *args, **kwargs: None)
        engine = SimpleNamespace(graph_runner=runner)
        activation = install377.install(engine, native)
        state = engine._branch_tail_state
        binding = state.adaptive377_bindings
        need(state.prepare is binding['prepare'] and state.adaptive377_enabled is False,
             'single original state and opt-in selection')
        cells = dict(zip(state.prepare.__code__.co_freevars, state.prepare.__closure__, strict=True))
        need(cells['prepare'].cell_contents is binding['branch_prepare']
             and binding['branch_prepare'].__self__ is state
             and cells['transform'].cell_contents is binding['transform']
             and Native._pad_spec_rhythm_target_tree_graph is binding['pad'],
             'actual retained prepared closure, layout dispatch, transform and pad callbacks')
        need(binding['branch_prepare'].__func__.__code__ is branch_tail255.State.prepare.__code__,
             'entire actual branch State.prepare code object remains original')
        need(activation['NPU_qualified'] is False, 'HOST double never yields neural qualification')
    finally:
        branch_tail255.install = old
    need('torch' not in sys.modules and 'torch_npu' not in sys.modules, 'no neural or device import')
    result = dict(status='PASS_HOST377_REAL227_GEOMETRY_AND_SINGLE_STATE_CALLBACKS',
        NPU_used=False, neural_qualified=False, physical_family_case_counts=count,
        graph_families=[list(v) for v in sorted(families)], real_queries_checked=real_queries,
        original_real_permutations_and_budgets_preserved=True,
        original_State_prepare_code_preserved=True, allocation_HOST_double_only=True,
        Raw=None, Goodput=None, adopted=False)
    (P / 'HOST377_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
