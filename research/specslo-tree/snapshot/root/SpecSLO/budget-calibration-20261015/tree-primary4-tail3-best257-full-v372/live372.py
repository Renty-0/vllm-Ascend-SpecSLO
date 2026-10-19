"""Install only after fresh same-instance neural qualification; count real work."""


def install(engine, native, qualification):
    original = qualification['original']
    if native.NativePearlEngine.draft_tree_forward is not original or not engine.is_draft:
        raise RuntimeError('ROOT372 exact original draft instance/function before installation')
    state = qualification['state']
    proposed = qualification['proposed']
    counts = {}

    def forward(current, *args, **kwargs):
        if current is not engine:
            return original(current, *args, **kwargs)
        before = len(state.calls)
        plans = args[0] if args else kwargs['plans']
        # Adopt only the logical families checked in all306 fresh cases.
        # Larger original batches retain the original implementation and
        # original warm/seal obligations; no private key is created for them.
        result = (proposed if 1 <= len(plans) <= 32 else original)(current, *args, **kwargs)
        used = len(state.calls) - before
        if used not in (0, 1):
            raise RuntimeError('ROOT372 at most one primary tail graph per original expansion')
        generation = getattr(current, '_dynamic_tree_generation', None)
        if getattr(current, '_canonical257_phase', None) == 'qualified' and type(generation) is int:
            row = counts.setdefault(str(generation), dict(calls=0, tail_applied=0, fallback=0,
                larger_than32_fallback=0, max_logical_owners=0,
                original_frontier_graphs=0, actual_tail_graphs=0, actual_four_primary_steps=0,
                actual_graph_calls=0, actual_physical_queries=0, logical_owners=0))
            row['calls'] += 1
            row['tail_applied'] += used
            row['fallback'] += int(not used)
            row['larger_than32_fallback'] += int(len(plans) > 32)
            row['max_logical_owners'] = max(row['max_logical_owners'], len(plans))
            if result is not None:
                row['actual_graph_calls'] += result['graph_calls']
                row['actual_physical_queries'] += result['query_count']
                row['logical_owners'] += len(result['num_draft_tokens'])
                if used:
                    frontier = int(bool(result['eager_frontier_tokens']))
                    if result['graph_calls'] != 2 + frontier or state.calls[-1]['actual_model_steps'] != 3:
                        raise RuntimeError('ROOT372 original frontier/first plus actual three-step tail')
                    row['original_frontier_graphs'] += frontier
                    row['actual_tail_graphs'] += 1
                    row['actual_four_primary_steps'] += 4
        return result

    native.NativePearlEngine.draft_tree_forward = forward
    qualification.update(forward=forward, counts=counts,
        run_draft_func=engine.graph_runner.run_draft_greedy.__func__,
        execute_draft_func=engine.graph_runner._execute_draft.__func__)
    engine._ROOT372_qualified_draft_tail = qualification
    return qualification


def snapshot(engine):
    qualification = getattr(engine, '_ROOT372_qualified_draft_tail', None)
    if not engine.is_draft:
        return dict(is_draft=False, changed_target_model_or_sampler=False)
    if qualification is None:
        raise RuntimeError('ROOT372 fresh qualification and exact draft callback required')
    return dict(is_draft=True, proof=qualification['proof'],
                counts_by_generation=qualification['counts'],
                private_entry_count=len(qualification['entries']),
                original_four_dependent_steps_preserved=True,
                Native_qualification_flags_assigned=False,
                standalone_component_qualification_inherited=False)
