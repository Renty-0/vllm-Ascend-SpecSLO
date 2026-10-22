"""Uninstalled HOST layout prototype for one-query repair/frontier PA fusion.

This describes rows only. It builds no tensor, reads no KV, runs no model, and
establishes no numerical qualification. The real caller must retain the Native
metadata, physical-slot, prefix/lease, matcher, finite, and publication gates.
"""


def _integer(value, name, upper=None):
    if type(value) is not int or value < 0 or upper is not None and value >= upper:
        raise ValueError('repair/frontier layout invalid ' + name)
    return value


def plan_single_query_frame(repairs, frontiers, *, next_bucket, maximum_rows=64):
    """Return a symbolic plan, or an explicit original-path fallback.

    Input tuples match FastV5: repair=(owner,start,tokens,current_root),
    frontier=(owner,position,parent_last_token). ``next_bucket`` is the original
    Native bucket function; logical owner IDs need not be physical row indices.
    """
    if type(maximum_rows) is not int or maximum_rows != 64:
        raise ValueError('prototype scope requires the actual64 work-row ceiling')
    repairs, frontiers = tuple(repairs), tuple(frontiers)
    if len(repairs) > 32 or len(frontiers) > 32 or len(repairs) + len(frontiers) > maximum_rows:
        raise ValueError('normal32/future32/combined64 row ceiling')
    owners = []
    for owner, position, tokens, root in repairs:
        owners.append(_integer(owner, 'owner'))
        _integer(position, 'position', 4096)
        _integer(root, 'root', 151936)
        if not isinstance(tokens, (tuple, list)) or not tokens:
            raise ValueError('repair tokens must be a nonempty HOST sequence')
        for token in tokens:
            _integer(token, 'repair token', 151936)
        if position + len(tokens) > 4096:
            raise ValueError('repair positions exceed the actual model limit')
    for owner, position, token in frontiers:
        owners.append(_integer(owner, 'owner'))
        _integer(position, 'position', 4096)
        _integer(token, 'frontier token', 151936)
    if len(owners) != len(set(owners)):
        raise ValueError('one-query PA requires disjoint unique owners')
    if not repairs or not frontiers:
        return dict(mode='original', reason='one group empty', production_installed=False)
    if any(len(tokens) != 1 for _, _, tokens, _ in repairs):
        return dict(mode='original', reason='multi-token repair keeps original ordered PA waves', production_installed=False)

    r, e = len(repairs), len(frontiers)
    frontier_physical = next_bucket(e, maximum_rows)
    old_repair_physical = next_bucket(r, maximum_rows)
    physical = next_bucket(frontier_physical + r, maximum_rows)
    if not all(type(value) is int for value in (frontier_physical, old_repair_physical, physical)):
        raise ValueError('original bucket returned a noninteger')
    if not e <= frontier_physical <= 32 or not r <= old_repair_physical <= 32 or not frontier_physical + r <= physical <= maximum_rows:
        raise ValueError('original bucket violates this scope')
    anchor = frontiers[0][0]

    def real(role, owner, position, token):
        return dict(role=role, owner=owner, position=position, token=token,
                    context_length=position + 1, slot_mapping='actual_original_owner_slot')

    def dummy(role):
        # Native _pad_inputs uses the first live row's page table, context1,
        # position0/token0, and slot-1. Put frontier first to preserve that row.
        return dict(role=role, owner=None, position=0, token=0, context_length=1,
                    slot_mapping=-1, block_table_anchor_owner=anchor)

    rows = [real('frontier', owner, position, token) for owner, position, token in frontiers]
    rows += [dummy('frontier_dummy') for _ in range(frontier_physical - e)]
    rows += [real('repair', owner, position, tokens[0]) for owner, position, tokens, _ in repairs]
    rows += [dummy('union_tail_dummy') for _ in range(physical - frontier_physical - r)]
    return dict(mode='candidate_one_query_PA', production_installed=False,
                normal_repair_rows=r, eager_frontier_rows=e, physical_M=physical,
                original_frontier_physical_M=frontier_physical,
                original_repair_physical_M=old_repair_physical,
                old_two_call_physical_queries=frontier_physical + old_repair_physical,
                new_one_call_physical_queries=physical, rows=rows,
                frontier_logical_rows=list(range(e)),
                frontier_original_head_rows=list(range(frontier_physical)),
                normal_repair_rows_in_union=list(range(frontier_physical, frontier_physical + r)),
                head_contract='Original compute_greedy_tokens(hidden[:frontier_physical_M]); use first E actual returned tokens',
                original_query_count=r + e, new_query_count=r + e,
                physical_slots_must_be_checked_on_actual_metadata=True,
                source_math_independence_is_not_numeric_equivalence=True)
