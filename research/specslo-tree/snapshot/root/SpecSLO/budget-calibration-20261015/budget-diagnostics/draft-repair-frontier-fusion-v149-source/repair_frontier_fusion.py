"""Uninstalled repair/frontier PA layout prototype. No Torch import or installer.

The caller supplies the original Native bucket/padding/metadata operations.
This module cannot establish numerical equivalence, live device-table ownership,
or permission to use an entry. A future endpoint must supply its real lease guard
and qualify every new model M independently before timing or adoption.
"""
from dataclasses import dataclass, fields, replace


METADATA_FIELDS = (
    'slot_mapping', 'context_lens', 'block_tables', 'actual_seq_lengths_q',
    'sequence_lens', 'request_block_tables', 'attention_mask',
    'use_fused_infer_attention', 'tree_attention', 'tree_attention_mask',
)


def require(value, message):
    if value is not True:
        raise ValueError(message)


def integer(value, name, lower=0):
    require(type(value) is int and value >= lower, 'invalid ' + name)
    return value


@dataclass(frozen=True)
class AllocationWitness:
    allocation_id: int
    block_size: int
    block_tables: tuple[tuple[int, ...], ...]


def witness_from_engine(engine):
    """Snapshot actual HOST allocation tables; this performs no device read.

    Caller must also check the real live lease, device-table consistency, source
    prefixes and write ownership. Allocation identity alone does not prove them.
    """
    allocation = engine.cache_allocation
    require(allocation is not None, 'missing original cache allocation')
    return AllocationWitness(id(allocation), engine.config.kvcache_block_size,
                             tuple(tuple(row) for row in allocation.block_tables))


@dataclass(frozen=True)
class FramePlan:
    repairs: tuple
    frontiers: tuple
    frontier_M: int
    old_repair_M: int
    union_M: int
    witness: AllocationWitness
    write_slots: tuple[int, ...]  # frontier real rows first, then repair real rows

    @property
    def E(self):
        return len(self.frontiers)

    @property
    def R(self):
        return len(self.repairs)


def plan_frame(repairs, frontiers, *, next_bucket, witness, maximum_rows=64):
    """Return a checked plan, or None for the unchanged empty/multi-token path.

    Owner IDs are logical, not row indices bounded by 64. This deliberately
    supports only independently leased pages across the participating owners;
    shared pages require another proof and are rejected rather than adopted.
    """
    require(maximum_rows == 64 and type(maximum_rows) is int, 'scope requires64 work rows')
    repairs, frontiers = tuple(repairs), tuple(frontiers)
    require(len(repairs) <= 32 and len(frontiers) <= 32, 'normal32/frontier32 ceiling')
    for owner, position, tokens, root in repairs:
        integer(owner, 'repair owner'); integer(position, 'repair position')
        integer(root, 'repair root')
        require(type(tokens) in (list, tuple) and bool(tokens), 'empty repair')
        for token in tokens:
            integer(token, 'repair token')
    for owner, position, token in frontiers:
        integer(owner, 'frontier owner'); integer(position, 'frontier position')
        integer(token, 'frontier token')
    owners = tuple(x[0] for x in frontiers) + tuple(x[0] for x in repairs)
    require(len(set(owners)) == len(owners), 'duplicate or intersecting owners')
    if not repairs or not frontiers or any(len(x[2]) != 1 for x in repairs):
        return None
    require(type(witness) is AllocationWitness, 'actual HOST allocation witness required')
    integer(witness.allocation_id, 'allocation identity', 1)
    block_size = integer(witness.block_size, 'block size', 1)
    positions = tuple(x[1] for x in frontiers) + tuple(x[1] for x in repairs)
    slots, pages = [], []
    for owner, position in zip(owners, positions, strict=True):
        require(owner < len(witness.block_tables), 'owner outside actual storage table')
        table = witness.block_tables[owner]
        require(type(table) is tuple and position // block_size < len(table), 'position outside actual table')
        # Every context page, not merely the write page, must actually exist.
        for page in table[:position // block_size + 1]:
            integer(page, 'allocated context page')
        for page in table:
            require(type(page) is int and page >= -1, 'invalid page sentinel')
            if page >= 0:
                pages.append(page)
        slots.append(table[position // block_size] * block_size + position % block_size)
    require(len(pages) == len(set(pages)), 'participating owners have aliased leased pages')
    require(len(slots) == len(set(slots)), 'physical write slots alias')
    F = next_bucket(len(frontiers), maximum_rows)
    old_R = next_bucket(len(repairs), maximum_rows)
    U = next_bucket(F + len(repairs), maximum_rows)
    require(all(type(x) is int for x in (F, old_R, U)), 'bucket must return integers')
    require(len(frontiers) <= F <= 32 and len(repairs) <= old_R <= 32
            and F + len(repairs) <= U <= 64, 'original bucket outside bounded scope')
    return FramePlan(tuple((o, p, tuple(t), r) for o, p, t, r in repairs),
                     frontiers, F, old_R, U, witness, tuple(slots))


def _pa_metadata(metadata, *, rows, logical_rows, positions):
    require(tuple(f.name for f in fields(metadata)) == METADATA_FIELDS, 'Native metadata schema changed')
    require(metadata.use_fused_infer_attention is False and metadata.tree_attention is False,
            'only original one-query PA metadata is eligible')
    require(all(getattr(metadata, k) is None for k in
                ('request_block_tables', 'attention_mask', 'tree_attention_mask')), 'unexpected PA mask/table')
    require(metadata.slot_mapping.shape == (rows,) and metadata.context_lens.shape == (rows,),
            'PA slot/context row mismatch')
    require(metadata.block_tables.ndim == 2 and metadata.block_tables.shape[0] == rows, 'PA table row mismatch')
    require(metadata.actual_seq_lengths_q == tuple(range(1, logical_rows + 1)), 'noncanonical PA q tuple')
    require(metadata.sequence_lens == tuple(p + 1 for p in positions) + (1,) * (rows - logical_rows),
            'original PA host context lengths changed')


def build_union(plan, *, tensor, prepare, pad, cat, lease):
    """Use original operations; pad frontier to F, append R, then pad to U.

    ``lease`` must run the future endpoint's actual lease guard and return its
    current AllocationWitness, both before and after metadata construction.
    The callback must never be a constant success or a copied old receipt.
    """
    require(type(plan) is FramePlan, 'eligible FramePlan required')
    require(lease() == plan.witness, 'stale lease before metadata')
    fids = [x[0] for x in plan.frontiers]; fpos = [x[1] for x in plan.frontiers]
    rids = [x[0] for x in plan.repairs]; rpos = [x[1] for x in plan.repairs]
    fi = tensor([x[2] for x in plan.frontiers])
    fp, fm = prepare(fids, fpos, False)
    _pa_metadata(fm, rows=plan.E, logical_rows=plan.E, positions=fpos)
    fi, fp, fm = pad(fi, fp, fm, plan.frontier_M)
    _pa_metadata(fm, rows=plan.frontier_M, logical_rows=plan.E, positions=fpos)
    ri = tensor([x[2][0] for x in plan.repairs])
    rp, rm = prepare(rids, rpos, False)
    _pa_metadata(rm, rows=plan.R, logical_rows=plan.R, positions=rpos)
    require(fm.block_tables.shape[1:] == rm.block_tables.shape[1:], 'page table widths differ')
    # Original pad preserves q=1..E while extending sequence_lens to F. The
    # existing F-E finite read-only dummy rows are now interior rows, followed
    # by real repairs. Explicitly represent all F+R rows as one-query requests;
    # concatenating q with a gap would violate Native PA host-task alignment.
    merged = replace(fm,
        slot_mapping=cat((fm.slot_mapping, rm.slot_mapping), dim=0),
        context_lens=cat((fm.context_lens, rm.context_lens), dim=0),
        block_tables=cat((fm.block_tables, rm.block_tables), dim=0),
        actual_seq_lengths_q=tuple(range(1, plan.frontier_M + plan.R + 1)),
        sequence_lens=fm.sequence_lens + rm.sequence_lens)
    inputs = cat((fi, ri), dim=0); positions = cat((fp, rp), dim=0)
    inputs, positions, merged = pad(inputs, positions, merged, plan.union_M)
    require(inputs.shape == positions.shape == (plan.union_M,), 'union input row mismatch')
    require(merged.sequence_lens == fm.sequence_lens + rm.sequence_lens
            + (1,) * (plan.union_M - plan.frontier_M - plan.R), 'union context/padding changed')
    require(lease() == plan.witness, 'stale lease after metadata')
    return inputs, positions, merged


def run_union(plan, packed, *, runner, model, vocabulary_size, lease):
    """Single hidden invocation plus original head on exactly F original rows.

    This function intentionally mutates no ledger, ticket, or Native flag. The
    caller retains original success/finite gates before publishing any result.
    Hidden output is graph-owned scratch; endpoint evidence must clone it before
    the next invocation if it needs to retain it.
    """
    require(runner.model is model, 'union model and production head model differ')
    integer(vocabulary_size, 'vocabulary size', 1)
    require(lease() == plan.witness, 'stale lease before union execution')
    hidden = runner.run_tree_hidden(*packed)
    require(hidden.ndim == 2 and hidden.shape[0] == plan.union_M and hidden.shape[1] > 0,
            'Native hidden must be [U,hidden_size], not a tuple or tokens')
    tokens = model.compute_greedy_tokens(hidden[:plan.frontier_M], vocabulary_size)
    require(tokens.shape == (plan.frontier_M,), 'original greedy head must return [F]')
    require(lease() == plan.witness, 'stale lease after union execution')
    return tokens[:plan.E]
