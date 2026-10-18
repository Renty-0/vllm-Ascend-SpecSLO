"""Observed normal-B5 draft plus independent physical-row numerical oracle.

All additional changed-token calls are LOCAL QUALIFICATION controls. They are
not a production optimization or a cost measurement. Original Native methods
alone set their validation flags. No candidate KV initializes the oracle.
"""
import copy
from contextlib import contextmanager


class DraftRejected(RuntimeError):
    pass


def _entry_state(entry):
    return {'runtime_validated': entry.runtime_validated,
            'validated_real_row_count': getattr(entry, 'validated_real_row_count', None)}


def _assert_execution(execution):
    if execution.mode not in ('capture_replay', 'replay') or execution.used_aclgraph is not True:
        raise DraftRejected('Original draft Native capture/replay gate rejected')


def _used_KV(engine, sequence_ids, starts, ends):
    from dual_bank_v146 import _arrays, _slots
    import torch
    index = torch.cat([_slots(engine, owner, range(start, end))
        for owner, start, end in zip(sequence_ids, starts, ends, strict=True)])
    return tuple(value.index_select(0, index).clone() for value in _arrays(engine))


def _metadata_record(meta, inputs, positions):
    return {'physical_rows': int(inputs.numel()), 'inputs': inputs.detach().clone(),
            'positions': positions.detach().clone(),
            'slots': meta.slot_mapping.detach().clone(),
            'contexts': meta.context_lens.detach().clone(),
            'actual_seq_lengths_q': tuple(meta.actual_seq_lengths_q),
            'sequence_lens': tuple(meta.sequence_lens),
            'use_fused_infer_attention': bool(meta.use_fused_infer_attention)}


def _assert_PA_padding(inputs, positions, metadata, real_rows, *, initial):
    import torch
    slots = metadata.slot_mapping
    # Native PA keeps lengths on CPU and positions on NPU. This untimed
    # observer compares host snapshots, without changing the model metadata.
    contexts = metadata.context_lens.detach().cpu()
    observed_positions = positions.detach().cpu()
    if (metadata.use_fused_infer_attention or not 0 < real_rows <= inputs.numel()
            or int(slots.numel()) != inputs.numel() or int(contexts.numel()) != inputs.numel()
            or bool((slots[:real_rows] < 0).any().item())
            or not bool((slots[real_rows:] == -1).all().item())
            or not bool((contexts[:real_rows] == observed_positions[:real_rows]+1).all().item())
            or not bool((contexts[real_rows:] == 1).all().item())
            or not bool((observed_positions[real_rows:] == 0).all().item())
            or (initial and not bool((inputs[real_rows:] == 0).all().item()))):
        raise DraftRejected('Actual PA real rows or fullphysical dummy slots/context/token contract differs')
    if real_rows < inputs.numel():
        expected = metadata.block_tables[:1].expand(inputs.numel()-real_rows, -1)
        if not torch.equal(metadata.block_tables[real_rows:], expected):
            raise DraftRejected('Original PA padding must read the first live owner table')


@contextmanager
def actual_draft_observer(engine, source, *, controls=True):
    """Clone immediately after return, before heads/other shared-pool users.

    The tail observer was installed before original startup captures. Here its
    saved capture buffers are cloned before any subsequent graph/reference.
    Controls preserve original arguments and never restore a ledger by hand.
    """
    import torch
    import root_seed
    import tail_hidden_observer
    from dual_bank_v146 import prefix_evidence, _arrays
    runner = engine.graph_runner
    old_hidden, old_tail = runner.run_tree_hidden, runner.run_draft_greedy
    old_head = root_seed.explicit_root_head
    old_overrides = {name: (name in runner.__dict__, runner.__dict__.get(name))
                     for name in ('run_tree_hidden', 'run_draft_greedy')}
    captured = {'first': None, 'tail': None, 'seed': None, 'controls': []}
    known = copy.deepcopy(engine._dynamic_tree_known_KV)
    histories = {}
    for owner, context in zip(source['ids'], source['contexts'], strict=True):
        tokens = context['token_ids']
        p = len(tokens)-1
        valid = known[owner]
        start = p
        for position in range(valid['safe_before'], p):
            if valid['tokens'].get(position) != tokens[position]:
                start = position
                break
        histories[owner] = tuple(tokens[:start])
    source_before = prefix_evidence(engine, histories)

    def query(meta):
        index = meta.slot_mapping.to(dtype=torch.long)
        index = index[index >= 0]
        return tuple(value.index_select(0, index).clone() for value in _arrays(engine))

    def hidden(inputs, positions, metadata):
        returned = old_hidden(inputs, positions, metadata)
        actual = returned.detach().clone()  # FIRST device operation after return.
        _assert_execution(runner.last_target_execution)
        real_rows = int((metadata.slot_mapping >= 0).sum().item())
        _assert_PA_padding(inputs, positions, metadata, real_rows, initial=True)
        if metadata.use_fused_infer_attention:
            raise DraftRejected('Normal root observer expected original PA metadata')
        key = ('target-hidden:0|steps:1|paged', (int(inputs.numel()),))
        entry = runner.target_entries.get(key)
        if entry is None:
            raise DraftRejected('Root PA did not create its authoritative Native hidden entry')
        before_state = _entry_state(entry)
        initial_query = query(metadata)
        original_meta = _metadata_record(metadata, inputs, positions)
        if controls:
            changed = inputs.clone()
            changed[0] = (changed[0]+1) % engine.draft_vocab_size
            old_hidden(changed, positions, metadata)
            _assert_execution(runner.last_target_execution)
            returned = old_hidden(inputs, positions, metadata)
            restored = returned.detach().clone()
            _assert_execution(runner.last_target_execution)
            if (not runner._target_outputs_match((actual,), (restored,))
                    or not runner._target_outputs_match(initial_query, query(metadata))):
                raise DraftRejected('Root PA changed-token control did not restore all hidden/query KV')
            captured['controls'].append({'kind': 'root-PA', 'key': repr(key),
                'before': before_state, 'after': _entry_state(entry),
                'changed_first_physical_token': True, 'actual_KV_restored': True})
        if entry.runtime_validated is not True or key in runner.disabled_entry_keys:
            raise DraftRejected('Original root PA Native changed-input gate remains open')
        captured['first'] = {'hidden': actual, 'metadata': original_meta,
            'query_KV_before_control': initial_query, 'native_key': repr(key),
            'native_entry': _entry_state(entry)}
        return returned

    def head(engine_arg, value, roots, rows):
        encoded, inputs = old_head(engine_arg, value, roots, rows)
        captured['seed'] = encoded.detach().clone()
        return encoded, inputs

    def tail(inputs, positions, metadata, vocabulary_size, **kwargs):
        returned = old_tail(inputs, positions, metadata, vocabulary_size, **kwargs)
        actual = returned.detach().clone()
        _assert_execution(runner.last_draft_execution)
        key = runner.last_draft_entry_key
        entry = runner.draft_entries.get(key)
        n = len(source['ids'])
        if entry is None or kwargs.get('valid_row_count') != n:
            raise DraftRejected('Normal tail actual Native key/real-row ABI differs')
        for pos, meta in zip(positions, metadata, strict=True):
            _assert_PA_padding(inputs, pos, meta, n, initial=False)
        actual_hidden, intervals, eager_intervals = tail_hidden_observer.snapshot(runner, key, len(positions))
        initial_queries = tuple(query(meta) for meta in metadata)
        before_state = _entry_state(entry)
        # A three-step tail must not touch its already-evaluated current root.
        # A four-step chain legitimately evaluates the root itself.
        first_positions = positions[0].detach().cpu().tolist()[:n]
        protected = {owner: tuple(context['token_ids'][:p])
            for owner, context, p in zip(source['ids'], source['contexts'], first_positions, strict=True)}
        protected_before = prefix_evidence(engine, protected)
        if controls:
            changed = inputs.clone()
            changed[0] = (changed[0]+1) % vocabulary_size
            old_tail(changed, positions, metadata, vocabulary_size, **kwargs)
            _assert_execution(runner.last_draft_execution)
            returned = old_tail(inputs, positions, metadata, vocabulary_size, **kwargs)
            restored = returned.detach().clone()
            _assert_execution(runner.last_draft_execution)
            if (not torch.equal(actual, restored)
                    or any(not runner._target_outputs_match(a, query(meta))
                           for a, meta in zip(initial_queries, metadata, strict=True))):
                raise DraftRejected('Tail changed-token control did not restore full encoded/query KV')
            restored_hidden, _, _ = tail_hidden_observer.snapshot(runner, key, len(positions))
            if not runner._target_outputs_match(actual_hidden, restored_hidden):
                raise DraftRejected('Tail changed-token control did not restore every physical hidden row')
            captured['controls'].append({'kind': 'tail', 'key': repr(key),
                'before': before_state, 'after': _entry_state(entry),
                'changed_first_physical_token': True, 'actual_KV_restored': True})
        if (entry.runtime_validated is not True or entry.validated_real_row_count < n
                or key in runner.disabled_entry_keys or prefix_evidence(engine, protected) != protected_before):
            raise DraftRejected('Original tail Native changed-input/real-row/protected-prefix gate rejected')
        captured['tail'] = {'encoded': actual, 'hidden': actual_hidden,
            'intervals': intervals, 'eager_intervals': eager_intervals,
            'native_key': repr(key), 'native_entry': _entry_state(entry),
            'valid_row_count': kwargs['valid_row_count'],
            'query_KV_before_control': initial_queries}
        return returned
    runner.run_tree_hidden, runner.run_draft_greedy = hidden, tail
    root_seed.explicit_root_head = head
    try:
        yield captured
        if prefix_evidence(engine, histories) != source_before:
            raise DraftRejected('Normal draft changed an independently valid committed source prefix')
    finally:
        root_seed.explicit_root_head = old_head
        for name, (existed, value) in old_overrides.items():
            if existed:
                setattr(runner, name, value)
            else:
                runner.__dict__.pop(name, None)


