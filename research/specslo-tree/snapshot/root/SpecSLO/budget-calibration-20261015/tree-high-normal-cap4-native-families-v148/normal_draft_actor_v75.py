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
    from independent_history_v75 import _arrays, _slots
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
    from independent_history_v75 import prefix_evidence, _arrays
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
        if controls and not entry.runtime_validated:
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
        if controls and (not entry.runtime_validated or entry.validated_real_row_count < n):
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


def generate_and_qualify(engine, native, epoch, *, stage_all4):
    """Original model/selector/transport with independent real-history reference."""
    import torch
    from vllm_ascend.spec_decode.pearl.tree_budget import TreeCandidateRequest
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan, pack_selected_tree_plan
    from independent_history_v75 import recompute_normal_draft, compare_normal_draft
    from backbone_selection import select
    def prepare_source():
        from independent_history_v75 import normal_draft_count_contract
        normal_draft_count_contract(epoch)
        contexts = epoch.get('logical_contexts', epoch['immutable_source'].tree_tokens)
        source = {'ids': list(epoch['candidate_tree_ids']), 'contexts': [
            dict(row, token_ids=list(values), output_offset=row['output_offset']+len(values)-len(original))
            for row, values, original in zip(epoch['fixture']['trees'], contexts,
                epoch['immutable_source'].tree_tokens, strict=True)]}
        plans = [cached_cpu_tree_speculation_plan(2, 4, len(values)-1, 4096, candidate_budget=5)
                 for values in contexts]
        return contexts,source,plans
    contexts,source,plans=stage_all4('V119-normal-draft-source-and-plan-contract',prepare_source)
    def candidate():
        if not engine.is_draft:
            return None
        engine._defer_tree_materialization = engine._defer_normal_tree_materialization = True
        before_ledger = copy.deepcopy(engine._dynamic_tree_known_KV)
        starts = []
        for owner, values in zip(source['ids'], contexts, strict=True):
            known = before_ledger[owner]
            start = len(values)-1
            for p in range(known['safe_before'], len(values)-1):
                if known['tokens'].get(p) != values[p]:
                    start = p
                    break
            starts.append(start)
        native_before = dict(engine.graph_metrics())
        with actual_draft_observer(engine, source) as capture:
            output = engine.draft_tree_forward(plans, [v[-1] for v in contexts], source['ids'],
                committed_catchup_token_ids=[list(values[-5:]) for values in contexts])
        if capture['tail'] is None:
            raise DraftRejected('Actual normal draft did not use the captured original tail')
        root_mode = output.get('execution_mode') == 'normal-packed-root-seed-top2-v1'
        tail = capture['tail']['hidden']
        if root_mode:
            if capture['first'] is None or capture['seed'] is None or len(tail) != 3:
                raise DraftRejected('Root repair lacks immediate fullphysical first/seed/tail snapshots')
            first = capture['first']['hidden']
            encoded = torch.cat((capture['seed'][:, None, :], capture['tail']['encoded']), dim=1)
        else:
            if capture['first'] is not None or len(tail) != 4:
                raise DraftRejected('Four-step normal chain has an unexpected root repair snapshot')
            first, tail = tail[0], tail[1:]
            encoded = capture['tail']['encoded']
        queries = _used_KV(engine, source['ids'], starts, [len(values)+3 for values in contexts])
        return {'output': output, 'first': first, 'tail': tail, 'physical_encoded': encoded,
                'query_KV': queries, 'ledger': copy.deepcopy(engine._dynamic_tree_known_KV),
                'capture': capture, 'root_mode': root_mode, 'starts': starts,
                'native_metrics_before': native_before, 'native_metrics_after': dict(engine.graph_metrics())}
    actual = stage_all4('V75-observed-original-normal-draft', candidate)
    reference = recompute_normal_draft(engine, native, epoch, stage_all4=stage_all4)
    compared = compare_normal_draft(engine, epoch, None if actual is None else actual['output'], reference,
        actual_first_hidden=None if actual is None else actual['first'],
        actual_tail_hidden=None if actual is None else actual['tail'],
        actual_physical_encoded=None if actual is None else actual['physical_encoded'],
        actual_query_KV=None if actual is None else actual['query_KV'],
        actual_ledger=None if actual is None else actual['ledger'], stage_all4=stage_all4)
    selected_budgets = epoch['scope'].budgets
    def choose():
        if not engine.is_draft:
            return None
        output = actual['output']
        # The original packet averages decoded millionth confidences before
        # its final FP32 tensor conversion; do not average already-rounded
        # node tensors and accidentally change that exact public contract.
        expected_mean = torch.tensor([sum(scores)/8 for scores in reference['scores']],
                                     dtype=output['draft_confidence'].dtype)
        if (output['num_draft_tokens'] != [5]*len(plans) or output['eager_frontier_tokens'] != {}
                or not torch.equal(output['draft_confidence'].cpu(), expected_mean)):
            raise DraftRejected('Original normal proposal mean8/B5/eager-frontier public ABI differs')
        requests = {i: TreeCandidateRequest(parents=plan.parent_indices.tolist(),
            token_ids=row, conditional_confidences=scores.cpu().tolist(), max_candidates=budget,
            minimum_candidates=budget) for i, plan, row, scores, budget in zip(source['ids'], plans,
                output['draft_token_ids'], output['node_confidences'], selected_budgets, strict=True)}
        refined = select(requests, 96)
        selected = [list(refined.selected_indices[i]) for i in source['ids']]
        rows = [[row[j] for j in indices] for row, indices in zip(output['draft_token_ids'], selected, strict=True)]
        confidence = [sum(float(scores[j]) for j in indices)/len(indices)
            for scores, indices in zip(output['node_confidences'], selected, strict=True)]
        return rows, selected, confidence
    chosen = stage_all4('V75-original-confidence-backbone-selection', choose)
    rows, selections, confidences = chosen if chosen is not None else (None, None, None)
    message = engine._exchange_spec_rhythm_tree_candidates(rows, plans,
        [None]*len(plans) if engine.is_draft else None, confidences,
        selected_indices=selections, draft_compute_ms=0.)
    def split():
        if engine.is_draft:
            values = rows, selections, confidences
        else:
            candidate_rows, frontier, confidence, selected, _ = engine._split_tree_candidates(message, plans)
            if any(value is not None for value in frontier):
                raise DraftRejected('Normal-only diagnostic cannot have eager frontiers')
            values = candidate_rows, selected, confidence
        packed_rows, selected, confidence = values
        if (len(packed_rows) != len(plans) or any(len(row) != budget for row,budget in zip(packed_rows,selected_budgets,strict=True))
                or any(not len(indices) == len(set(indices)) == budget for indices,budget in zip(selected,selected_budgets,strict=True))):
            raise DraftRejected('Original normal proposal exchange changed B5 rows/selection')
        return [pack_selected_tree_plan(p, indices) for p, indices in zip(plans, selected, strict=True)], packed_rows, confidence, selected
    packed, packed_rows, confidence, selected = stage_all4('V75-original-proposal-transport-split', split)
    return {'source': source, 'plans': packed, 'rows': packed_rows, 'confidences': confidence,
        'selected_indices': selected, 'draft_numeric': compared,
        'actual_snapshot': actual, 'reference_snapshot': reference}


def probe_published_prompt_next_draft(engine, native, epoch, context, *, stage_all4):
    """Actual first normal draft of the newly published prompt, with fresh ref.

    The independent prompt owner retains the two real reference chunks. Its
    next root is the target's independently agreed first token, never candidate
    hidden or KV. This is an additional local numerical probe, not a new round
    of online target verification or a performance measurement.
    """
    import torch
    from types import SimpleNamespace
    from chain_metadata import build
    from independent_history_v75 import prefix_evidence
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    owner, reference_owner = epoch['prompt_id'], epoch['reference_prompt_id']
    logical = context['draft_states'][owner]
    tokens = list(logical.token_ids)
    p = len(tokens)-1
    def preflight():
        if (p != 2301 or epoch['independent_prompt_cursor'] != 2301
                or tuple(epoch['reference_prefixes'][len(epoch['candidate_tree_ids'])]) != tuple(tokens[:-1])
                or logical.committed_completion_token_ids != [tokens[-1]]
                or owner not in context['active'] or owner in context['pending_admission']):
            raise DraftRejected('New prompt lacks genuine independent final history and one published root')
        if engine.is_draft and engine._dynamic_tree_known_KV.get(owner) != dict(safe_before=p, tokens={}):
            raise DraftRejected('Published prompt inherited a stale or invented draft known-KV claim')
    stage_all4('V75-published-prompt-next-draft-history', preflight)
    def other_prefixes():
        contexts = epoch['logical_contexts']
        result = {}
        for ids, ledger in ((epoch['candidate_tree_ids'], engine._dynamic_tree_known_KV),
                            (epoch['reference_tree_ids'], epoch['draft_reference_ledger'])):
            for index, values in zip(ids, contexts, strict=True):
                end = len(values)-1
                if engine.is_draft:
                    known = ledger[index]
                    if not 0 <= known['safe_before'] <= end:
                        raise DraftRejected('Other normal owner has an invalid draft known-KV frontier')
                    for position in range(known['safe_before'], end):
                        if known['tokens'].get(position) != values[position]:
                            end = position
                            break
                result[index] = tuple(values[:end])
        return result
    protected_histories = stage_all4('V75-new-prompt-other-owner-valid-prefix-descriptors', other_prefixes)
    protected_before = stage_all4('V75-new-prompt-other-owner-prefix-before',
        lambda: prefix_evidence(engine, protected_histories))
    plan = cached_cpu_tree_speculation_plan(2, 4, p, 4096, candidate_budget=5)
    def actual_forward():
        if not engine.is_draft:
            return None
        source = {'ids': [owner], 'contexts': [{'token_ids': tokens}]}
        before = prefix_evidence(engine, {owner: tokens[:-1]})
        with actual_draft_observer(engine, source) as captured:
            packet = engine.draft_tree_forward([plan], [tokens[-1]], [owner],
                committed_catchup_token_ids=[tokens[-5:]])
        if captured['first'] is not None or len(captured['tail']['hidden']) != 4:
            raise DraftRejected('A fully materialized prompt unexpectedly required committed-suffix repair')
        query = _used_KV(engine, [owner], [p], [p+4])
        if before != prefix_evidence(engine, {owner: tokens[:-1]}):
            raise DraftRejected('First normal draft overwrote its published prompt prefix')
        return packet, captured, query
    actual = stage_all4('V75-real-published-prompt-normal-draft', actual_forward)
    def reference_forward():
        if not engine.is_draft:
            return None
        before = prefix_evidence(engine, {reference_owner: tokens[:-1]})
        engine._ensure_cache_capacity([reference_owner]*4, list(range(p, p+4)))
        head = native._next_linear_draft_graph_bucket(1, engine.config.max_num_seqs)
        inputs, positions, metadata = build(engine, [SimpleNamespace(prefix_len=p)],
            [reference_owner], [tokens[-1]], 4, head)
        hidden, encoded = [], []
        for pos, meta in zip(positions, metadata, strict=True):
            value = engine.model(inputs, pos, meta).clone()
            hidden.append(value)
            probability = torch.softmax(engine.model.compute_logits(value)[:, :engine.draft_vocab_size].float(), dim=-1)
            scores, choices = probability.topk(2, dim=-1)
            encoded.append(choices.long()*1_000_001+torch.round(scores.clamp(0, 1)*1_000_000).long())
            inputs = choices[:, 0]
        if before != prefix_evidence(engine, {reference_owner: tokens[:-1]}):
            raise DraftRejected('Independent next draft changed its retained genuine prompt history')
        return tuple(hidden), torch.stack(encoded, dim=1), _used_KV(engine, [reference_owner], [p], [p+4])
    reference = stage_all4('V75-independent-published-prompt-next-draft', reference_forward)
    stage_all4('V75-published-prompt-next-original-finite-vote',
               engine._vote_spec_rhythm_prefill_finiteness)
    def compare():
        if prefix_evidence(engine, protected_histories) != protected_before:
            raise DraftRejected('New prompt next query changed another normal owner valid prefix')
        allocations = []
        for index, table in enumerate(engine.cache_allocation.block_tables):
            pages = tuple(page for page in table if page >= 0)
            if len(set(pages)) != len(pages) or any(set(pages) & set(old) for _, old in allocations):
                raise DraftRejected('Published prompt expansion aliased another live actual/reference owner')
            if pages:
                allocations.append((index, pages))
        if not engine.is_draft:
            return {'draft_model_executed': False, 'scope': 'target join only'}
        packet, capture, actual_KV = actual
        hidden, encoded, independent_KV = reference
        matcher = engine.graph_runner._target_outputs_match
        if (any(not bool(torch.isfinite(value).all().item()) for value in
                (*capture['tail']['hidden'], *hidden, *actual_KV, *independent_KV))
                or not matcher(capture['tail']['hidden'], hidden)
                or not matcher(actual_KV, independent_KV)
                or not torch.equal(capture['tail']['encoded'], encoded)):
            raise DraftRejected('Published prompt next draft differs in fullphysical hidden/encoded/all28KV')
        pairs = encoded[0].cpu().tolist()
        row = [v[0]//1_000_001 for v in pairs]+[v[1]//1_000_001 for v in pairs]
        scores = [v[0]%1_000_001/1_000_000 for v in pairs]+[v[1]%1_000_001/1_000_000 for v in pairs]
        expected_ledger = dict(safe_before=p, tokens={p+j: value
            for j, value in enumerate((tokens[-1], *row[:3]))})
        if (packet['draft_token_ids'] != [row] or packet['root_token_ids'] != [tokens[-1]]
                or packet['num_draft_tokens'] != [5] or packet['eager_frontier_tokens'] != {}
                or not torch.equal(packet['node_confidences'][0].cpu(), torch.tensor(scores))
                or not torch.equal(packet['draft_confidence'].cpu(), torch.tensor([sum(scores)/8]))
                or engine._dynamic_tree_known_KV[owner] != expected_ledger):
            raise DraftRejected('Published prompt next public rows/confidence/known-KV ledger differs')
        return {'draft_model_executed': True, 'actual_owner': owner,
            'independent_owner': reference_owner, 'prefix_len': p,
            'all_physical_hidden_original_matcher': True, 'all28_query_KV_original_matcher': True,
            'all_new_query_hidden_and_KV_finite': True, 'original_finite_WORLD_vote_completed': True,
            'exact_top2_confidence_and_public_rows': True, 'exact_materialized_ledger': True,
            'actual_tail_key': capture['tail']['native_key'],
            'actual_tail_entry': capture['tail']['native_entry'],
            'actual_tail_real_row_count': capture['tail']['valid_row_count'],
            'actual_physical_head_rows': int(capture['tail']['encoded'].shape[0]),
            'actual_cold_changed_controls': capture['controls'],
            'local_scope': 'newly published N1/B5/P2301 normal four-step chain',
            'other_normal_valid_prefixes_preserved': True,
            'actual_reference_and_other_live_pages_disjoint_after_expansion': True,
            'page_expansion_after_publication_is_not_an_atomic_lease_change': True,
            'candidate_KV_used_to_initialize_reference': False,
            'new_target_verification_claimed': False, 'online_qualified': False}
    return stage_all4('V75-published-prompt-next-draft-original-gates', compare)
