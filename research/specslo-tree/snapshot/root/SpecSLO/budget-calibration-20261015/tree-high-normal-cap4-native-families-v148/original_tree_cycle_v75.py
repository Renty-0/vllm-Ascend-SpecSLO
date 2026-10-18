"""Bounded normal-B5 diagnostic adapter around unchanged original tree commits.

This is an actor for a future independent two-cycle qualification, not an online
executor or qualification receipt.  Capture the original protocol BEFORE the
startup installer replaces the decoder with its fresh_decode closure.  Source
history is supplied by the independent-history actor, never copied from a
candidate KV cache.  All model/physical-KV checks belong to the caller.
"""
import hashlib
import inspect
import textwrap
from collections import Counter
from pathlib import Path


ORIGINAL_DECODE_SHA256 = 'fd53817d55fa73164fef345eba8615b4cd1ac790cdd509a82fa7d590dde6de8f'
CANONICAL_DECODE_SHA256 = 'ed44e73fd5ad72088aa36b41a8c94e6a1d4dccb15da909f26a060bab5a0997ae'
PROTOCOL_SOURCE_SHA256 = '74d1a745eb8841afefc2b5763a5802e1229b63e2285878daaa36c91e0a66ab49'


class CycleContractError(RuntimeError):
    pass


def capture_original_protocol_before_startup(native):
    """Compile original Native AST lanes; reject any already-installed wrapper."""
    import original_native_protocol_v75 as protocol
    source = textwrap.dedent(inspect.getsource(
        native.NativePearlEngine._generate_spec_rhythm_tree_decode))
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_DECODE_SHA256:
        raise CycleContractError('Capture requires the exact original Native decoder before startup')
    if hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest() != PROTOCOL_SOURCE_SHA256:
        raise CycleContractError('Original protocol AST extractor source changed')
    compiled = protocol.compile_protocol(native)
    # The protocol's AST uses the existing CPU transport helper.  Capturing
    # BEFORE startup means Native's global dictionary does not contain the
    # install-time _dt_cpu_reduce binding yet. Bind the exact original helper
    # into the independent compiled namespace, without mutating Native.
    import host_tree_transport
    observations = []
    original_reduce = host_tree_transport.reduce
    def observed_reduce(engine, value, **kwargs):
        result = original_reduce(engine, value, **kwargs)
        if kwargs.get('kind') == 'commit_consensus':
            values = result.cpu().tolist()
            if (len(values) != 2 or any(type(v) is not int or v < 0 for v in values)):
                raise CycleContractError('Original commit consensus returned an invalid status ABI')
            observations.append(tuple(values))
        return result
    namespace = compiled['preflight_consensus_compact'].__globals__
    if namespace.get('_dt_cpu_reduce') not in (None, original_reduce):
        raise CycleContractError('Unexpected pre-startup CPU consensus helper')
    namespace['_dt_cpu_reduce'] = observed_reduce
    compiled['actual_commit_consensus_observations'] = observations
    compiled['original_cpu_reduce'] = original_reduce
    _require_compiled(compiled)
    return compiled


def _require_compiled(compiled):
    if (not isinstance(compiled, dict)
            or compiled.get('original_decode_sha256') != ORIGINAL_DECODE_SHA256
            or compiled.get('canonical_decode_sha256') != CANONICAL_DECODE_SHA256
            or compiled.get('extracted_structural_and_numerical_lanes_unchanged') is not True
            or not isinstance(compiled.get('actual_commit_consensus_observations'), list)
            or any(not callable(compiled.get(k))
                   for k in ('preflight_consensus_compact', 'finish', 'cleanup'))):
        raise CycleContractError('Actual original pre-startup protocol extraction is required')


def _source_contract(source, plans, rows, confidences):
    ids = source.get('ids')
    contexts = source.get('contexts')
    if (not isinstance(ids, list) or not 1<=len(ids)<=32
            or ids != list(range(len(ids)))
            or not isinstance(contexts, list)
            or not len(contexts) == len(plans) == len(rows) == len(confidences) == len(ids)):
        raise CycleContractError('Bounded actor retains exactly 15 or 16 original owners')
    for context, plan, row, confidence in zip(contexts, plans, rows, confidences):
        tokens = context.get('token_ids')
        if (not isinstance(tokens, list) or not tokens
                or any(type(t) is not int or t < 0 for t in tokens)
                or type(context.get('output_offset')) is not int
                or not 0 <= context['output_offset'] < 246
                or type(getattr(plan,'candidate_budget',None)) is not int or not 1<=plan.candidate_budget<=5
                or plan.parent_indices.numel() != plan.candidate_budget
                or plan.prefix_len != len(tokens)-1
                or not 1 <= int(plan.depth) <= 4
                or len(row) != plan.candidate_budget or any(type(t) is not int or t < 0 for t in row)):
            raise CycleContractError('Actual normal B5 plans/rows must match the immutable source prefix')
        if confidence is not None:
            import math
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise CycleContractError('Original proposal confidence must be finite and in [0,1]')


def setup_normal_cycle(native, source, plans, rows, confidences, *, compiled_protocol):
    """Create original controllers once; later rounds must reuse this same state."""
    from vllm_ascend.spec_decode.pearl.spec_rhythm import (
        SpecRhythmRuntimeState, SpecRhythmPipelineController)
    _require_compiled(compiled_protocol)
    _source_contract(source, plans, rows, confidences)
    ids = list(source['ids'])
    states = {i: SpecRhythmRuntimeState(i, i % 2,
        slo_class=c['slo_class'], output_token_budget=256-c['output_offset'])
        for i, c in zip(ids, source['contexts'])}
    state = dict(controller=SpecRhythmPipelineController(states),
        logical={i: native.PearlPipelineState(list(c['token_ids']), len(c['token_ids'])-1,
            max_tokens=256-c['output_offset'], ignore_eos=True)
            for i, c in zip(ids, source['contexts'])},
        ids=ids, mappings={}, counters=Counter(), payloads={},
        compiled_protocol=compiled_protocol, last_committed_cycle_id=None,
        proposal_cycle_id=None, history=[], invalid_after_partial_failure=False)
    return state


def attach_normal_cycle_proposals(state, source, plans, rows, confidences, *, cycle_id):
    """Publish true next-history tickets using the original advanced controller."""
    _source_contract(source, plans, rows, confidences)
    last = state['last_committed_cycle_id']
    if (state['invalid_after_partial_failure'] or type(cycle_id) is not int or cycle_id < 0
            or (last is not None and cycle_id <= last)
            or state['proposal_cycle_id'] is not None
            or source['ids'] != state['ids']):
        raise CycleContractError('A new cycle requires an unused id and a valid consumed predecessor')
    for i, context in zip(state['ids'], source['contexts']):
        if context['token_ids'] != state['logical'][i].token_ids:
            raise CycleContractError('Next proposals must use the actual observed committed history')
    controller = state['controller']
    if controller.ready or controller.staged_eager or state['payloads'] or state['mappings']:
        raise CycleContractError('Original normal ticket/physical-map cleanup is incomplete')
    tickets = [controller.new_ticket(i, gamma=p.candidate_budget, eager=False) for i,p in zip(state['ids'],plans,strict=True)]
    controller.publish(tickets)
    payloads = {t.proposal_id: dict(ticket=t, plan=p, row=list(row), confidence=confidence,
        draft_kv_materialized=False) for t, p, row, confidence in
        zip(tickets, plans, rows, confidences)}
    state.update(tickets=tickets, payloads=payloads,
        target_payloads=[payloads[t.proposal_id] for t in tickets],
        proposal_cycle_id=cycle_id)


def run_original_tree_cycle(engine, native, source, plans, rows, target_output, state,
        *, cycle_id, target_home, stage_all4):
    """Real original verdict, finite/structural vote, compact, commit and finish.

    stage_all4(label, thunk) must be the actual four-rank Native diagnostic
    coordinator.  It is used BEFORE broadcast to make a leader-side shape error
    visible to all ranks.  No controlled verdict or candidate KV is accepted as
    an oracle.  A partial failure permanently invalidates this actor state.
    """
    def validate():
        _require_compiled(state['compiled_protocol'])
        _source_contract(source, plans, rows,
            [p['confidence'] for p in state['target_payloads']])
        if (state['invalid_after_partial_failure']
                or type(cycle_id) is not int or state['proposal_cycle_id'] != cycle_id
                or type(target_home) is not int or target_home not in (0, 1)):
            raise CycleContractError('Cycle must consume its actual current proposal exactly once')
        for i, context, plan, payload in zip(state['ids'], source['contexts'], plans,
                state['target_payloads']):
            if (context['token_ids'] != state['logical'][i].token_ids
                    or payload['plan'] is not plan or payload['row'] != rows[i]
                    or state['controller'].validate_verification(i) is not payload['ticket']):
                raise CycleContractError('Original current prefix/plan/row/ticket identity changed')
    stage_all4('V75-original-tree-cycle-preflight', validate)
    def verdict_local():
        if engine.rank != engine.topology.target_leader_rank:
            return None
        depth = max(int(p.depth) for p in plans)
        verdict = engine._verify_tree_outputs_host(rows, plans,
            target_output['target_query_token_ids'], depth, engine.device)
        if (tuple(verdict.token_ids.shape) != (len(plans), depth+1)
                or tuple(verdict.accepted_node_indices.shape) != (len(plans), depth)):
            raise CycleContractError('Original verdict rows must use actual packed depth, not node budget')
        return verdict
    verdict = stage_all4('V75-original-tree-host-verdict', verdict_local)
    outputs, accepted, _ = engine._broadcast_spec_rhythm_tree_verdict(verdict, plans, 0.)
    compiled = state['compiled_protocol']
    observations = compiled['actual_commit_consensus_observations']
    previous_observations = len(observations)
    epochs_before = {i: state['controller'].request_states[i].prefix_epoch for i in state['ids']}
    def commit_original():
        import torch
        compiled['preflight_consensus_compact'](engine, state['controller'],
            state['payloads'], state['mappings'], state['ids'], plans,
            state['target_payloads'], rows, target_output, state['logical'], outputs,
            accepted, state['counters'])
        # These are the ACTUAL original collective's returned integers. Reaching
        # a successful Python line is not a fabricated structural/numerical=0.
        if (len(observations) != previous_observations+1
                or observations[-1] != (0, 0)):
            raise CycleContractError('Missing or rejected actual original commit consensus result')
        torch.npu.synchronize()
        emitted = []
        for row_index, i in enumerate(state['ids']):
            values = [int(t) for t in outputs[row_index] if t >= 0]
            promoted = compiled['finish'](engine, state['controller'], i, plans[row_index],
                state['target_payloads'][row_index:row_index+1], state['payloads'],
                outputs[row_index:row_index+1], accepted[row_index:row_index+1],
                values, False, state['counters'])
            if promoted is not None:
                raise CycleContractError('Normal-only bounded actor cannot promote an eager child')
            state['logical'][i].apply_tree_verification(output_token_ids=values,
                accepted=sum(node >= 0 for node in accepted[row_index]), proposed_tokens=plans[row_index].candidate_budget)
            moves = compiled['cleanup'](engine, i, state['payloads'], state['mappings'],
                state['tickets'][row_index], None, promoted, state['counters'])
            if moves:
                raise CycleContractError('Deferred normal commits must not relocate draft KV')
            emitted.append(values)
        state['controller'].finish_cycle(target_home)
        torch.npu.synchronize()
        return emitted
    try:
        emitted = stage_all4('V75-original-tree-consensus-compact-finish', commit_original)
        def validate_committed():
            if state['controller'].ready or state['controller'].staged_eager or state['payloads'] or state['mappings']:
                raise CycleContractError('Actual original ticket and payload cleanup did not finish')
            for i in state['ids']:
                if state['controller'].request_states[i].prefix_epoch != epochs_before[i]+1:
                    raise CycleContractError('Actual original controller prefix epoch did not advance once')
            if state['controller'].next_target_home_batch_id != 1-target_home:
                raise CycleContractError('Original finish_cycle did not advance the home rotation')
        stage_all4('V75-original-tree-commit-epoch-confirmation', validate_committed)
    except BaseException:
        state['invalid_after_partial_failure'] = True
        raise
    state['last_committed_cycle_id'] = cycle_id
    state['proposal_cycle_id'] = None
    evidence = dict(cycle_id=cycle_id, target_home=target_home,
        committed_tokens=emitted, accepted_paths=[list(a) for a in accepted],
        actual_prefix_epochs={i: state['controller'].request_states[i].prefix_epoch for i in state['ids']},
        actual_logical_tokens=[list(state['logical'][i].token_ids) for i in state['ids']],
        original_decode_sha256=compiled['original_decode_sha256'],
        canonical_decode_sha256=compiled['canonical_decode_sha256'],
        extracted_structural_and_numerical_lanes_unchanged=True,
        original_finish_cycle_executed=True, counters=dict(state['counters']),
        actual_structural_failed=observations[-1][0],
        actual_numerical_failed=observations[-1][1],
        actual_preflight_error=None,
        independent_KV_qualification=False, NPU_qualification_receipt=False,
        formal_Goodput=False)
    state['history'].append(evidence)
    return evidence
