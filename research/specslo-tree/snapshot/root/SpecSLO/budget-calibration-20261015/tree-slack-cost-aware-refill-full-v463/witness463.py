"""Add decision evidence to the original same-model refill witness."""

import witness431 as original

charged = original.charged


def before(engine, active, admitted, states, local_states, now, params, controller):
    witness = original.before(engine, active, admitted, states, local_states, now, params, controller)
    if not admitted:
        return witness
    decision = getattr(engine, '_root463_pending_admission', None)
    if (decision is None or decision['generation'] != witness['source_generation431']
            or decision['active'] != list(witness['old'])
            or decision['selected'] != list(witness['new'])
            or decision['arrival_clock'] != witness['arrival_clock']):
        raise RuntimeError('Same actual admission decision and original clock required')
    for row, owner, elapsed in zip(decision['incumbents'], witness['old'], witness['old_decode_ms'], strict=True):
        if (row['owner'] != owner or row['decode_elapsed_ms'] != elapsed
                or row['delivered_tokens'] != states[owner].delivered_tokens
                or row['delivered_tokens'] != len(local_states[owner].committed_completion_token_ids)):
            raise RuntimeError('Credit must use the actual committed count and original elapsed value')
    witness['ROOT463_admission'] = decision
    return witness


def finish(engine, witness, active, states, local_states, controller):
    original.finish(engine, witness, active, states, local_states, controller)
    generation = witness['source_generation431']
    row = engine._root431_refill_rows[generation][-1]
    if row['admitted'] != witness['ROOT463_admission']['selected']:
        raise RuntimeError('Original committed refill membership required')
    row['ROOT463_admission'] = witness['ROOT463_admission']
    engine._root463_pending_admission = None
