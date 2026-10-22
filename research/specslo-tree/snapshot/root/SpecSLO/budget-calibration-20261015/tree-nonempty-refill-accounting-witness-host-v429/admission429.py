"""HOST proposal: unchanged original clocks plus explicit refill witnesses.

This module has no installer, model constructor, or production permission.
The projected admission is for CPU checks only until a future same-model
incumbent-KV protocol and full own-AR experiment qualify it.
"""
import ast
import copy
import hashlib
import inspect
import math
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent / 'tree-nonempty-wholeprompt-refill-resource-host-v426'))
import refill426 as original


def need(value, message):
    if value is not True:
        raise RuntimeError('HOST429 accounting witness: ' + message)


def before(engine, active, admitted, states, local_states, now, params, controller):
    """Observe existing HOST values; no clock, tensor, or device access."""
    old, new = tuple(active), tuple(admitted)
    need(len(old) == len(set(old)) and len(new) == len(set(new))
         and not set(old).intersection(new) and len(old) + len(new) <= 64,
         'unique old/new owners within unchanged global64')
    if old:
        need(len(old) <= 32 and 8 <= len(new) <= 32,
             'declared low-water32/minimum8/maximum32 nonempty scope')
    need(all(params[i].arrival_ts is None or params[i].arrival_ts <= now for i in new),
         'no future arrival admitted')
    pages = 64 + sum((int(local_states[i].prompt_length)
                      + local_states[i].max_tokens + 5 + 127) // 128
                     for i in (*old, *new))
    need(pages <= 544, 'actual full-output and scratch reservations within544')
    return dict(old=old, new=new, old_decode_ms=[states[i].decode_elapsed_ms for i in old],
        old_tokens=[tuple(local_states[i].token_ids) for i in old],
        old_epochs=[states[i].prefix_epoch for i in old],
        ready=dict(controller.ready), total_reserved_pages=pages,
        arrival_clock=now, prefill_ms=None,
        old_home=[states[i].home_batch_id for i in old])


def charged(witness, states, local_states, controller, refill_ms):
    """Check the actual original elapsed broadcast and old-active charge."""
    need(type(refill_ms) is float and math.isfinite(refill_ms) and refill_ms >= 0.,
         'original finite prefill duration')
    old = witness['old']
    after = [states[i].decode_elapsed_ms for i in old]
    need(after == [v + refill_ms for v in witness['old_decode_ms']],
         'each original incumbent charged the exact original elapsed value')
    need([tuple(local_states[i].token_ids) for i in old] == witness['old_tokens']
         and [states[i].prefix_epoch for i in old] == witness['old_epochs']
         and [states[i].home_batch_id for i in old] == witness['old_home']
         and controller.ready == witness['ready'],
         'incumbent committed tokens, epochs, homes and original ready tickets retained')
    witness.update(prefill_ms=refill_ms, old_after_ms=after)


def finish(engine, witness, active, states, local_states, controller):
    need(witness['prefill_ms'] is not None,
         'whole prompt helper and original incumbent charge actually completed')
    old, new = witness['old'], witness['new']
    need(tuple(active) == (*old, *new), 'same original append/publication membership')
    need([states[i].delivered_tokens for i in new] == [1] * len(new)
         and all(len(local_states[i].committed_completion_token_ids) == 1 for i in new),
         'each new owner has exactly the genuine first token')
    need(controller.ready == witness['ready'], 'new admission leaves old ready tickets valid')
    row = dict(incumbents=list(old), admitted=list(new),
        prefill_ms=witness['prefill_ms'],
        incumbent_decode_before_ms=witness['old_decode_ms'],
        incumbent_decode_after_ms=witness['old_after_ms'],
        original_arrival_clock=witness['arrival_clock'],
        total_reserved_pages=witness['total_reserved_pages'],
        original_incumbent_tokens_epochs_homes_and_ready_tickets_unchanged=True,
        new_root_tokens=[local_states[i].committed_completion_token_ids[0] for i in new])
    engine._HOST429_refill_witnesses.append(row)


def source():
    """Three observations around exact426 admission; entire inverse checked."""
    before_source, candidate = original.sources()
    fn = ast.parse(candidate).body[0]
    clean = copy.deepcopy(fn)
    anchors = [i for i, n in enumerate(fn.body)
               if isinstance(n, ast.If) and ast.unparse(n.test) == 'needs_prefill']
    need(len(anchors) == 1, 'one original whole-prompt prefill block')
    block = fn.body[anchors[0]]
    begin = ast.parse('_witness429 = _before429(self, active, admitted, states, local_states, now, request_params, controller)').body[0]
    fn.body.insert(anchors[0], begin)
    charge_anchors = [i for i, n in enumerate(block.body)
                      if isinstance(n, ast.For) and ast.unparse(n.target) == 'index'
                      and ast.unparse(n.iter) == 'active']
    need(len(charge_anchors) == 1, 'one original incumbent charge loop')
    charge = ast.parse('_charged429(_witness429, states, local_states, controller, refill_ms)').body[0]
    block.body.insert(charge_anchors[0] + 1, charge)
    end = ast.parse('if admitted:\n _finish429(self, _witness429, active, states, local_states, controller)').body[0]
    need(isinstance(fn.body[-1], ast.Return), 'original terminal admission return')
    fn.body.insert(len(fn.body) - 1, end)
    projected = ast.unparse(ast.fix_missing_locations(fn)) + '\n'
    fn.body.remove(begin); fn.body.remove(end); block.body.remove(charge)
    need(ast.dump(fn, include_attributes=False) == ast.dump(clean, include_attributes=False),
         'entire426 admission body and clocks restored after three observations')
    return before_source, candidate, projected


def construct(namespace):
    """Explicit CPU test construction; no permission or runtime installer."""
    _, _, projected = source()
    ns = dict(namespace, _REFILL_LOW_WATER426=32, _REFILL_MIN426=8, _REFILL_MAX426=32,
              _before429=before, _charged429=charged, _finish429=finish)
    ns['self']._HOST429_refill_witnesses = []
    exec(compile(projected, __file__ + '::HOST-only-actual-admission', 'exec', dont_inherit=True), ns)
    return ns['admit_available']


def descriptor():
    old, candidate, projected = source()
    functions = [before, charged, finish]
    tree = ast.parse('\n\n'.join(inspect.getsource(f) for f in functions))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    need(not any('time.' in v or 'torch' in v or '_broadcast' in v for v in calls),
         'witness reads no clock, device, tensor or collective')
    return dict(original_425_function_sha256=hashlib.sha256(old.encode()).hexdigest(),
        resource_426_function_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        proposed_429_function_sha256=hashlib.sha256(projected.encode()).hexdigest(),
        entire_AST_inverse=True, original_clocks_and_publication_unchanged=True,
        existing_duration_broadcast_reused=True, new_clock_calls=0, new_collective_calls=0,
        global_active_limit=64, scratch_pages=64, total_pages=544,
        low_water=32, minimum_nonempty_refill=8, maximum_nonempty_refill=32,
        NPU_executed=False, NN_incumbent_KV_qualified=False,
        serving_installed=False, Raw=None, Goodput=None)
