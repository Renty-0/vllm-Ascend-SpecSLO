"""Proposed nonempty admission: original timing witnesses plus same-model proof.

There is no installer or launcher. Permission cannot come from the HOST429
results: it requires actual original427 NN evidence and an issued425 scope in
the same real engine. Neither has been supplied to this proposal.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(B/'tree-nonempty-refill-accounting-witness-host-v429'))
from admission429 import need, before as original_before, charged

_ISSUER = object()


@dataclass(frozen=True)
class Permission:
    engine: object
    model: object
    inherited: object
    proof: object
    issuer: object


def issue(engine, inherited):
    from scope425 import Scope
    need(type(inherited) is Scope and inherited.require(engine) == frozenset(),
         'same actual full425 original scope before new admission permission')
    proof = getattr(engine, '_root427_incumbent_prefill_actual', None)
    need(type(proof) is dict and proof['same_model_instance_id'] == id(engine.model)
         and proof['cache_released'] is True and proof['actual_entry_objects_preserved'] is True
         and proof['audit']['passed'] is True
         and proof['audit']['all_old_allocated_KV_pages_and_draft_ledger_protected'] is True
         and proof['audit']['original_new_prefix_reference_root_next_and_twozero_gates_retained'] is True
         and [(row['pass_index'], row['N']) for row in proof['cases']]
             == [(p, n) for p in range(3) for n in (2, 32)]
         and not hasattr(engine, '_root431_nonempty_permission'),
         'all fresh actual427 NN cases, repeated zero passes, old32 pages and draft ledger')
    value = Permission(engine, engine.model, inherited, proof, _ISSUER)
    engine._root431_nonempty_permission = value
    engine._root431_refill_rows = {}
    return value


def current(engine):
    value = getattr(engine, '_root431_nonempty_permission', None)
    need(type(value) is Permission and value.issuer is _ISSUER
         and value.engine is engine and value.model is engine.model
         and value.proof is getattr(engine, '_root427_incumbent_prefill_actual', None),
         'explicit same-model current NN permission; HOST prototype never authorizes refill')
    return value


def before(engine, *args):
    current(engine)
    generation = getattr(engine, '_dynamic_tree_generation', None)
    need(type(generation) is int and generation > 0, 'original actual source generation')
    rows = engine._root431_refill_rows.setdefault(generation, [])
    engine._root431_refill_current_rows = rows
    value = original_before(engine, *args)
    value['source_generation431'] = generation
    return value


def original_finish():
    from projection425 import project_function
    from admission429 import finish
    return project_function(finish, (('_HOST429_refill_witnesses', '_root431_refill_current_rows'),))


_FINISH = None


def finish(engine, witness, *args):
    global _FINISH
    current(engine)
    generation = witness['source_generation431']
    need(engine._dynamic_tree_generation == generation
         and engine._root431_refill_current_rows is engine._root431_refill_rows[generation],
         'same original generation and exact witness storage')
    if _FINISH is None:
        _FINISH = original_finish()
    return _FINISH(engine, witness, *args)


def snapshot(engine, generation):
    current(engine)
    rows = engine._root431_refill_rows.get(generation)
    need(type(rows) is list and rows, 'actual admissions retained for this generation')
    return list(rows)
