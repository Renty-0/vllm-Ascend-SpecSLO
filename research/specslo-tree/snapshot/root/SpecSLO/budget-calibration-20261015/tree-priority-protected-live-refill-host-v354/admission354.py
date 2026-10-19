"""HOST candidate: postpone refills while protected SLO owners are decoding."""
import ast
import copy
import hashlib

from admission345 import patch as coalesced_refill


MODES = ('closed', 'protect_tight_normal', 'protect_tight')


def allowed_classes(mode):
    if mode == 'closed':
        return ()
    if mode == 'protect_tight_normal':
        return ('loose',)
    if mode == 'protect_tight':
        return ('normal', 'loose')
    raise ValueError('undeclared admission354 mode')


def patch(source, mode, minimum=4):
    allowed = allowed_classes(mode)
    base = coalesced_refill(source, minimum)
    tree = ast.parse(base)
    before = ast.dump(tree, include_attributes=False)
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == 'admit_available']
    if len(functions) != 1:
        raise RuntimeError('one actual original admit_available')
    function = functions[0]
    first = function.body[0]
    if not (isinstance(first, ast.Assign) and ast.unparse(first.targets[0]) == 'closed'
            and 'receive_live_admissions' in ast.unparse(first.value)):
        raise RuntimeError('receive original admissions before the scheduling guard')
    guard = ast.parse(f'''
if active and not all(states[index].slo_class in {allowed!r} for index in active):
    counters['spec_rhythm_ROOT354_priority_guard_polls'] = counters.get('spec_rhythm_ROOT354_priority_guard_polls', 0) + 1
    return closed
''').body[0]
    function.body.insert(1, guard)
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    function.body.remove(guard)
    if ast.dump(tree, include_attributes=False) != before:
        raise RuntimeError('whole coalesced-refill source AST inverse')
    # coalesced_refill separately checks its complete inverse to the supplied
    # original source. No time, model, callback or publication edits are made.
    return result


def descriptor(source, mode, minimum=4):
    projected = patch(source, mode, minimum)
    return dict(mode=mode, allowed_incumbent_classes=list(allowed_classes(mode)),
                minimum_refill=minimum, active_limit=48, private_pages=64,
                page_pool=544, original_global_budget=192,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                projected_sha256=hashlib.sha256(projected.encode()).hexdigest(),
                guard_after_original_receive=True,
                original_arrival_expressions_preserved=True,
                arrival_clock_poll_frequency_unchanged=False,
                root_publication_not_delayed_after_computation=True,
                incumbent_refill_time_charged_by_original_body=True,
                model_or_tensor_operations_added=False,
                guard_counter_counts_polls_not_prevented_prefills=True,
                NPU_qualified=False, full463_qualified=False, Goodput=None)
