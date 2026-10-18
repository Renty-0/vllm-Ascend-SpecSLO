"""Reviewable V63 AST integration, deliberately not installed or qualified.

Calling build_blocked_runtime is host-only source construction. Installation
and the generated function require the separate root-source-bound LOCAL
bounded diagnostic permit. Formal/online launch is unconditionally blocked.
"""
import ast
import hashlib
from pathlib import Path
from types import FunctionType

from launch_guard import require_online_qualification
from diagnostic_authorization import require_bounded_scope


REFERENCE_SHA256 = '89303eadf7753d9a0482a24d6fa0613c482b5a50a8c979a3bc1aec5ef3e4095c'
REFERENCE = Path(__file__).with_name('v63_runtime_reference.py')


class PatchContractError(RuntimeError):
    pass


def _statements(text):
    return ast.parse(text).body


def _one(values, label):
    if len(values) != 1:
        raise PatchContractError(label+' expected one anchor, found '+str(len(values)))
    return values[0]


def _assigns(node, name):
    if isinstance(node, ast.Assign):
        return any(isinstance(n, ast.Name) and n.id == name
                   for target in node.targets for n in ast.walk(target))
    return isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name


def _is_call(node, owner, name):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == owner
            and node.func.attr == name)


def _direct_function(parent, name):
    return _one([n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name], name)


def _insert_after(body, node, extra):
    index = body.index(node)
    body[index+1:index+1] = extra


def _insert_before(body, node, extra):
    index = body.index(node)
    body[index:index] = extra


def build_blocked_runtime(source=None):
    """Construct source only; exact frozen V63 bytes and all anchors must match.

    No compile(), exec(), import of Native, installation, or device work occurs
    here. This builder itself has not been executed in V75 preparation.
    """
    source = REFERENCE.read_text() if source is None else source
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise PatchContractError('Runtime input is not the pinned complete V63 source')
    module = ast.parse(source)
    function = _one([n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_generate_spec_rhythm_tree_decode'], 'V63 decode function')
    # Retain the original docstring, then block before any engine state exists.
    function.body[1:1] = _statements('''
_v75_require_local_bounded_scope()
_v75_hooks = _v75_make_hooks(self)
''')

    admission = _direct_function(function, 'admit_available')
    closed = _one([n for n in admission.body if _assigns(n, 'closed')], 'original closed result')
    _insert_after(admission.body, closed, _statements('''
_v75_hooks.reset_for_original_admission_fallback(active=active,
        cache_activated=cache_activated, cache_released=cache_released,
        release_online_cache=release_online_cache)
'''))
    capacity_guard = _one([n for n in admission.body if isinstance(n, ast.If)
                          and any(isinstance(x, ast.Name) and x.id == 'initial_batch_size'
                                  for x in ast.walk(n.test))], 'original active capacity guard')
    capacity_guard.test = ast.parse(
        'not pending_admission or len(active) + _v75_hooks.reserved_slots() >= initial_batch_size',
        mode='eval').body
    vacancy = _one([n for n in admission.body if _assigns(n, 'vacant_capacity')], 'original vacant capacity')
    vacancy.value = ast.parse(
        'initial_batch_size - len(active) - _v75_hooks.reserved_slots()', mode='eval').body
    candidates = _one([n for n in admission.body if _assigns(n, 'admission_candidates')], 'original admission ordering')
    _insert_after(admission.body, candidates, _statements('''
admission_candidates = _v75_hooks.filter_pending_candidates(admission_candidates)
'''))
    prefill = _one([n for n in admission.body if isinstance(n, ast.If)
                   and isinstance(n.test, ast.Name) and n.test.id == 'needs_prefill'], 'original standalone prefill')
    _insert_before(admission.body, prefill, _statements('''
if _v75_hooks.reserve_at_original_admission(admitted=admitted, needs_prefill=needs_prefill, active=active,
        pending=pending_admission, capacity=initial_batch_size, local_states=local_states):
    return closed
'''))

    receive = _direct_function(function, 'receive_live_admissions')
    abort_call = _one([n for n in ast.walk(receive) if isinstance(n, ast.Expr)
                      and _is_call(n.value, 'controller', 'invalidate_request')], 'original live abort invalidation')
    abort_parent = _one([n for n in ast.walk(receive) if hasattr(n, 'body')
                        and isinstance(n.body, list) and abort_call in n.body], 'live abort owner')
    _insert_before(abort_parent.body, abort_call, _statements(
        '_v75_hooks.abort_reserved(index, release_online_cache=release_online_cache)'))

    loop = _one([n for n in function.body if isinstance(n, ast.While)], 'original decode loop')
    submit = _direct_function(loop, '_dt_submit_current_tree_target')
    _insert_before(loop.body, submit, _statements('''
_v75_hooks.begin_target_cycle(target_payloads=target_payloads, target_indices=target_indices,
        local_states=local_states, cycle_id=round_count)
'''))
    temperatures = _one([n for n in submit.body if _assigns(n, 'target_temperatures')], 'target temperature rows')
    _insert_after(submit.body, temperatures, _statements('''
_v75_layout = _v75_hooks.plan_target(plans=target_plans, roots=target_roots,
        rows=target_rows, sequence_ids=target_request_ids,
        temperatures=target_temperatures, cycle_id=round_count)
'''))
    padding = _one([n for n in submit.body if isinstance(n, ast.If)
                    and any(_is_call(x, 'self', '_pad_spec_rhythm_target_tree_graph')
                            for x in ast.walk(n))], 'original deterministic target padding')
    padding.test = ast.BoolOp(op=ast.And(), values=[
        ast.parse('_v75_layout is None', mode='eval').body, padding.test])
    actual_target = _one([n for n in ast.walk(submit) if _assigns(n, 'target_output')], 'original target forward')
    target_with = _one([n for n in submit.body if isinstance(n, ast.With)
                       and actual_target in n.body], 'original TargetCompute scope')
    target_index = target_with.body.index(actual_target)
    replacement = _statements('''
if _v75_layout is not None:
    target_output = _v75_hooks.execute_target()
else:
    pass
''')[0]
    replacement.orelse = [actual_target]
    target_with.body[target_index] = replacement

    target_context = _one([n for n in loop.body if _assigns(n, 'target_output')
                          and isinstance(n, ast.Assign) and isinstance(n.value, ast.Name)
                          and n.value.id == 'target_context'], 'original target-context join')
    _insert_after(loop.body, target_context, _statements('''
if _v75_hooks.has_planned():
    _v75_hooks.prepare_chunk_at_original_target_join(local_states)
'''))
    # The new draft chunk joins at the original target-context boundary BEFORE
    # the existing target stream synchronize. The V45 native helper retains its
    # own local-stream fence / CPU rendezvous / finite vote / WORLD token order.
    # This is a proposed host schedule, not measured device-overlap evidence.

    finish = _one([n for n in loop.body if isinstance(n, ast.Expr)
                   and _is_call(n.value, 'controller', 'finish_cycle')], 'original successful finish_cycle')
    _insert_after(loop.body, finish, _statements('''
_v75_hooks.commit_after_original_finish_cycle(round_count,
        structural_failed=structural_failed, numerical_failed=numerical_failed,
        preflight_error=preflight_error, controller=controller, target_home=plan.target_home_batch_id)
'''))

    active_filter = _one([n for n in loop.body if _assigns(n, 'active')
                         and isinstance(n.value, ast.ListComp)], 'original finished-active filter')
    _insert_after(loop.body, active_filter, _statements('''
_v75_hooks.publish_after_original_cycle_clock(active=active, pending=pending_admission,
        capacity=initial_batch_size, local_states=local_states, context={
            'draft_states': draft_states, 'target_states': target_states, 'states': states,
            'prefetched': prefetched, 'pending_admission': pending_admission, 'active': active,
            'request_params': request_params, 'completed': completed,
            'finished_decode_elapsed': finished_decode_elapsed,
            'release_online_cache': release_online_cache, 'finished_predicate': _finished,
            'counters': counters, 'eos_token_ids': self.eos_token_ids,
        })
'''))
    # Publishing is deliberately AFTER incumbent cycle-clock charging and
    # finished-owner release. Publishing beside finish_cycle would charge the
    # new request for its own first-token prefill and change original TPOT.
    release = _one([n for n in function.body if isinstance(n, ast.Expr)
                    and _is_call(n.value, 'self', '_release_cache')], 'original generation cache release')
    _insert_before(function.body, release, _statements('_v75_hooks.close_scope_before_original_release()'))
    _insert_after(function.body, release, _statements(
        'self.last_worker_atomic_prefill_evidence = _v75_hooks.evidence()'))
    # Original docstring, guard and factory remain outside this error handler.
    # The supervisor is still responsible for a hardware/TP collective hang.
    original_body = function.body[3:]
    error_wrapper = _statements('''
try:
    pass
except BaseException as _v75_original_error:
    try:
        from exception_evidence_v140 import retain_decode_failure
        retain_decode_failure(self, _v75_original_error, locals())
    except BaseException:
        pass  # Retention/import failure must not suppress original fatal cleanup.
    _v75_hooks.fatal_cleanup_all4(_v75_original_error)
    raise
''')[0]
    error_wrapper.body = original_body
    function.body[3:] = [error_wrapper]
    ast.fix_missing_locations(module)
    return ast.unparse(module)+'\n'


def replace_fresh_decode_implementation(wrapper, implementation):
    """Preserve V63 generation/policy/progress wrapper; replace only its delegate."""
    names = wrapper.__code__.co_freevars
    if names.count('decode_impl') != 1 or wrapper.__closure__ is None:
        raise PatchContractError('Expected the ORIGINAL fresh_decode closure with one decode_impl')
    def cell(value):
        return (lambda: value).__closure__[0]
    cells = tuple(cell(implementation) if name == 'decode_impl' else existing
                  for name, existing in zip(names, wrapper.__closure__, strict=True))
    updated = FunctionType(wrapper.__code__, wrapper.__globals__, wrapper.__name__,
                           wrapper.__defaults__, cells)
    updated.__kwdefaults__ = wrapper.__kwdefaults__
    updated.__annotations__ = dict(wrapper.__annotations__)
    updated.__dict__.update(wrapper.__dict__)
    updated.__qualname__ = wrapper.__qualname__
    updated.__doc__ = wrapper.__doc__
    return updated


def install(native):
    # Only a local bounded diagnostic permit from actual fresh CPU/freeze/prior
    # evidence can enter. This never enables the formal online launch guard.
    require_bounded_scope()
    from runtime_factory import make_runtime_hooks
    original = native.NativePearlEngine._generate_spec_rhythm_tree_decode
    if Path(original.__globals__.get('P', '/')).resolve() != REFERENCE.parent:
        raise PatchContractError('A separately prepared V75 original fresh_decode wrapper is required; old directory writes forbidden')
    namespace = vars(native).copy()
    namespace.update(_v75_require_local_bounded_scope=require_bounded_scope,
                     _v75_make_hooks=make_runtime_hooks)
    source = build_blocked_runtime()
    exec(compile(source, '<V75-blocked-atomic-mixed-prefill>', 'exec'), namespace)
    native.NativePearlEngine._generate_spec_rhythm_tree_decode = replace_fresh_decode_implementation(
        original, namespace['_generate_spec_rhythm_tree_decode'])


if __name__ == '__main__':
    require_online_qualification()
