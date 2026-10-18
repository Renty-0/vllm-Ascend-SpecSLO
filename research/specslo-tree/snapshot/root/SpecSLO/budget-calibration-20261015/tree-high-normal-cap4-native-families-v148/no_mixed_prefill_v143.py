"""Production-only admission control; no clocks, tensors or collectives.

The unchanged qualification actors continue to use their original class API.
The production caller validates this contract in its existing untimed install
stage, then binds only the fresh generation's live hook instance.
"""
import hashlib
import json
from pathlib import Path
from types import FunctionType

MODE = 'standalone-whole-prompt-no-new-reservation'
P = Path(__file__).resolve().parent
ORIGINAL_SOURCES = {
    'original_hook_adapter.py': 'b4038ad6088ba82ed7707c9cfc1982100d03091a54dfe37309641c9d23be5a11',
    'live_runtime_hooks.py': '445eb475cf2ac8a854bebc039caf3078aacf758e904e2c82f891c41adc1308f5',
}
_NONE_FIELDS = ('_planned', '_flexible', '_target_samples', '_target_submit_error',
    '_ready_publication_identity', '_eligible_cycle', '_standalone_fallback_owner',
    'descriptor', 'prepared_call', '_cycle_inputs', '_cycle_tickets')
_FALSE_FIELDS = ('_fallback_requested', '_execution_attempted', '_fatal')


def _require(value, message):
    if not value:
        raise RuntimeError('V143 production prefill: '+message)


def prepare_production_contract(directory=P):
    """Call only inside the existing untimed all-rank install callback."""
    from original_hook_adapter import RuntimeHooks
    from live_runtime_hooks import LiveRuntimeHooks
    from poisson_coverage_probe import _expected_code
    directory = Path(directory).resolve()
    _require(directory == P, 'actual helper directory changed')
    configuration = json.loads((directory/'CANDIDATE_V132.json').read_text())
    _require(configuration.get('production_mode') == MODE, 'explicit production mode missing')
    for name, expected in ORIGINAL_SOURCES.items():
        path = directory/name
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                 'original qualification class source changed: '+name)
    method = RuntimeHooks.reserve_at_original_admission
    _require(isinstance(method, FunctionType)
        and Path(method.__globals__.get('__file__', '')).resolve()
            == directory/'original_hook_adapter.py'
        and Path(LiveRuntimeHooks.__init__.__globals__.get('__file__', '')).resolve()
            == directory/'live_runtime_hooks.py'
        and method.__code__ == _expected_code(directory/'original_hook_adapter.py',
                                             'reserve_at_original_admission')
        and LiveRuntimeHooks.reserve_at_original_admission is method,
        'original qualification reservation class method changed')
    return dict(schema='V143.production-prefill-contract.v1', mode=MODE,
        original_class_source_sha256=dict(ORIGINAL_SOURCES),
        original_qualification_class_method_unchanged=True,
        common_guard='existing untimed install-original-live stage',
        new_clock_calls=0, new_collective_calls=0, new_fence_calls=0,
        mixed_qualification_actors_modified=False,
        per_callback_observer_or_counter_installed=False)


def _assert_empty(hooks):
    """Read only existing HOST manager fields; fail through original handling."""
    _require(hooks.transactions.current is None and hooks.reserved_slots() == 0
        and hooks.transactions.logical_terminal_requests == (),
        'generation already owns a reservation or publication')
    _require(all(getattr(hooks, name) is None for name in _NONE_FIELDS),
             'generation already owns a mixed descriptor, chunk or publication')
    _require(all(getattr(hooks, name) is False for name in _FALSE_FIELDS)
        and hooks._unreleased_dispositions == {},
        'generation has fallback, fatal or unreleased reservation state')


class _Control:
    def __init__(self, hooks, callback, contract):
        self.hooks, self.callback, self.contract = hooks, callback, dict(contract)

    def evidence(self):
        # Observed only when retaining the existing completed trace. No new tap
        # runs per round, and no invocation count is invented.
        error = None
        try:
            _assert_empty(self.hooks)
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
        return dict(contract=self.contract, instance_callback_installed=True,
            initial_empty_guard_passed=True,
            callback_owned_at_trace_end=(self.hooks.__dict__.get(
                'reserve_at_original_admission') is self.callback),
            terminal_empty_guard_passed=error is None, terminal_guard_error=error,
            admission_callback_invocations=None,
            reservation_callback_result=False,
            current_generation_only=True, native_or_class_method_replaced=False)


def install_on_fresh_hooks(hooks, contract):
    """HOST-only instance bind, called by the production generation factory."""
    _require(contract.get('mode') == MODE
        and contract.get('original_class_source_sha256') == ORIGINAL_SOURCES
        and contract.get('common_guard') == 'existing untimed install-original-live stage'
        and contract.get('original_qualification_class_method_unchanged') is True,
        'untimed original source/mode contract missing')
    _require('reserve_at_original_admission' not in hooks.__dict__,
             'instance reservation callback was already overridden')
    _assert_empty(hooks)

    def standalone(*, admitted, needs_prefill, active, pending, capacity, local_states):
        _assert_empty(hooks)
        # The original admission continuation now performs its own entire
        # prompt prefill, finite vote, first-token broadcast and old-active charge.
        return False

    hooks.reserve_at_original_admission = standalone
    return _Control(hooks, standalone, contract)


def audit_production_mode_trace(trace, count):
    """Additional gate; the caller also runs every unchanged original audit."""
    _require(trace.get('production_prefill_mode') == MODE, 'trace production mode differs')
    control = trace.get('production_prefill_control')
    _require(type(control) is dict and control.get('contract', {}).get('mode') == MODE
        and control['contract'].get('original_class_source_sha256') == ORIGINAL_SOURCES
        and control['contract'].get('original_qualification_class_method_unchanged') is True
        and control['contract'].get('common_guard') == 'existing untimed install-original-live stage'
        and all(control['contract'].get(name) == 0 for name in
                ('new_clock_calls', 'new_collective_calls', 'new_fence_calls'))
        and control['contract'].get('mixed_qualification_actors_modified') is False
        and control['contract'].get('per_callback_observer_or_counter_installed') is False,
        'actual production install/source contract missing')
    _require(all(control.get(name) is True for name in
        ('instance_callback_installed', 'initial_empty_guard_passed',
         'callback_owned_at_trace_end', 'terminal_empty_guard_passed', 'current_generation_only'))
        and control.get('terminal_guard_error') is None
        and control.get('admission_callback_invocations') is None
        and control.get('reservation_callback_result') is False
        and control.get('native_or_class_method_replaced') is False,
        'production callback ownership or empty-state evidence differs')
    _require(trace.get('logical_events') == [] and trace.get('begin_inputs') == []
        and trace.get('close_abort_identities') == [], 'production mixed reservation/work observed')
    terminal = trace.get('terminal_evidence', {})
    _require(terminal.get('reservation_remaining') is False
        and terminal.get('logical_terminal_request_indices') == []
        and all(terminal.get(name) is None for name in
            ('committed_cursor', 'prepared_cursor', 'last_committed_cycle_id',
             'initial_fixture_cursor', 'initial_fixture_scope')),
        'terminal mixed reservation/publication state observed')
    prompts = trace.get('tokenized_inputs')
    _require(type(prompts) is list and len(prompts) == count,
             'complete tokenized inputs absent')
    owners, singleton, multiple = [], 0, 0
    calls = trace.get('standalone_prefill_calls')
    _require(type(calls) is list and calls, 'actual standalone calls absent')
    for call in calls:
        ids = call.get('sequence_ids')
        _require(type(ids) is list and ids and len(set(ids)) == len(ids)
            and all(type(i) is int and 0 <= i < count for i in ids), 'standalone owners invalid')
        _require(call.get('cached_tokens') == [0]*len(ids)
            and call.get('prompt_lengths') == [len(prompts[i]) for i in ids]
            and call.get('uncached_queries') == sum(len(prompts[i]) for i in ids)
            and call.get('precomputed_target_tokens') is False
            and call.get('draft_prefill_completed') is False,
            'standalone must materialize the whole genuine prompt in both roles')
        owners.extend(ids)
        singleton += int(len(ids) == 1)
        multiple += int(len(ids) > 1)
    _require(sorted(owners) == list(range(count)), 'every owner needs exactly one standalone admission')
    return dict(mode=MODE, mixed_commits=0, mixed_publications=0, reservations=0,
        whole_prompt_standalone_owners=len(owners), standalone_batches=len(calls),
        singleton_batches=singleton, multi_owner_batches=multiple,
        exact_201_graph_execution_inferred_for_multi_owner_batches=False,
        old_qualification_scopes_retained=True, numerical_qualification_inferred=False)
