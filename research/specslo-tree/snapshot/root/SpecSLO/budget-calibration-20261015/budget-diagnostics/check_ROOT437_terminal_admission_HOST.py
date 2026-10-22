"""Model-free synthetic controls for the future terminal admission auditor.

The clocks and callbacks here are explicit synthetic auditor inputs. They do
not constitute Native execution, NN qualification, or performance evidence.
"""
import ast
import collections
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-refill-witness-boolean-retention-full-v437'


class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
            raise RuntimeError('HOST437 auditor cannot import a model: ' + fullname)


def need(value, message):
    if not value:
        raise AssertionError(message)


def main():
    sys.meta_path.insert(0, NoModel())
    for folder in ('tree-nonempty-current-incumbent-full-proposal-v432',
                   'tree-nonempty-refill-accounting-witness-host-v429',
                   'tree-nonempty-wholeprompt-refill-resource-host-v426'):
        sys.path.insert(0, str(B / folder))
    auditor = D / 'audit_ROOT437_terminal_full_HOST.py'
    parsed = ast.parse(auditor.read_text())
    calls = [n for n in ast.walk(parsed) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'original' and n.func.attr == 'definitions']
    need(len(calls) == 1, 'one actual future terminal function projection')
    replacements = ast.literal_eval(calls[0].args[2])
    parent = D / 'audit_v197_terminal_HOST.py'
    source = parent.read_text()
    for before, after in replacements:
        source = source.replace(before, after)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'admission_and_full_statistics')
    ns = dict(need=need, collections=collections,
              common_full_statistics=lambda traces, reports: dict(synthetic_auditor_control=True))
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(auditor) + '::exact-terminal-admission-projection', 'exec'), ns)
    actual = ns['admission_and_full_statistics']
    owners = [list(range(8)), list(range(8, 16))]
    witnesses = []
    for k, group in enumerate(owners):
        old = [] if k == 0 else owners[0]
        witnesses.append(dict(incumbents=old[:], admitted=group[:],
            original_arrival_clock=1000., prefill_ms=30.,
            total_reserved_pages=64 + 3 * (len(old) + len(group)),
            incumbent_decode_before_ms=[50.] * len(old),
            incumbent_decode_after_ms=[80.] * len(old),
            original_incumbent_tokens_epochs_homes_and_ready_tickets_unchanged=True,
            new_root_tokens=[100 + i for i in group]))
    trace = dict(scope_count=16, tokenized_inputs=[list(range(63)) for _ in range(16)],
        absolute_arrivals=[999.] * 16,
        standalone_prefill_calls=[dict(sequence_ids=g) for g in owners],
        stream_events=[dict(request_index=i, token_ids=[100+i], finished=False)
                       for i in range(16)] +
                      [dict(request_index=i, token_ids=[7], finished=True)
                       for i in range(16)],
        ROOT432_refill_witnesses=witnesses, sealed_before=True, sealed_after=True,
        cache_released=True, configured_capacity=64, admission_window160=64,
        active_peak=16, actual_capacity64_scope_completed=False,
        decode_counters=dict(spec_rhythm_slo_aware_admission_enabled=1,
            spec_rhythm_target_slo_first_enabled=1, spec_rhythm_prefill_coalesce_enabled=0,
            spec_rhythm_prefill_coalesce_deferred_polls=0))
    config = dict(spec_rhythm_prefill_coalesce_min_requests=1,
        spec_rhythm_prefill_coalesce_max_wait_ms=0, spec_rhythm_verification_budget=96)
    reports = [dict(configuration_before_constructor=config,
                    configuration_after_constructor=config) for _ in range(4)]
    traces = [copy.deepcopy(trace) for _ in range(4)]
    result = actual(traces, reports)
    replay = result['actual_admission_cohort_replay']
    need(replay['nonempty_cohorts'] == 1 and replay['exact_original_incumbent_prefill_charge'],
         'actual new nonempty witness contract accepted by exact terminal projection')
    need('empty_active_cohort_replay' not in result and
         result['incumbent_prefill_pause_memberships_exactly_charged'] is True,
         'truthful nonempty labels retained')
    negatives = []
    for kind in ('missing_old_charge', 'wrong_old_membership', 'missing_witness', 'future_arrival'):
        bad = copy.deepcopy(traces)
        for t in bad:
            w = t['ROOT432_refill_witnesses']
            if kind == 'missing_old_charge':
                w[1]['incumbent_decode_after_ms'][0] = 50.
            elif kind == 'wrong_old_membership':
                w[1]['incumbents'][0] = 999
            elif kind == 'missing_witness':
                t.pop('ROOT432_refill_witnesses')
            else:
                w[1]['original_arrival_clock'] = 998.
        try:
            actual(bad, reports)
        except (RuntimeError, AssertionError) as error:
            negatives.append(dict(kind=kind, rejected=True, error=str(error)))
        else:
            raise AssertionError('terminal admission negative accepted: ' + kind)
    value = dict(status='PASS_HOST437_EXACT_TERMINAL_NONEMPTY_CONTRACT',
        terminal_auditor=str(auditor),
        terminal_auditor_sha256=hashlib.sha256(auditor.read_bytes()).hexdigest(),
        parent_function_source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        explicit_synthetic_admission_and_completion_inputs=True,
        original_common_metric_function_not_executed_in_this_control=True,
        positive_nonempty_cohorts=1, rejected_controls=negatives,
        NPU_used=False, Torch_imported=False, NN_qualification=False,
        Raw=None, Goodput=None)
    output = D / 'ROOT437_TERMINAL_ADMISSION_HOST_RESULT.json'
    with output.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
