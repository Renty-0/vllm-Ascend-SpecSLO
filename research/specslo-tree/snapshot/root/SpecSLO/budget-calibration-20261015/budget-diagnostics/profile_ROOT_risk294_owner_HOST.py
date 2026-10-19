"""Measure an allocation-only hypothesis without importing the device runtime."""
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
import types
from pathlib import Path


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
SOURCE = BASE / 'tree-request-conditional-risk-gain-host-v294/risk294.py'
POLICY_DIR = BASE / 'tree-high-normal-cap4-native-families-v148'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cached_owner(module, state, request):
    cls = request.slo_class
    if cls not in module.CLASSES:
        raise ValueError('ROOT294 original three SLO classes')
    key = str(request.request_index)
    try:
        row = state['requests'][key]
    except KeyError:
        row = dict(slo_class=cls, raw=module.blank(), recent=module.blank(),
                   observations=0, cache=None)
        state['requests'][key] = row
    if row['slo_class'] != cls:
        raise RuntimeError('ROOT294 request identity changed class')
    return row


def verdict(budget, path, remaining):
    tokens = list(range(100, 101 + len(path)))
    return dict(parents=(-1, 0, 1, 2, -1, 4, 5, 6)[:budget],
                accepted=path + [-1] * (4 - len(path)),
                outputs=tokens + [-1] * (5 - len(tokens)),
                remaining_before=remaining)


def policy(module, model):
    from budget_policy import TreeBudgetPolicy
    result = TreeBudgetPolicy(model)
    module.begin_generation(result, 1)
    return result


def compare(original, candidate, model):
    policies = [policy(m, model) for m in (original, candidate)]
    requests = [types.SimpleNamespace(request_index=i,
                slo_class=original.CLASSES[i % 3], output_token_budget=256,
                delivered_tokens=1) for i in range(48)]
    rng = random.Random(294)
    observations = 0
    predictions = 0
    for generation in range(1, 4):
        if generation > 1:
            for module, current in zip((original, candidate), policies):
                current.observed.clear()
                module.begin_generation(current, generation)
        for turn in range(720):
            request = requests[rng.randrange(len(requests))]
            budget = rng.randint(1, 8)
            if budget >= 5 and rng.randrange(3) == 0:
                depth = rng.randint(1, budget - 4)
                path = list(range(4, 4 + depth))
            else:
                path = list(range(rng.randint(0, min(4, budget))))
            event = verdict(budget, path, 4 if turn % 37 == 0 else 255)
            results = [module.observe(current, request, **event)
                       for module, current in zip((original, candidate), policies)]
            assert results[0] == results[1]
            observations += 1
            for n in range(9):
                assert policies[0].gain(request, n) == policies[1].gain(request, n)
                predictions += 1
            assert original.prediction(policies[0], request) == candidate.prediction(policies[1], request)
        assert policies[0].observed == policies[1].observed
        assert original.snapshot(policies[0]) == candidate.snapshot(policies[1])
    # Request identities and generation resets remain fail-closed.
    for module, current in zip((original, candidate), policies):
        wrong = copy.copy(requests[0])
        wrong.slo_class = 'normal'
        try:
            module.prediction(current, wrong)
        except RuntimeError:
            pass
        else:
            raise AssertionError('class change accepted')
        try:
            module.begin_generation(current, 2)
        except ValueError:
            pass
        else:
            raise AssertionError('stale generation accepted')
    return policies, requests, dict(observations=observations,
                                   exact_gain_comparisons=predictions,
                                   complete_state_and_history_equal=True,
                                   identity_and_stale_generation_rejected=True)


def measure(module, current, requests, iterations, invalidate):
    checksum = 0.
    started = time.perf_counter_ns()
    for turn in range(iterations):
        if invalidate:
            current._ROOT294['revision'] += 1
        request = requests[turn % len(requests)]
        checksum += current.gain(request, 4) + current.gain(request, 8)
    return (time.perf_counter_ns() - started) / 1e9, checksum


def check_eligibility(policies, requests):
    sys.path.extend(str(BASE / name) for name in (
        'tree-current-prepare-cost-eligibility-host-v312',
        'tree-spare-capacity-budget-host-v290',
        'tree-spare-branch-group-budget-full-v260'))
    from budget312 import eligible
    rng = random.Random(312)
    states = {request.request_index: request for request in requests}
    for request in requests:
        request.max_gamma = 8
        request.projected_progress_gap = lambda wait, i=request.request_index: i / 7.
    nonempty = 0
    cases = 576
    for case in range(cases):
        count = 1 + case % 24
        normal = tuple(rng.sample(list(states), count))
        choices = [eligible(current, states, normal) for current in policies]
        assert choices[0] == choices[1]
        nonempty += bool(choices[0])
    assert 0 < nonempty < cases
    return dict(actual_original_budget312_eligible_cases=cases,
                identical_selected_owner_sets=True, cases_selecting_branches=nonempty,
                identical_policy_counters=policies[0].counts == policies[1].counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-eligibility', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    sys.path.insert(0, str(POLICY_DIR))
    original, candidate = load('original_risk294'), load('lookup_only_risk294')
    candidate.owner = lambda state, request: cached_owner(candidate, state, request)
    model = json.loads((POLICY_DIR / 'tree-gain-model.json').read_text())
    policies, requests, checks = compare(original, candidate, model)
    if args.check_eligibility:
        checks.update(check_eligibility(policies, requests))
    rows = []
    # Alternate execution order; these are isolated CPU microbenchmarks.
    for invalidate in (False, True):
        samples = [[], []]
        for repeat in range(8):
            values = {}
            for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                values[arm] = measure((original, candidate)[arm], policies[arm],
                                      requests, 12000, invalidate)
                samples[arm].append(values[arm][0])
            assert values[0][1] == values[1][1]
        medians = [statistics.median(s) for s in samples]
        rows.append(dict(cache_revision_changes_each_request=invalidate,
                         requests_per_sample=12000, gain_calls_per_sample=24000,
                         original_seconds=samples[0], lookup_only_seconds=samples[1],
                         original_median=medians[0], lookup_only_median=medians[1],
                         isolated_speedup=medians[0] / medians[1]))
    assert not any(n.split('.')[0] in ('torch', 'torch_npu') for n in sys.modules)
    assert original.owner is not candidate.owner
    result = dict(status='HOST_only_lookup_allocation_equivalence_and_microcost',
                  original_source=str(SOURCE),
                  original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  profile_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checks=checks, timings=rows, NPU_used=False,
                  production_source_changed=False, active348_changed=False,
                  synthetic_legal_verdicts=True, measured_end_to_end_Raw=None,
                  measured_Goodput=None, implementation_selected=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
