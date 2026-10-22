"""Run real frozen control-plane code without importing any model backend."""
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

HERE = Path(__file__).resolve().parent
B = HERE.parent
FROZEN = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust')
PEARL = FROZEN / 'vllm_ascend/spec_decode/pearl'
BACKENDS = {'torch', 'torch_npu', 'vllm', 'transformers'}


class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BACKENDS:
            raise RuntimeError('HOST401 attempted backend import ' + fullname)
        return None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_control_sources():
    sys.meta_path.insert(0, NoBackend())
    for name, path in [('vllm_ascend', FROZEN / 'vllm_ascend'),
                       ('vllm_ascend.spec_decode', PEARL.parent),
                       ('vllm_ascend.spec_decode.pearl', PEARL)]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    load('vllm_ascend.spec_decode.pearl.roofline', PEARL / 'roofline.py')
    return load('vllm_ascend.spec_decode.pearl.spec_rhythm', PEARL / 'spec_rhythm.py')


def state_set(rhythm, n, rng):
    classes = [('tight', 40.), ('normal', 50.), ('loose', 150.)]
    states = {}
    for i in range(n):
        label, slo = classes[i % 3]
        states[i] = rhythm.SpecRhythmRuntimeState(
            i, i % 2, slo_class=label, slo_tpot_ms=slo, max_gamma=rng.choice([1, 2, 5, 5, 5]),
            delivered_tokens=rng.randrange(1, 100), decode_elapsed_ms=rng.randrange(10000),
            decode_start_elapsed_ms=0., output_token_budget=256,
            acceptance_ema=rng.random(), verification_rounds=rng.randrange(50))
    return states


def main():
    rhythm = original_control_sources()
    sys.path[:0] = [str(B / 'tree-active64-whole-proposal-segment-limit-v398'),
                    str(B / 'tree-M120-S32-frugal-budget-projection-host-v397')]
    import policy401
    import packing401
    import projection397
    original_method = rhythm.SpecRhythmPipelineController.build_plan
    rng = random.Random(0)
    legacy = wide = packed = 0
    for trial in range(600):
        n = rng.randrange(25, 65) if trial % 2 else rng.randrange(1, 65)
        states = state_set(rhythm, n, rng)
        before_states = copy.deepcopy(states)
        shaper = rhythm.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
        normal_count = (rng.randrange(25, min(n, 32) + 1) if trial % 2
                        else rng.randrange(1, min(n, 32) + 1))
        normal = list(range(normal_count))
        eager = list(range(normal_count, min(n, normal_count + 4)))
        kwargs = dict(plan_id=trial, normal_request_indices=normal, eager_request_indices=eager,
                      states=states, projected_wait_ms=20., context_len=2048,
                      draft_token_budget=192, batch_size=n, eager_token_cap=4,
                      eager_reserve_tokens=8, verification_roof=rng.choice([0, 16, 64, 88, 96]))
        expected_shaper = copy.deepcopy(shaper)
        if not policy401.use_wide_normal_scope(batch_size=n, normal_indices=normal):
            expected = expected_shaper.shape(**kwargs)
            got = policy401.shape(shaper, **kwargs)
            assert expected == got and expected_shaper.__dict__ == shaper.__dict__
            legacy += 1
        else:
            got = policy401.shape(shaper, **kwargs)
            budgets = list(got.normal_budgets.values())
            assert not got.eager_budgets and len(budgets) <= 32 and sum(budgets) <= 88
            assert budgets.count(3) <= 24
            if budgets:
                lengths = [b + 1 for b in budgets]
                real_q = sum(lengths)
                remainder = 120 - real_q
                while remainder:
                    size = min(5, remainder)
                    lengths.append(size)
                    remainder -= size
                total = 0
                ends = []
                for size in lengths:
                    total += size
                    ends.append(total)
                layout = projection397.plan(ends, real_q)
                assert layout['selected_candidates'] == sum(budgets)
                assert sorted(layout['physical_to_original']) == list(range(120))
            wide += 1
        assert states == before_states and shaper.max_gamma == 8
    for trial in range(600):
        n = rng.randrange(1, 65)
        states = state_set(rhythm, n, rng)
        ctl = rhythm.SpecRhythmPipelineController(states)
        budgets = {i: rng.randrange(1, 4) for i in range(n)}
        for i, budget in budgets.items():
            ticket = ctl.new_ticket(i, gamma=budget, eager=False)
            ctl.publish([ticket])
            ctl._ready_wait_cycles[i] = rng.randrange(8)
        ready = dict(ctl.ready)
        got = packing401.build_plan(ctl, range(n), active_indices_registered_unique=True,
                                   verification_budget=88, max_target_requests=32,
                                   ready_candidate_counts=budgets,
                                   max_large_groups401=24,
                                   ready_large_groups401={i: int(b == 3) for i, b in budgets.items()},
                                   merge_ready_homes=True, target_slo_first=True,
                                   priority=True, projected_wait_ms=20.)
        selected = got.target_request_indices
        assert len(selected) <= 32 and sum(budgets[i] for i in selected) <= 88
        assert sum(budgets[i] == 3 for i in selected) <= 24
        assert set(selected).isdisjoint(got.deferred_target_request_indices)
        assert set(selected) | set(got.deferred_target_request_indices) == set(range(n))
        assert got.target_candidate_budgets == {i: budgets[i] for i in selected}
        assert all(ctl.ready[i] is ticket for i, ticket in ready.items())
        packed += 1
    # C88 and N32 alone cannot reject25 B3 tickets: they cost only82 candidates.
    # The independent large-group resource must retain a complete deferred ticket.
    states = state_set(rhythm, 32, rng)
    ctl = rhythm.SpecRhythmPipelineController(states)
    budgets = dict(enumerate([3] * 25 + [1] * 7))
    for i, budget in budgets.items():
        ctl.publish([ctl.new_ticket(i, gamma=budget, eager=False)])
    kwargs = dict(active_indices_registered_unique=True, verification_budget=88,
                  max_target_requests=32, ready_candidate_counts=budgets,
                  merge_ready_homes=True)
    old = original_method(copy.deepcopy(ctl), range(32), **kwargs)
    assert len(old.target_request_indices) == 32
    bounded = packing401.build_plan(ctl, range(32), **kwargs,
                                   max_large_groups401=24,
                                   ready_large_groups401={i: int(b == 3) for i, b in budgets.items()})
    assert len(bounded.target_request_indices) == 31
    assert len(bounded.deferred_target_request_indices) == 1
    assert sum(budgets[i] == 3 for i in bounded.target_request_indices) == 24
    try:
        packing401.build_plan(copy.deepcopy(ctl), range(32), **kwargs,
                              max_large_groups401=24,
                              ready_large_groups401={i: int(b == 3) for i, b in budgets.items()},
                              reserved_target_request_indices=old.target_request_indices,
                              reserved_target_home_batch_id=old.target_home_batch_id)
    except ValueError as error:
        assert 'large-group limit' in str(error)
    else:
        raise AssertionError('unsafe already-submitted large-group reservation accepted')
    # Shared original deferred-age accounting must survive repeated capped calls.
    states = state_set(rhythm, 64, rng)
    shaper = rhythm.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
    admitted = set()
    state_bytes = copy.deepcopy(states)
    for step in range(8):
        plan = policy401.shape(shaper, plan_id=step, normal_request_indices=range(32),
                               eager_request_indices=(), states=states,
                               projected_wait_ms=20., context_len=2048, batch_size=64,
                               draft_token_budget=192, verification_roof=8)
        admitted.update(plan.normal_budgets)
        assert plan.allocated_draft_tokens <= 8
    assert admitted == set(range(32)) and states == state_bytes
    prioritized = {
        i: rhythm.SpecRhythmRuntimeState(
            i, i % 2, slo_class='tight' if i < 24 else 'loose',
            slo_tpot_ms=40. if i < 24 else 150., max_gamma=5,
            delivered_tokens=1, decode_elapsed_ms=0. if i < 24 else 100000.)
        for i in range(32)
    }
    caps = policy401.normal_caps(shaper, range(32), prioritized, 20.)
    assert all(caps[i] == (3 if i < 24 else 2) for i in range(32))
    try:
        policy401.use_wide_normal_scope(batch_size=65, normal_indices=range(32))
    except ValueError:
        pass
    else:
        raise AssertionError('global65 admitted instead of rejected')
    assert rhythm.SpecRhythmPipelineController.build_plan is original_method
    assert not any(n.split('.')[0] in BACKENDS for n in sys.modules)
    result = dict(status='PASS_HOST401_ORIGINAL_CONTROL_ACCOUNTING_AND_WHOLE_TICKET_RESOURCES',
                  exact_legacy_shaper_cases=legacy, paid_wide_shaper_cases=wide,
                  whole_ready_ticket_resource_cases=packed,
                  C88_does_not_replace_independent_B3_quota_regression=True,
                  unsafe_replay_reservation_rejected=True,
                  repeated_paid_roof_original_age_coverage=32,
                  strict_class_priority_large_caps=True,
                  global65_rejected_without_lowering_global64=True,
                  original_request_states_and_controller_class_unchanged=True,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [PEARL / 'roofline.py', PEARL / 'spec_rhythm.py']},
                  packing=packing401.descriptor(), actual_global_limit=64,
                  HOST_package_shells_used_only_to_load_exact_pure_control_sources=True,
                  model_backends_imported=False, NPU_used=False, models_constructed=0,
                  production_runtime_installed=False, model_qualification=False,
                  production_permission=False, Raw=None, Goodput=None)
    (HERE / 'HOST401_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'exact_legacy_shaper_cases',
                     'paid_wide_shaper_cases', 'whole_ready_ticket_resource_cases']}))


if __name__ == '__main__':
    main()
