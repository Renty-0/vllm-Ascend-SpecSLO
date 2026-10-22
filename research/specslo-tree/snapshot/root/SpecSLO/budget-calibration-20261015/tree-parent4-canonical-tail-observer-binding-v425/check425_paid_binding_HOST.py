"""Real CPU controllers/derived callbacks; explicit device-proof double only.

The permission check is mocked only inside a context for control-plane tests.
No production record is issued and no model/Native work is executed.
"""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def main():
    import torch
    import torch_npu
    def blocked(*args, **kwargs):
        raise AssertionError('HOST425 no Native constructor/NPU initialization')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    readiness = require_ready()
    spec = dict(action=PARENT_ACTION, identity=readiness['parent']['identity'])
    B = Path(__file__).resolve().parent.parent
    with activate_local_diagnostic(spec):
        from extension425 import prepare_install, install_entry, after_install
        output = Path(tempfile.mkdtemp(prefix='HOST425-paid-',
            dir=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'))
        entry, Controls = prepare_install(output)
        from vllm_ascend.spec_decode.pearl import native_engine as native, native_graph as graph, spec_rhythm as rhythm
        from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan, pack_selected_tree_plan
        policy = install_entry(entry, native)
        import live_decode_patch
        overlay = after_install(native, graph, live_decode_patch)
        assert overlay['frontier_observer392_before_constructor'] is True
        native.NativePearlEngine.__init__ = blocked
        from factory425 import current
        current(native)
        from cpu_default_device_cleanup_host import capture_before_native_constructor
        constructor = capture_before_native_constructor(torch)
        assert constructor.eligible
        torch.set_default_device('cpu')
        from startup_config_v132 import make_config
        engine = SimpleNamespace(config=make_config().to_native(), cache_allocation=None,
                                 cache_block_tables=None, _dynamic_tree_generation=1)
        combination = json.loads((B/'tree-high-normal-cap4-native-families-v148/CANDIDATE_V132.json').read_text())
        controls = Controls(native=native, rhythm=rhythm, policy=policy, torch=torch,
            constructor_state=constructor, is_draft=True, configuration=combination['controls'])
        controls.select('candidate', engine=engine)
        engine._root425_controls = controls
        shaper = rhythm.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
        from normal_admission_v140 import reset
        import shape425
        import live425
        import admission425
        from audit_normal425 import verify_snapshot
        rng = random.Random(425)
        supported = fallbacks = 0
        for trial in range(160):
            generation = trial+1
            engine._dynamic_tree_generation = generation
            reset(policy, generation)
            states = {i: rhythm.SpecRhythmRuntimeState(i, i % 2,
                slo_class=('tight','normal','loose')[i % 3],
                slo_tpot_ms=(40.,50.,150.)[i % 3], max_gamma=5,
                delivered_tokens=rng.randrange(1,100), decode_elapsed_ms=rng.randrange(10000),
                decode_start_elapsed_ms=0., output_token_budget=256,
                acceptance_ema=rng.random(), verification_rounds=rng.randrange(50))
                for i in range(64)}
            controller = rhythm.SpecRhythmPipelineController(states)
            local = {i: SimpleNamespace(token_ids=[0]*129) for i in range(64)}
            payloads = {}
            parents = tuple(range(48,56))
            for i in parents:
                budget = 3 if trial % 5 == 0 else 4
                plan = pack_selected_tree_plan(cached_cpu_tree_speculation_plan(
                    2,4,128,4096,candidate_budget=8), range(budget))
                ticket = controller.new_ticket(i, gamma=budget, eager=False)
                controller.publish([ticket])
                payloads[ticket.proposal_id] = dict(ticket=ticket, plan=plan, row=list(range(budget)))
            before = copy.deepcopy(states)
            before_counts = dict(policy.counts)
            kwargs = dict(plan_id=trial,
                normal_request_indices=tuple(range(24 if trial % 2 == 0 else 32)),
                eager_request_indices=parents, states=states, projected_wait_ms=20.,
                context_len=1024, batch_size=64, draft_token_budget=192,
                eager_token_cap=3, eager_reserve_tokens=32, verification_roof=96)
            # This double exercises only HOST dispatch. It confers no actual
            # engine-owned proof/capability and is restored after each block.
            with patch.object(live425, 'permission', return_value=object()):
                counters = {}
                selected = shape425.filter_parents(engine, parents, controller, payloads, counters, local)
                assert selected == list(parents)
                result = shape425.shape(engine, shaper, current_controller=controller,
                    current_payloads=payloads, current_local_states=local, **kwargs)
            assert not hasattr(engine, '_root425_E4_permission')
            assert states == before
            snapshot = shape425.snapshot(engine)
            verify_snapshot(snapshot, generation)
            assert snapshot['normal_admission']['call_count'] == 1
            assert len(result.normal_budgets) <= 24
            assert result.allocated_selected_candidates <= 192
            actual = lambda key: policy.counts[key]-before_counts.get(key, 0)
            assert actual('actual_draft_normal_rows') == len(result.normal_budgets)
            assert actual('actual_draft_eager_rows') == len(result.eager_budgets)
            assert actual('actual_draft_work_rows') == len(result.normal_budgets)+len(result.eager_budgets)
            assert actual('split_draft_final_normal_candidates') == sum(result.normal_budgets.values())
            assert actual('split_draft_final_eager_candidates') == sum(result.eager_budgets.values())
            work = engine._root425_current_work
            if work is not None:
                supported += 1
                assert len(result.normal_budgets) == 24 and len(result.eager_budgets) == 4
                assert result.allocated_selected_candidates == 112
                assert result.work_limit.eager_per_request == 4
                assert work.owner is engine and work.generation == generation
                assert work.ids == tuple((*result.normal_budgets,*result.eager_budgets))
                for owner, plan, row in work.parents:
                    actual = payloads[controller.ready[owner].proposal_id]
                    assert actual['plan'] is plan and tuple(actual['row']) == row
            else:
                fallbacks += 1
                assert all(b == 3 for b in result.eager_budgets.values())
        assert supported and fallbacks
        # Exact real gate is restored: even a selected HOST work descriptor
        # cannot issue permission or invoke the neural E4 path.
        try:
            live425.permission(engine)
        except RuntimeError as error:
            assert 'current engine/model proof record' in str(error)
        else:
            raise AssertionError('HOST control descriptor granted NN permission')
        controls.select('original', engine=engine)
        torch.set_default_device(None)
        value = dict(status='PASS_HOST425_ACTUAL_DERIVED_CONTROL_PAID_E4_AND_FALLBACK_BINDING',
            HOST_cases=160, supported=supported, fallbacks=fallbacks,
            actual_original_controls_and_controller=True, original_gain_and_states_preserved=True,
            one_age_finish_per_call=True, same_actual_parent_objects_and_rows=True,
            explicit_device_permission_double_only_for_HOST=True,
            real_permission_restored_and_unissued=True, Native_models_constructed=0,
            NPU_used=False, production_permission=False, Raw=None, Goodput=None)
        (Path(__file__).resolve().parent/'HOST425_PAID_BINDING_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value))


if __name__ == '__main__':
    main()
