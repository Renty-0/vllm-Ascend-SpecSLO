#!/usr/bin/env python3
"""V121: stdlib-only source feasibility and counterexamples; installs nothing.

The contract below is a diagnostic type/validator, not an executable scheduler.
Original allocators, wrapper and selectors are evaluated from their real ASTs.
No Torch, Native module import, model, tensor operation or device query occurs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-s35-combined-controls-launchfix-v116'
CURRENT = B / 'tree-s35-qualified-counts-v119'
SHADOW = B / 'tree-eager-shadow-adapter-v118'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
FREEZE = P / 'PREPARED_SOURCE_HASHES_V116.json'
FREEZE_SHA = '79b7044096f278de2ad2032bde5fad7c16d8b675c4036abe42cc9dd3b6e25885'
CORE = ('budget_policy.py', 'candidate_budget_policy.py', 'tree_entry.py',
        'runtime_tree_decode.py', 'backbone_selection.py', 'packed_window.py',
        'startup_config_v75.py', 'tree-gain-model.json', 'canonical_eager.py',
        'fast_tree_draft_v5.py', 'root_seed.py', 'combined_controls_v113.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node(source, name, kind):
    found = [n for n in ast.walk(ast.parse(source)) if isinstance(n, kind) and n.name == name]
    if len(found) != 1:
        raise AssertionError((name, len(found)))
    return found[0]


def execute_nodes(path, nodes, additions=None):
    scope = dict(__name__='v121_readonly_ast', dataclass=dataclass,
                 Mapping=Mapping, Sequence=Sequence, math=math)
    scope.update(additions or {})
    # Dataclasses below do not use postponed annotation strings. Executing
    # selected original nodes avoids their surrounding Native imports.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+':V121-AST',
                 'exec', dont_inherit=True), scope)
    return scope


def reject(callback, contains=None):
    try:
        callback()
    except (ValueError, RuntimeError, TypeError) as error:
        if contains is not None:
            assert contains in str(error), (contains, str(error))
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError('Required negative case was accepted')


@dataclass(frozen=True)
class TargetAdmissionLimit:
    """Current target call only; remains C96/Q128 and at most32 real rows."""
    candidates: int = 96
    queries: int = 128
    rows: int = 32

    def check(self, budgets):
        values = tuple(budgets)
        if any(type(b) is not int or b <= 0 for b in values):
            raise ValueError('Target budgets must be positive HOST integers')
        c, q = sum(values), sum(b+1 for b in values)
        if len(values) > self.rows or c > self.candidates or q > self.queries:
            raise ValueError('Current target exceeds unchanged C96/Q128/N32 admission')
        return dict(real_rows=len(values), C=c, Q=q)


@dataclass(frozen=True)
class SplitDraftWorkAudit:
    """Proposed typed validator only, deliberately no verification_roof alias.

    Units are retained candidates, not explored ABI nodes, model queries or
    milliseconds. Window/row caps remain independent mandatory constraints.
    Existing proposal tickets retain epochs, parent tokens and lifecycle.
    """
    normal: tuple[tuple[int, int], ...]
    future_eager: tuple[tuple[int, int], ...]
    window_eager_rows: int
    normal_candidate_limit: int = 96
    future_eager_candidate_limit: int = 96
    global_selected_candidate_limit: int = 192
    caller_selected_candidate_limit: int | None = None
    eager_per_request_limit: int = 3
    normal_row_limit: int = 32
    eager_row_limit: int = 32
    physical_work_row_limit: int = 64
    target_limit: TargetAdmissionLimit = TargetAdmissionLimit()

    def __post_init__(self):
        for name in ('normal_candidate_limit', 'future_eager_candidate_limit',
                     'global_selected_candidate_limit', 'eager_per_request_limit',
                     'normal_row_limit', 'eager_row_limit', 'physical_work_row_limit'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Limits must be positive HOST integers')
        if type(self.window_eager_rows) is not int or self.window_eager_rows < 0:
            raise ValueError('Window must explicitly admit a nonnegative number of eager rows')
        if self.caller_selected_candidate_limit is not None and (
                type(self.caller_selected_candidate_limit) is not int
                or self.caller_selected_candidate_limit < 0):
            raise ValueError('Explicit caller draft cap must be nonnegative')
        rows = self.normal + self.future_eager
        if any(type(i) is not int or i < 0 or type(b) is not int or b <= 0 for i, b in rows):
            raise ValueError('Owner IDs and candidate counts must be HOST integers')
        if len({i for i, _ in rows}) != len(rows):
            raise ValueError('Normal/future eager owners must be unique and disjoint')
        nc, ec = sum(b for _, b in self.normal), sum(b for _, b in self.future_eager)
        if nc > self.normal_candidate_limit or nc > self.target_limit.candidates:
            raise ValueError('Normal candidates exceed their independent96 limit')
        if ec > self.future_eager_candidate_limit:
            raise ValueError('Future eager candidates exceed their independent96 limit')
        if any(b > self.eager_per_request_limit for _, b in self.future_eager):
            raise ValueError('Future eager per-request cap3 is explicit')
        limit = self.global_selected_candidate_limit
        if self.caller_selected_candidate_limit is not None:
            limit = min(limit, self.caller_selected_candidate_limit)
        if nc+ec > limit:
            raise ValueError('Combined draft selected candidates exceed explicit global/caller cap')
        if (len(self.normal) > self.normal_row_limit
            or len(self.future_eager) > self.eager_row_limit
            or len(rows) > self.physical_work_row_limit):
            raise ValueError('Original normal/eager/combined row bounds still apply')
        if len(self.future_eager) > self.window_eager_rows:
            raise ValueError('Unadmitted eager work cannot bypass the measured row window')


def fixture_state(index, gap, *, max_gamma=5):
    state = SimpleNamespace(request_index=index, max_gamma=max_gamma,
        slo_class='tight', slo_tpot_ms=40., decode_elapsed_ms=gap*40.,
        decode_start_elapsed_ms=0., delivered_tokens=1, output_token_budget=256)
    state.projected_progress_gap = lambda wait: max(0., gap + max(0., wait)/40.)
    return state


def traced_allocate(policy, normal, eager, states, limit):
    captured = {}
    code = policy.allocate.__func__.__code__
    previous = sys.gettrace()
    def trace(frame, event, value):
        if frame.f_code is code and event == 'return':
            captured.update({key: frame.f_locals[key] for key in ('needs', 'caps', 'grants', 'left')})
        return trace
    try:
        sys.settrace(trace)
        normal_grants, eager_grants = policy.allocate(normal, eager, states, 0., limit)
    finally:
        sys.settrace(previous)
    return dict(normal=normal_grants, eager=eager_grants,
        normal_C=sum(normal_grants.values()), eager_C=sum(eager_grants.values()),
        actual_original_locals=captured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=D/'V121_SPLIT_DRAFT_TARGET_BUDGET_HOST_AUDIT.json')
    args = parser.parse_args()
    started = time.perf_counter()
    assert sys.version_info[:2] == (3, 11), 'Retained3.11 is the source AST interpreter'
    assert not any(name == 'torch' or name.startswith(('torch.', 'torch_npu', 'vllm')) for name in sys.modules)
    assert sha(FREEZE) == FREEZE_SHA
    frozen = json.loads(FREEZE.read_text())
    paths = [FREEZE, Path(__file__).resolve()]
    core_binding = {}
    for name in CORE:
        parent, current = P/name, CURRENT/name
        assert sha(parent) == frozen['files'][str(parent)]
        assert sha(current) == sha(parent), 'V119 budgeting core differs: '+name
        core_binding[name] = dict(parent=str(parent), current=str(current), sha256=sha(parent))
        paths += [parent, current]
    paths += [NATIVE/'spec_rhythm.py', NATIVE/'tree_budget.py', NATIVE/'api.py',
              SHADOW/'eager_shadow_v118.py', SHADOW/'source_contract_v118.py',
              SHADOW/'runtime_taps_v118.py', SHADOW/'PARENT_BINDING_V118.json']
    before = {str(p): sha(p) for p in paths}
    tests, observations = [], {}
    def check(name, callback):
        value = callback()
        tests.append(dict(name=name, passed=True))
        return value

    policy_source = (P/'budget_policy.py').read_text()
    candidate_source = (P/'candidate_budget_policy.py').read_text()
    original = execute_nodes(P/'budget_policy.py', ast.parse(policy_source).body)['TreeBudgetPolicy']
    family = execute_nodes(P/'candidate_budget_policy.py', ast.parse(candidate_source).body)['TreeBudgetPolicy']
    model = json.loads((P/'tree-gain-model.json').read_text())
    seed = original(model)
    gap5 = (seed.base['tight'][4]+seed.base['tight'][5])/2
    gap3 = (seed.base['tight'][2]+seed.base['tight'][3])/2
    normal, eager = tuple(range(20)), (20,)
    states = {i: fixture_state(i, gap5 if i in normal else gap3) for i in normal+eager}
    a = traced_allocate(original(model), normal, eager, states, None)
    b = traced_allocate(original(model), normal, eager, states, 192)
    assert a == b
    assert a['normal_C'] == 96 and a['eager_C'] == 0
    assert Counter(a['normal'].values()) == Counter({5:19, 1:1})
    assert all(a['actual_original_locals']['needs'][i] == 5 for i in normal)
    assert a['actual_original_locals']['needs'][20] == 3
    check('actual_original_allocator_19B5_plus1B1_exhausts96_despite_eligible_need3', lambda: True)
    check('draft_limit192_alone_leaves_actual_grants_unchanged', lambda: True)
    observations['shared_original_fixture'] = dict(gap5=gap5, gap3=gap3,
        default=a, draft_limit192=b, all20B5_C=100, all20B5_allowed_under96=False,
        runtime_eligibility_or_window_qualified=False, actual_workload_observation=False)
    # In the current combined arm N>16 has normal cap4. Re-use a gap that
    # actually fits B3 to demonstrate the cap branch independently of unfit.
    family_states = {i: fixture_state(i, gap3) for i in range(64)}
    f20 = traced_allocate(family(model), tuple(range(20)), (20,), family_states, 192)
    assert max(f20['actual_original_locals']['caps'][i] for i in range(20)) == 4
    assert f20['eager_C'] > 0
    check('current_familycap_N20_is_not_allB5_and_can_leave_eager_work', lambda: True)
    f32 = traced_allocate(family(model), tuple(range(32)), tuple(range(32,64)), family_states, 192)
    assert f32['normal_C'] == 96 and f32['eager_C'] == 0
    check('familycap_N32_needs3_can_still_exhaust_shared96', lambda: True)
    observations['familycap_fixtures'] = dict(N20=f20, N32=f32,
        actual_workload_observation=False)

    rhythm_source = (NATIVE/'spec_rhythm.py').read_text()
    plan_ast = node(rhythm_source, 'SpecRhythmBudgetPlan', ast.ClassDef)
    OldPlan = execute_nodes(NATIVE/'spec_rhythm.py', [plan_ast])['SpecRhythmBudgetPlan']
    old_kwargs = dict(plan_id=1, normal_budgets=a['normal'], eager_budgets={20:3},
        progress_gaps={}, eager_priorities={}, verification_roof=96,
        draft_token_budget=192, allocated_draft_tokens=99)
    observations['old_contract_rejection'] = check('actual_Native_class_rejects96plus3_with_draft192',
        lambda: reject(lambda: OldPlan(**old_kwargs), 'global target verification roofline'))
    forged = OldPlan(**{**old_kwargs, 'verification_roof':192})
    assert forged.verification_input_tokens == 192
    check('inflating_old_verification_roof_also_falsely_advertises_target192', lambda: True)
    observations['forbidden_shortcut'] = dict(verification_roof=forged.verification_roof,
        verification_input_tokens=forged.verification_input_tokens, recommended=False)

    wrapper_ast = node((P/'tree_entry.py').read_text(), 'shape', ast.FunctionDef)
    wrapper_policy = original(model)
    wrapper = execute_nodes(P/'tree_entry.py', [wrapper_ast],
        dict(policy=wrapper_policy, rhythm=SimpleNamespace(SpecRhythmBudgetPlan=OldPlan)))['shape']
    kw = dict(plan_id=2, normal_request_indices=[0], eager_request_indices=[1],
        states={i:fixture_state(i,gap5) for i in range(2)}, projected_wait_ms=0.,
        context_len=200, batch_size=2, draft_token_budget=192,
        eager_token_cap=3, eager_reserve_tokens=3, verification_roof=3)
    wrapped = wrapper(None, **kw)
    assert dict(wrapped.eager_budgets) == {1:5} and wrapped.verification_roof == 96
    check('actual_wrapper_ignores_caller_eager3_reserve3_and_roof3', lambda: True)
    zero = wrapper(None, **{**kw, 'draft_token_budget':0})
    assert zero.allocated_draft_tokens == 0 and zero.draft_token_budget == 192
    check('actual_wrapper_explicit_zero_reports192_but_grants_zero', lambda: True)
    # The API rejects an explicitly configured zero; this is an isolated
    # wrapper semantic counterexample, not a claim about legal production cfg.
    reads = set()
    for child in ast.walk(wrapper_ast):
        if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name) and child.value.id == 'kwargs' and isinstance(child.slice, ast.Constant):
            reads.add(child.slice.value)
        if isinstance(child, ast.Call) and ast.unparse(child.func) == 'kwargs.get' and child.args and isinstance(child.args[0], ast.Constant):
            reads.add(child.args[0].value)
    runtime_tree = ast.parse((P/'runtime_tree_decode.py').read_text())
    shape_calls = [c for c in ast.walk(runtime_tree) if isinstance(c, ast.Call) and ast.unparse(c.func) == 'shaper.shape']
    assert len(shape_calls) == 1
    passed = {k.arg for k in shape_calls[0].keywords}
    ignored = sorted(passed-reads)
    assert ignored == ['batch_size','context_len','eager_reserve_tokens','eager_token_cap','verification_roof']
    observations['wrapper_actual_AST'] = dict(keyword_reads=sorted(reads), runtime_passed=sorted(passed),
        passed_but_not_read=ignored, granted_eager5_with_caller_cap3=dict(wrapped.eager_budgets),
        returned_roof_with_caller3=wrapped.verification_roof,
        explicit_zero_reported_draft_budget=zero.draft_token_budget,
        explicit_zero_API_legal=False, installed_wrapper_modified=False)

    # Evaluate the actual HOST refinement cap statements and publication gamma
    # assignment, without executing any tensor reads in the surrounding loop.
    assigns = [c for c in ast.walk(runtime_tree) if isinstance(c, ast.Assign)]
    cap_first = next(c for c in assigns if ast.unparse(c) == 'cap = runtime.max_gamma or shaper.max_gamma')
    cap_eager = next(c for c in ast.walk(runtime_tree) if isinstance(c, ast.If)
        and ast.unparse(c.test) == 'work_eager[row_id]' and any(
            isinstance(s, ast.Assign) and ast.unparse(s) == 'cap = min(cap, effective_eager_cap)' for s in c.body))
    gamma_assign = next(c for c in assigns if ast.unparse(c) == 'ticket.gamma = len(row)')
    refinement = execute_nodes(P/'runtime_tree_decode.py', [cap_first, cap_eager],
        dict(runtime=SimpleNamespace(max_gamma=5),shaper=SimpleNamespace(max_gamma=8),
             work_eager=[True],row_id=0,effective_eager_cap=3))
    selected_cap = refinement['cap']
    ticket = SimpleNamespace(gamma=5)
    execute_nodes(P/'runtime_tree_decode.py', [gamma_assign],dict(ticket=ticket,row=[11,12,13]))
    assert selected_cap == ticket.gamma == 3
    check('actual_eager_refinement_and_published_ticket_still_clip_to3', lambda: True)
    observations['downstream_cap'] = dict(work_grant_B=5, selected_max_candidates=selected_cap,
        published_ticket_gamma=ticket.gamma, statement_AST=[ast.unparse(cap_first),ast.unparse(cap_eager),ast.unparse(gamma_assign)],
        tensor_execution=False)

    budget_source = (NATIVE/'tree_budget.py').read_text()
    classes = [node(budget_source,name,ast.ClassDef) for name in ('TreeCandidateRequest','GlobalTreeCandidateSelection')]
    types = execute_nodes(NATIVE/'tree_budget.py',classes)
    backbone_ast = node((P/'backbone_selection.py').read_text(),'select',ast.FunctionDef)
    imports = [ast.unparse(s) for s in backbone_ast.body if isinstance(s,ast.ImportFrom)]
    assert imports == ['from vllm_ascend.spec_decode.pearl.tree_budget import GlobalTreeCandidateSelection']
    backbone_ast.body = [s for s in backbone_ast.body if not isinstance(s,ast.ImportFrom)]
    from math import prod
    selector = execute_nodes(P/'backbone_selection.py',[backbone_ast],
        dict(PARENTS=(-1,0,1,2,-1,0,1,2),prod=prod,
             GlobalTreeCandidateSelection=types['GlobalTreeCandidateSelection']))['select']
    requests = {i:types['TreeCandidateRequest'](parents=(-1,0,1,2,-1,0,1,2),
        token_ids=tuple(range(8)),conditional_confidences=(.8,)*8,max_candidates=3,
        minimum_candidates=1 if i<32 else 0) for i in range(33)}
    observations['selector_old_rejection'] = check('actual_backbone_selector_also_rejects99_under96',
        lambda: reject(lambda:selector(requests,96),'Preallocated candidates exceed'))
    normal_selection = selector({i:r for i,r in requests.items() if i<32},96)
    eager_selection = selector({i:r for i,r in requests.items() if i>=32},96)
    assert normal_selection.total_candidates == 96 and eager_selection.total_candidates == 3
    assert set(normal_selection.selected_indices).isdisjoint(eager_selection.selected_indices)
    assert all(v == (0,1,2) for v in [*normal_selection.selected_indices.values(),*eager_selection.selected_indices.values()])
    check('original_backbone_per_role96_preserves_all_same_selected_nodes',lambda:True)
    observations['per_role_selector_control'] = dict(normal_selected=96,eager_selected=3,
        combined_selected=99,each_original_selector_limit=96,
        recommended_new_typed_merge=True,old_selection_verification_budget_192_used=False,
        removed_import_for_HOST_only=imports,real_graph_execution=False)

    valid = SplitDraftWorkAudit(normal=tuple((i,3) for i in range(32)),
        future_eager=tuple((i,3) for i in range(32,64)),window_eager_rows=32)
    assert not hasattr(valid,'verification_roof')
    check('proposed_typed_validator_accepts96normal96future192selected64rows', lambda:True)
    target = valid.target_limit.check([3]*32)
    check('current_target_still_C96_Q128_N32',lambda:True)
    negatives = {
        'normal97':lambda: replace(valid,normal=((0,97),)),
        'eager97':lambda: replace(valid,future_eager=tuple((i,3) for i in range(32,64))+((64,1),),window_eager_rows=33),
        'eager_B4':lambda: replace(valid,future_eager=((32,4),)),
        'global191':lambda: replace(valid,global_selected_candidate_limit=191),
        'caller191':lambda: replace(valid,caller_selected_candidate_limit=191),
        'caller0':lambda: replace(valid,caller_selected_candidate_limit=0),
        'overlap':lambda: replace(valid,future_eager=((0,3),)),
        'unadmitted_window31':lambda: replace(valid,window_eager_rows=31),
        # Increase role bounds solely to isolate physical64 rejection.
        'physical65':lambda: replace(valid,normal=tuple((i,1) for i in range(33)),
            normal_row_limit=33,future_eager=tuple((i,1) for i in range(33,65))),
        'target_C97_Q129':lambda: valid.target_limit.check([4]+[3]*31),
        'target_N33':lambda: valid.target_limit.check([1]*33),
        'target_Q129_without_C_overflow':lambda: TargetAdmissionLimit(candidates=128).check([4]+[3]*31),
    }
    negative_results = {name:check('typed_contract_rejects_'+name,lambda f=callback:reject(f))
        for name,callback in negatives.items()}
    observations['proposed_diagnostic_contract'] = dict(valid=asdict(valid), unchanged_current_target=target,
        rejected_cases=negative_results, installed=False, executable_scheduler=False,
        explored_row_ABI_units=64*8, global192_units='retained/selected candidates',
        original_epoch_parent_frontier_ledger_checks_still_required=True)
    after = {str(p):sha(p) for p in paths}
    assert before == after
    check('all_explicit_source_hashes_unchanged',lambda:True)
    assert not any(name == 'torch' or name.startswith(('torch.', 'torch_npu', 'vllm')) for name in sys.modules)
    check('no_Torch_Native_import_model_or_device_use',lambda:True)
    report = dict(schema='V121.split-draft-target-source-feasibility.HOST.v1',
        created_UTC=datetime.now(timezone.utc).isoformat(),status='HOST_SOURCE_FEASIBILITY_ONLY',
        argv=sys.argv,python=sys.executable,python_version=sys.version,
        elapsed_seconds=time.perf_counter()-started,tests_run=len(tests),tests=tests,
        source_hashes=before,V116_frozen_budgeting_core_matches_V119=core_binding,
        source_unchanged=True,observations=observations,
        NPU_started=False,existing_frozen_or_Native_sources_modified=False,
        V118_actual_eager_gate_data_read=False,current_low_eager_root_cause_proved=False,
        numerical_qualification_inherited=False,performance_or_Goodput_claim=False,
        no_runtime_candidate_or_freeze_created=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),status=report['status'],
        tests_run=len(tests),elapsed_seconds=report['elapsed_seconds']),ensure_ascii=False))


if __name__ == '__main__':
    main()
