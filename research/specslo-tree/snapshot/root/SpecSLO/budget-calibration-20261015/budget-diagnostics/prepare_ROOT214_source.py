"""Prepare a separate candidate; never edit frozen experiments or launch devices."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
DONOR = B / 'tree-cached-parent-slo-admission-full-v213'
P = B / 'tree-cost-aware-mixed16-budget-full-v214'
assert not P.exists(), 'fresh source directory required'
P.mkdir()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


records = []
for old in sorted(DONOR.iterdir()):
    if old.suffix not in ('.py', '.sh'):
        continue
    new = P / old.name.replace('213', '214')
    source = old.read_text()
    baseline = source.replace(str(DONOR), str(P)).replace('213', '214')
    new.write_text(baseline)
    records.append(dict(source=str(old), source_sha256=sha(old), new=str(new),
                        namespace_old='213', namespace_new='214',
                        path_old=str(DONOR), path_new=str(P)))


def change(name, before, after):
    p = P / name
    text = p.read_text()
    assert text.count(before) == 1, (name, before)
    p.write_text(text.replace(before, after))


change('extension214.py', 'return entry,Controls',
       "from cost214 import budget_controls\n    return entry,budget_controls(Controls,_STATE)")
change('extension214.py', "SLO_AWARE_ADMISSION=True,cached_source_precheck=True,",
       "SLO_AWARE_ADMISSION=True,cached_source_precheck=True,cost_aware_mixed16_budget=True,")
change('extension214.py', "ROOT214_overlay='cached-parent-and-SLO-admission1'",
       "ROOT214_overlay='cached-parent-SLO-admission1-cost-aware-mixed16'")
change('extension214.py', "new_model_or_graph_or_budget=False,",
       "new_model_or_graph=False,budget_grants_may_change=True,")
change('guard214.py', "r['overlay']['ROOT214_overlay']=='cached-parent-and-SLO-admission1'",
       "r['overlay']['ROOT214_overlay']=='cached-parent-SLO-admission1-cost-aware-mixed16'")
change('host214.py', "ROOT214_overlay='cached-parent-and-SLO-admission1'",
       "ROOT214_overlay='cached-parent-SLO-admission1-cost-aware-mixed16'")
change('guard214.py', 'changed_model_or_budget_or_arrival_or_clock=False,',
       'changed_model_or_arrival_or_clock=False,budget_grants_may_change=True,')
change('guard214.py', "production_delta='cached mapper from211 plus original SLO-aware-admission1 from203; unchanged clients/model/budget/clock'",
       "production_delta='same213 plus budget-cost repair: mixed16 B3/B4 to allB4 only within original limits; clients/model/clock unchanged'")
change('guard214.py', 'original_budget_and_arrivals=True,',
       'original_budget_limits_and_arrivals=True,cost_aware_mixed16_budget=True,')
change('guard214.py', 'cached mapper plus original SLO-aware-admission1',
       'cached mapper plus original SLO-aware-admission1 and cost-aware mixed16 budget')
change('guard214.py', 'snapshots.append(cache)',
       "from cost214 import audit_snapshot\n        audit_snapshot(cache, trace['source_generation'])\n        require(r['overlay']['cost_aware_mixed16_budget']is True,'actual cost-aware overlay')\n        snapshots.append(cache)")

P.joinpath('cost214.py').write_text('''"""Budget units remain retained candidates; no new Native graph or math path."""
from collections import Counter
from dataclasses import replace
import types


def repair_plan(policy, plan, *, context_len, counts):
    counts['shape_calls'] += 1
    normal = plan.normal_budgets
    if (len(plan.requested_normal) != 16 or len(normal) != 16
            or set(normal) != set(plan.requested_normal)
            or set(normal.values()) != {3, 4}):
        return plan
    counts['eligible_mixed16'] += 1
    extra = sum(4 - b for b in normal.values())
    allowed = (type(context_len) is int and context_len + 4 <= 4096
               and all(plan.caps[i] >= 4 for i in normal)
               and plan.work_limit.normal_per_request >= 4
               and 64 <= min(plan.work_limit.normal_candidates, plan.target_limit.candidates)
               and 16 <= plan.target_limit.rows and 80 <= plan.target_limit.queries
               and plan.allocated_selected_candidates + extra <= plan.available_selected_candidates)
    if not allowed:
        counts['blocked_by_original_limits'] += 1
        return plan
    upgraded = {i: 4 for i in normal}
    # Original immutable DTO reruns every cap, role, reserve and ownership check.
    result = replace(plan, normal_budgets=upgraded)
    for i, before in normal.items():
        if before == 3:
            if policy.counts['grant_B3'] <= 0:
                raise RuntimeError('ROOT214 original grant counters must precede cost repair')
            policy.counts['grant_B3'] -= 1
            policy.counts['grant_B4'] += 1
    counts['repaired_calls'] += 1
    counts['upgraded_owners'] += extra
    counts['added_selected_candidates'] += extra
    counts['proposed_target_physical_queries_saved'] += 48
    # Already max B4 before repair: neither draft stages nor eager filtering change.
    return result


def bind_candidate_shape(control, state):
    old = control.candidate_shape
    shape = old.__globals__['shape']
    original_solver = shape.__globals__['shape_draft_work']
    def solver(policy, shaper, **kwargs):
        plan = original_solver(policy, shaper, **kwargs)
        generation = getattr(control.engine, '_dynamic_tree_generation', None)
        if type(generation) is not int or generation < 0:
            raise RuntimeError('ROOT214 actual current engine generation required')
        counts = state['by_generation'].setdefault(str(generation), Counter())
        return repair_plan(policy, plan, context_len=kwargs['context_len'], counts=counts)
    private_shape = types.FunctionType(shape.__code__, dict(shape.__globals__, shape_draft_work=solver),
                                       shape.__name__, shape.__defaults__, shape.__closure__)
    control.candidate_shape = types.FunctionType(old.__code__, dict(old.__globals__, shape=private_shape),
                                               old.__name__, old.__defaults__, old.__closure__)
    if control.candidate_shape.__closure__ is not old.__closure__:
        raise RuntimeError('ROOT214 original policy closure changed')
    state['private_shape_bindings'] += 1
    return control.candidate_shape


def budget_controls(Parent, cache_state):
    state = dict(private_shape_bindings=0, by_generation={})
    cache_state['cost_aware_mixed16_budget'] = state
    class Controls(Parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.mode != 'original' or self.engine is not None:
                raise RuntimeError('ROOT214 bind only before original first engine selection')
            bind_candidate_shape(self, state)
    return Controls


def audit_snapshot(cache, full_generation):
    state = cache['cost_aware_mixed16_budget']
    if (state['private_shape_bindings'] != 1
            or str(full_generation) not in state['by_generation']
            or state['by_generation'][str(full_generation)]['shape_calls'] <= 0):
        raise RuntimeError('ROOT214 actual fresh full463 shape execution missing')
    for row in state['by_generation'].values():
        if (any(type(v) is not int or v < 0 for v in row.values())
                or row.get('eligible_mixed16', 0) != row.get('repaired_calls', 0) + row.get('blocked_by_original_limits', 0)
                or row.get('added_selected_candidates', 0) != row.get('upgraded_owners', 0)
                or row.get('proposed_target_physical_queries_saved', 0) != 48 * row.get('repaired_calls', 0)):
            raise RuntimeError('ROOT214 retained repair accounting mismatch')
    return True
''')

tests = '''

class CostTests(unittest.TestCase):
    def plan(self, budgets=None, *, caps=None, available=192, normal_limit=96, eager=()):
        from collections import Counter
        from split_budget_v132 import DraftWorkPlan,DraftWorkLimit,TargetAdmissionLimit
        budgets=dict(enumerate([3,4]*8)) if budgets is None else dict(budgets)
        ids=tuple(budgets);all_ids=ids+tuple(eager)
        policy=types.SimpleNamespace(counts=Counter({f'grant_B{b}':list(budgets.values()).count(b) for b in range(1,6)}))
        eb={i:3 for i in eager}
        plan=DraftWorkPlan(plan_id=7,target_limit=TargetAdmissionLimit(),
            work_limit=DraftWorkLimit(normal_candidates=normal_limit,normal_per_request=4),
            available_selected_candidates=available,requested_normal=ids,admitted_eager=tuple(eager),
            normal_budgets=budgets,eager_budgets=eb,progress_gaps=dict.fromkeys(all_ids,1.),
            eager_priorities=dict.fromkeys(eager,1.),caps=dict.fromkeys(all_ids,4) if caps is None else caps,
            needs=dict.fromkeys(all_ids,1),reserve_requested=sum(eb.values()),reserve_used=sum(eb.values()),
            reserved_by_owner=eb,raw_reserved_by_owner=eb,raw_eager_budgets=eb)
        return policy,plan

    def test_original_typed_plan_and_budget_accounting(self):
        from collections import Counter
        from cost214 import repair_plan
        policy,plan=self.plan(eager=(16,17));counts=Counter()
        new=repair_plan(policy,plan,context_len=4000,counts=counts)
        self.assertEqual(list(plan.normal_budgets.values()),[3,4]*8)
        self.assertEqual(list(new.normal_budgets.values()),[4]*16)
        self.assertEqual(new.allocated_selected_candidates,70)
        self.assertEqual(dict(new.eager_budgets),dict(plan.eager_budgets))
        self.assertEqual(new.reserve_used,plan.reserve_used)
        self.assertEqual(dict(new.raw_reserved_by_owner),dict(plan.raw_reserved_by_owner))
        self.assertEqual(policy.counts['grant_B3'],0);self.assertEqual(policy.counts['grant_B4'],16)
        self.assertEqual(counts['added_selected_candidates'],8)
        for field in plan.__dataclass_fields__:
            if field!='normal_budgets':self.assertEqual(getattr(new,field),getattr(plan,field))
        self.assertTrue(new.require_work(list(range(18)),[4]*16+[3,3],[False]*16+[True,True],
            [types.SimpleNamespace(candidate_budget=b)for b in [4]*16+[3,3]]))

    def test_cap_context_role_and_global_limit_block(self):
        from collections import Counter
        from cost214 import repair_plan
        cases=[dict(caps={i:(3 if i==0 else 4)for i in range(16)}),dict(available=56),dict(normal_limit=56)]
        for kw in cases:
            with self.subTest(kw=kw):
                policy,plan=self.plan(**kw);counts=Counter();before=dict(policy.counts)
                self.assertIs(repair_plan(policy,plan,context_len=4092,counts=counts),plan)
                self.assertEqual(dict(policy.counts),before);self.assertEqual(counts['blocked_by_original_limits'],1)
        policy,plan=self.plan();counts=Counter()
        self.assertIs(repair_plan(policy,plan,context_len=4093,counts=counts),plan)

    def test_scope_excludes_shared3_homogeneous4_other_sizes_and_B2(self):
        from collections import Counter
        from cost214 import repair_plan
        for values in ([3]*16,[4]*16,[3,4]*7,[3,4]*8+[3], [2]+[4]*15):
            policy,plan=self.plan(dict(enumerate(values)));counts=Counter()
            self.assertIs(repair_plan(policy,plan,context_len=100,counts=counts),plan)
            self.assertEqual(counts['eligible_mixed16'],0)

    def test_exact_private_function_and_original_closure_retained(self):
        from collections import Counter
        from cost214 import bind_candidate_shape
        import split_shape_v132
        original=dict(vars(split_shape_v132));policy,plan=self.plan()
        def solver(p,s,**kw):self.assertIs(p,policy);return plan
        shape=types.FunctionType(split_shape_v132.shape.__code__,dict(vars(split_shape_v132),
            shape_draft_work=solver,select_normal=lambda *a,**k:types.SimpleNamespace(admitted=tuple(range(16))),
            finish_normal=lambda *a:None))
        ns={'shape':shape}
        exec('def factory(policy):\\n def candidate_shape(instance,**kwargs):return shape(policy,instance,**kwargs)\\n return candidate_shape',ns)
        old=ns['factory'](policy);control=types.SimpleNamespace(candidate_shape=old,engine=types.SimpleNamespace(_dynamic_tree_generation=5))
        state=dict(private_shape_bindings=0,by_generation={});new=bind_candidate_shape(control,state)
        self.assertIs(new.__closure__,old.__closure__);self.assertEqual(new.__code__.co_code,old.__code__.co_code)
        self.assertIs(new.__globals__['shape'].__code__,shape.__code__)
        self.assertIs(new.__globals__['shape'].__globals__['refine_draft_groups'],shape.__globals__['refine_draft_groups'])
        states={i:types.SimpleNamespace(max_gamma=5) for i in range(16)}
        value=new(types.SimpleNamespace(max_gamma=5),normal_request_indices=tuple(range(16)),eager_request_indices=(),states=states,context_len=4092)
        self.assertEqual(list(value.normal_budgets.values()),[4]*16)
        self.assertEqual(state['by_generation']['5']['repaired_calls'],1)
        self.assertTrue(all(vars(split_shape_v132)[k]is v for k,v in original.items()))

    def test_actual_mapper_cost_discontinuity(self):
        import mapper173
        self.assertEqual(mapper173.expected_geometry(16,80,'causal')[:2],(80,16))
        for q in range(65,80):self.assertEqual(mapper173.expected_geometry(16,q,'causal')[:2],(128,38))

    def test_actual_generation_and_invalid_accounting_rejected(self):
        from cost214 import audit_snapshot
        cache=dict(cost_aware_mixed16_budget=dict(private_shape_bindings=1,by_generation={'5':dict(shape_calls=12,eligible_mixed16=4,repaired_calls=3,blocked_by_original_limits=1,upgraded_owners=8,added_selected_candidates=8,proposed_target_physical_queries_saved=144)}))
        self.assertTrue(audit_snapshot(cache,5))
        with self.assertRaises(RuntimeError):audit_snapshot(cache,6)
        cache['cost_aware_mixed16_budget']['by_generation']['5']['added_selected_candidates']=9
        with self.assertRaises(RuntimeError):audit_snapshot(cache,5)
'''
host = P / 'host214.py'
text = host.read_text()
text = text.replace("if __name__=='__main__':", tests + "\n\nif __name__=='__main__':")
text = text.replace('loadTestsFromTestCase(Tests)', 'loadTestsFromModule(sys.modules[__name__])')
# The original terminal-loader HOST stub explicitly supplies fresh214 evidence.
text = text.replace("ROOT214_overlay='cached-parent-SLO-admission1-cost-aware-mixed16',SLO_AWARE_ADMISSION=True)",
                    "ROOT214_overlay='cached-parent-SLO-admission1-cost-aware-mixed16',SLO_AWARE_ADMISSION=True,cost_aware_mixed16_budget=True)")
text = text.replace("canonical_mapper_or_base_module_mutated=False),", "canonical_mapper_or_base_module_mutated=False,cost_aware_mixed16_budget=dict(private_shape_bindings=1,by_generation={'5':dict(shape_calls=1)})),")
text = text.replace("trace=dict(decode_counters=", "trace=dict(source_generation=5,decode_counters=")
host.write_text(text)

for item in records:
    old = Path(item['source']).read_text().replace(str(DONOR), str(P)).replace('213', '214')
    new = Path(item['new']).read_text()
    item['patch_after_namespace_projection'] = ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='projected213',tofile='candidate214'))
    item['new_sha256'] = sha(Path(item['new']))
    ast.parse(new) if Path(item['new']).suffix == '.py' else None
projection = dict(schema='ROOT214.source-projection-only.v1', donor=str(DONOR),
    donor_freeze_sha256=sha(DONOR/'FREEZE213.json'), projection=records,
    new_module=dict(path=str(P/'cost214.py'),sha256=sha(P/'cost214.py')),
    production_delta='mixed16 B3/B4 -> allB4 only within original limits; original outer shape bytecode and policy closure retained',
    source_peer_review=False, new_native_keys=0, new_NPU_ops=0, NPU_run=False,
    arrival_model_clock_changed=False, budget_grants_may_change=True)
(P/'PROJECTION214.json').write_text(json.dumps(projection,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='source_prepared_unfrozen',directory=str(P),files=len(records)+1,NPU_run=False)))
