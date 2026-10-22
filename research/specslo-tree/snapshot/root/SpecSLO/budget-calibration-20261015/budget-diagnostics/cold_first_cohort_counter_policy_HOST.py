"""Uninstalled measured first-cohort coalescing candidate, stdlib only.

The server may defer initial ready admission to fill a batch. All wait stays
inside the original generation wall; client arrivals and later cohorts stay
unchanged. The original coalesce setting is separate and remains disabled.
"""
import ast
import copy

FIELDS = dict(spec_rhythm_cold_first_cohort_minimum_requests=8,
              spec_rhythm_cold_first_cohort_maximum_wait_ms=1000.0,
              spec_rhythm_cold_first_cohort_deferred_polls=0,
              spec_rhythm_cold_first_cohort_released_requests=0,
              spec_rhythm_cold_first_cohort_wait_observed_ms=0.0)
BLOCK = '''if arrival_gated and not active and not completed and admitted:
    if (len(admitted) < 8 and len(pending_admission) >= 8
            and not token_budget_saturated
            and all(request_params[i].arrival_ts is not None for i in admitted)):
        _root_cold_oldest = min(request_params[i].arrival_ts for i in admitted)
        if 0.0 <= (now - _root_cold_oldest) * 1000.0 < 1000.0:
            counters['spec_rhythm_cold_first_cohort_deferred_polls'] += 1
            admitted = []
    if admitted:
        counters['spec_rhythm_cold_first_cohort_released_requests'] = len(admitted)
        _root_cold_arrivals = [request_params[i].arrival_ts for i in admitted]
        if all(t is not None for t in _root_cold_arrivals):
            counters['spec_rhythm_cold_first_cohort_wait_observed_ms'] = max(0.0, (now - min(_root_cold_arrivals)) * 1000.0)
'''


def need(value, message):
    if not value:
        raise ValueError('cold first cohort: ' + message)


def same(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def locate(tree):
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_generate_spec_rhythm_tree_decode']
    need(len(main) == 1, 'one original decode')
    admission = [n for n in ast.walk(main[0]) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available']
    counters = [n for n in ast.walk(main[0]) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'counters']
    need(len(admission) == len(counters) == 1, 'one original admission and counter initializer')
    return admission[0], counters[0].value


def block():
    return ast.parse(BLOCK).body[0]


def patch(source):
    tree = ast.parse(source)
    admission, counters = locate(tree)
    need(all(k.value not in FIELDS for k in counters.keys if isinstance(k, ast.Constant)), 'no existing policy counters')
    for key, value in FIELDS.items():
        counters.keys.append(ast.Constant(key))
        counters.values.append(ast.Constant(value))
    anchors = [i for i,n in enumerate(admission.body) if isinstance(n,ast.If)
               and ast.unparse(n.test) == 'admitted and prefill_coalescing']
    need(len(anchors) == 1 and isinstance(admission.body[anchors[0]-1], ast.For), 'original ready-only cohort selected before original coalesce')
    admission.body.insert(anchors[0], block())
    result = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    need(same(ast.parse(source), ast.parse(unpatch(result))), 'complete original AST inverse')
    return result


def unpatch(source):
    tree = ast.parse(source)
    admission, counters = locate(tree)
    blocks = [i for i,n in enumerate(admission.body) if same(n, block())]
    need(len(blocks) == 1, 'one exact added cold-cohort decision')
    del admission.body[blocks[0]]
    for key, value in reversed(tuple(FIELDS.items())):
        need(counters.keys[-1].value == key and counters.values[-1].value == value, 'exact added counter '+key)
        counters.keys.pop()
        counters.values.pop()
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def install_builder(entry, state):
    """Private namespace binding only, no canonical module mutation."""
    need('_cold_first_cohort_installed' not in entry, 'one private builder binding')
    if '_live_builder' in entry:
        original = entry['_live_builder']
    else:
        import live_decode_patch
        original = live_decode_patch.build_live_runtime
    need(callable(original), 'original callable live builder')
    state['cold_first_cohort_source_builds'] = 0
    def build():
        source = original()
        result = patch(source)
        state['cold_first_cohort_source_builds'] += 1
        return result
    entry['_live_builder'] = build
    entry['_cold_first_cohort_installed'] = True


def evaluate(env):
    """Execute exactly the inserted decision in HOST tests only."""
    namespace = dict(env)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(block())],type_ignores=[])),
                 '<cold-first-cohort-counter-policy-HOST>', 'exec'), namespace)
    return namespace['admitted']
