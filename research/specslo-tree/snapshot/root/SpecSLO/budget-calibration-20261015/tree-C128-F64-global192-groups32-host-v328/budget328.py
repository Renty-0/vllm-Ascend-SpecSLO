"""HOST proposal: normal128 + future64, global192, N32 and32 physical groups.

This is capacity validation only. Branch ROI needs measurements of the165
backend before this proposal can be integrated into an actual experiment.
"""
import ast
import copy
import inspect
import textwrap
import types

import budget290 as previous
from budget260 import altered

_TYPES = None
_SHAPE = None
_BOUND = None
_SHAPE_CALL = None
_CONTROLLER_FACTORY = None


def limit_classes():
    global _TYPES
    if _TYPES is not None:
        return _TYPES
    import split_budget_v132 as original
    classes = []
    for name in ('TargetAdmissionLimit', 'DraftWorkLimit'):
        node = ast.parse(textwrap.dedent(inspect.getsource(getattr(original, name)))).body[0]
        edits = []
        if name == 'DraftWorkLimit':
            values = dict(normal_candidates=128, future_eager_candidates=64, normal_per_request=8)
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and ast.unparse(field.target) in values:
                    edits.append((field, 'value', ast.Constant(values[ast.unparse(field.target)])))
            for keyword in ast.walk(node):
                if isinstance(keyword, ast.keyword) and keyword.arg in values:
                    edits.append((keyword, 'value', ast.Constant(values[keyword.arg])))
            assert len(edits) == 6
        else:
            values = dict(candidates=128, queries=160)
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and ast.unparse(field.target) in values:
                    edits.append((field, 'value', ast.Constant(values[ast.unparse(field.target)])))
            for comparison in ast.walk(node):
                if isinstance(comparison, ast.Compare) and ast.unparse(comparison.left) in ('self.candidates', 'self.queries'):
                    key = ast.unparse(comparison.left).split('.')[1]
                    assert len(comparison.comparators) == 1
                    edits.append((comparison, 'comparators', [ast.Constant(values[key])]))
            assert len(edits) == 4
        source = altered(node, edits, 'explicit128normal64future_global192_or160queries')
        namespace = dict(vars(original))
        exec(compile(source, __file__ + '::private-' + name, 'exec'), namespace)
        classes.append(namespace[name])
    _TYPES = tuple(classes)
    return _TYPES


def work_limit_class():
    return limit_classes()[1]


def eligible(policy, states, normal):
    from split_budget_v132 import PRIORITY
    spare = max(0, min(12, 32 - len(normal), (128 - 4 * len(normal)) // 4))
    ordered = sorted(normal, key=lambda i: (
        PRIORITY[states[i].slo_class], max(0., policy.gain(states[i], 8) - policy.gain(states[i], 4)),
        float(states[i].projected_progress_gap(0.)), -i), reverse=True)
    return frozenset([i for i in ordered if policy.gain(states[i], 8) > policy.gain(states[i], 4)][:spare])


def prepared_shape():
    global _SHAPE
    if _SHAPE is None:
        template = previous.prepared_shape()
        allocator = template.__globals__['shape_draft_work']
        private = types.FunctionType(
            allocator.__code__, dict(allocator.__globals__, eligible=eligible,
                                     TargetAdmissionLimit=limit_classes()[0]),
            allocator.__name__, allocator.__defaults__, allocator.__closure__)
        _SHAPE = types.FunctionType(
            template.__code__, dict(template.__globals__, DraftWorkLimit=work_limit_class(),
                                    shape_draft_work=private), template.__name__,
            template.__defaults__, template.__closure__)
    return _SHAPE


def prepared_bound_shape(function):
    global _BOUND
    if _BOUND is None:
        template = previous.prepared_bound_shape
        _BOUND = types.FunctionType(template.__code__, dict(template.__globals__, _BOUND={},
            prepared_shape=prepared_shape, work_limit_class=work_limit_class),
            template.__name__, template.__defaults__, template.__closure__)
    return _BOUND(function)


def shape(shaper, **kwargs):
    global _SHAPE_CALL
    if _SHAPE_CALL is None:
        template = previous.shape
        _SHAPE_CALL = types.FunctionType(template.__code__,
            dict(template.__globals__, prepared_bound_shape=prepared_bound_shape),
            template.__name__, template.__defaults__, template.__closure__)
    return _SHAPE_CALL(shaper, **kwargs)


def controller_function(function):
    global _CONTROLLER_FACTORY
    if _CONTROLLER_FACTORY is None:
        import budget260
        original = ast.parse(textwrap.dedent(inspect.getsource(budget260.controller_function)))
        projected = copy.deepcopy(original)
        edits = []
        for node in ast.walk(projected):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.replace('value == 5', 'value > 4')
                value = value.replace('candidate_counts[index] == 5', 'candidate_counts[index] > 4')
                value = value.replace('> 24', '> 32').replace('exceeds24', 'exceeds32')
                if value != node.value:
                    edits.append((node, node.value))
                    node.value = value
        assert len(edits) == 4
        code = ast.unparse(ast.fix_missing_locations(projected)) + '\n'
        for node, value in edits:
            node.value = value
        assert ast.dump(projected) == ast.dump(original)
        namespace = dict(vars(budget260), _CONTROLLERS={})
        exec(compile(code, __file__ + '::original-controller-factory-groups32', 'exec'), namespace)
        _CONTROLLER_FACTORY = namespace['controller_function']
    return _CONTROLLER_FACTORY(function)


def build_plan(controller, *args, **kwargs):
    result = controller_function(type(controller).build_plan)(controller, *args, **kwargs)
    budgets = result.target_candidate_budgets
    if len(budgets) > 32 or len(budgets) + sum(b > 4 for b in budgets.values()) > 32:
        raise RuntimeError('ROOT314 whole-ticket physical group bound')
    if sum(budgets.values()) > 128 or len(budgets) + sum(budgets.values()) > 160:
        raise RuntimeError('ROOT314 actual normal128 root-inclusive160')
    return result


filter_eager_parents = previous.filter_eager_parents
