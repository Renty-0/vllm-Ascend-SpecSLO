"""Preserve all594 original344 math cases and retain each finished cost panel.

Adds a balanced FULL2560/bucket cost panel only after that bank's original
math and budget costs. No device result or qualification is inherited.
"""
import ast
import copy
import inspect

import probe344 as original
from benchmark361 import run as paired_modes361


def projected_run():
    source = inspect.getsource(original.run)
    tree = ast.parse(source)
    before = copy.deepcopy(tree)
    loop = next(node for node in tree.body[0].body if isinstance(node, ast.For))
    index = next(i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and ast.unparse(node.value.func) == 'reports.append') + 1
    extra = ast.parse('''
mode_cost_start = len(engine._branch_tail_state.static_graph_calls)
mode_costs = paired_modes361(engine, native, lane, bank)
mode_full_calls = engine._branch_tail_state.static_graph_calls[mode_cost_start:]
mode_compact_calls = [{key:call[key] for key in compact_keys} for call in mode_full_calls]
mode_call_sha = hashlib.sha256(json.dumps(mode_full_calls, sort_keys=True).encode()).hexdigest()
del engine._branch_tail_state.static_graph_calls[mode_cost_start:]
mode_prefixes_intact = lane('all72-prefixes-after-paired-KV-modes361', protected)
reports[-1]['paired_KV_modes361'] = mode_costs
reports[-1]['actual_KV_mode_calls361'] = mode_compact_calls
reports[-1]['reconstructable_KV_mode_call_SHA361'] = mode_call_sha
reports[-1]['all72_prefixes_after_KV_modes361'] = mode_prefixes_intact
lane('retain-complete-cost-panel361', lambda: save(folder/f'rank{engine.rank}-cost-and-protection361.json',
    dict(rank=engine.rank, bank=name, original_budget_cost=costs, paired_KV_modes361=mode_costs,
         actual_KV_mode_calls361=mode_compact_calls, reconstructable_KV_mode_call_SHA361=mode_call_sha,
         all72_prefixes_after_original_cost=intact,
         all72_prefixes_after_KV_modes361=mode_prefixes_intact,
         original_math_completed=True, full594_completed=False,
         real_draft_generation=False, whole_workload_AR_qualified=False, performance=False, Goodput=None)))
''').body
    loop.body[index:index] = extra
    projected = copy.deepcopy(tree)
    del loop.body[index:index + len(extra)]
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('ROOT361 complete original344 finite-probe source inverse')
    namespace = dict(vars(original), paired_modes361=paired_modes361)
    exec(compile(ast.fix_missing_locations(projected), __file__ + '::original344-plus-postmath-paired-mode-panels',
                 'exec'), namespace)
    return namespace['run']


run = projected_run()
