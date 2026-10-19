"""Fresh same-instance draft qualification, derived from frozen probe371.

All306 neural cases remain. The separate component's timing loop is optional;
the full-worker qualification does not measure or inherit component costs.
New private entries must be naturally qualified; pre-existing entries continue
through the unchanged original warm/two-zero/seal requirements.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / 'tree-primary4-tail3-eager-allN-device-v371/probe371.py'
SOURCE_SHA = 'e9f4bb1826c47572a3368a2e21ffe25627adaeae2e4f65baa0e7e6180c007f37'


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT372 draft qualification: ' + message)


def new_entries(runner):
    from runner369 import MARKER
    return {key: entry for key, entry in runner.draft_entries.items() if MARKER in key[0]}


def new_entries_qualified(runner, state):
    entries = new_entries(runner)
    if state.runner.runner is not runner or len(entries) != 64:
        return False
    return all(entry.runtime_validated is True and key not in runner.disabled_entry_keys
               and int(entry.validated_real_row_count) == int(key[1])
               for key, entry in entries.items())


def project_body():
    need(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA, 'immutable371 original neural source')
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    bodies = [n for n in main.body if isinstance(n, ast.With)]
    need(len(bodies) == 1, 'one complete original inference body')
    body = copy.deepcopy(bodies[0])
    original = copy.deepcopy(body)
    costs = [(i, n) for i, n in enumerate(body.body) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'n'
             and isinstance(n.iter, ast.Tuple) and ast.literal_eval(n.iter) == (4, 12, 24, 32)]
    need(len(costs) == 1, 'one original component timing loop')
    cost_index, cost_loop = costs[0]
    body.body[cost_index] = ast.If(test=ast.Name(id='measure_costs', ctx=ast.Load()), body=[cost_loop], orelse=[])
    summaries = [n for n in body.body if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'summary']
    need(len(summaries) == 1, 'one original measured-cost summary')
    summary = summaries[0]
    old_summary = copy.deepcopy(summary.value)
    summary.value = ast.IfExp(test=ast.Name(id='measure_costs', ctx=ast.Load()),
                              body=copy.deepcopy(old_summary), orelse=ast.Dict(keys=[], values=[]))
    checks = [n for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == 'need' and len(n.args) == 2 and isinstance(n.args[1], ast.Constant)
              and n.args[1].value == 'all resident Native qualification intact']
    need(len(checks) == 1, 'one original standalone graph inventory assertion')
    check = checks[0]
    old_check = copy.deepcopy(check.args[0])
    check.args[0] = ast.parse('_ROOT372_new_entries_qualified(engine.graph_runner, state)', mode='eval').body
    stage_names = [n for n in ast.walk(body) if isinstance(n, ast.Constant)
                   and n.value == 'all306_math_passed_start_balanced_cost']
    need(len(stage_names) == 1, 'one original timing-stage label')
    stage_names[0].value = 'all306_same_instance_math_passed_optional_costs'
    projected = copy.deepcopy(body)
    body.body[cost_index] = cost_loop
    summary.value = old_summary
    check.args[0] = old_check
    stage_names[0].value = 'all306_math_passed_start_balanced_cost'
    need(ast.dump(body, include_attributes=False) == ast.dump(original, include_attributes=False),
         'entire original371 inference-body AST inverse')
    return ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[]))


def qualify(engine, native, directory):
    import torch
    need(engine.is_draft is True and engine.cache_allocation is None and engine.cache_block_tables is None,
         'actual original draft instance with no live cache lease')
    need(not engine.graph_runner.graph_cache_sealed, 'qualification before original graph sealing')
    model = engine.model
    need(len(model.layers) == 28 and all(p.dtype == torch.bfloat16 for p in model.parameters() if p.is_floating_point()),
         'actual28-layer original BF16 draft')
    need(engine.prefix_cache.num_blocks == 544 and engine._cache_storage_sequence_capacity == 527,
         'original full257 pool544/storage527 retained')
    for name in ('tree-primary4-tail3-eager-allN-device-v371', 'tree-primary4-tail3-top2-chain-host-v369',
                 'tree-primary4-top2-chain-device-v365', 'tree-M160-S32-N32-target-math-cost-v326',
                 'tree-high-normal-cap4-native-families-v148'):
        path = str(BASE / name)
        if path not in sys.path:
            sys.path.append(path)
    import probe371
    from forward369 import build
    from bank326 import allocate
    from dual_bank_v146 import prefix_evidence
    from probe365 import compare_kv
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    original = native.NativePearlEngine.draft_tree_forward
    proposed, state = build(original, engine)
    folder = Path(directory)
    folder.mkdir(exist_ok=False)
    prefixes = json.loads(probe371.FIXTURE.read_text())['prefixes']
    previous_policy = getattr(engine, '_defer_tree_materialization', None)
    had_policy = hasattr(engine, '_defer_tree_materialization')
    engine._defer_tree_materialization = True
    try:
        prompts, allocation = allocate(engine, prefixes)
        namespace = dict(vars(probe371), engine=engine, native=native, model=model,
            original=original, proposed=proposed, state=state, torch=torch,
            prefixes=prefixes, prompts=prompts, allocation=allocation, roots=[], rows=[], costs=[],
            prefix_evidence=prefix_evidence, compare_kv=compare_kv,
            cached_cpu_tree_speculation_plan=cached_cpu_tree_speculation_plan,
            output=folder / 'rank0-original371-neural-body.json', measure_costs=False,
            _ROOT372_new_entries_qualified=new_entries_qualified)
        exec(compile(project_body(), __file__ + '::original371-qualification-only', 'exec'), namespace)
        result = namespace['value']
        need(len(result['math_rows']) == 306 and result['costs'] == [] and result['median_seconds'] == {},
             'all306 fresh neural cases and no inherited or invented component cost')
        need(result['all_committed_prefix_KV_protected'] is True and new_entries_qualified(engine.graph_runner, state),
             'fresh committed prefixes and all64 naturally qualified private entries')
        need(engine.cache_allocation is None and engine.cache_block_tables is None,
             'fresh neural cache bank released before original LIVE warm')
        proof = dict(schema='ROOT372.same-instance-draft-tail3-proof.v1',
            source371_sha256=SOURCE_SHA, original_body_result_sha256=hashlib.sha256(
                namespace['output'].read_bytes()).hexdigest(), actual_engine_rank=engine.rank,
            cases=306, fresh_same_model_instance=True, all_new_private_entries_naturally_qualified=True,
            private_entries=len(new_entries(engine.graph_runner)), costs_measured=False,
            original_warm_twozero_seal_required_for_all_entries=True,
            native_qualification_flag_assignments=False, model_qualified_for_LIVE=False,
            whole463_qualified=False, Raw=None, Goodput=None, adopted=False)
        with (folder / 'rank0-tail3-same-instance-proof.json').open('x') as saved:
            json.dump(proof, saved, indent=2)
            saved.write('\n')
        return dict(proof=proof, state=state, original=original, proposed=proposed,
                    entries=new_entries(engine.graph_runner), directory=str(folder))
    finally:
        if had_policy:
            engine._defer_tree_materialization = previous_policy
        elif hasattr(engine, '_defer_tree_materialization'):
            del engine._defer_tree_materialization
