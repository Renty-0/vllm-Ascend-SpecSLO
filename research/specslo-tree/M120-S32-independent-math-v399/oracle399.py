"""New S32 nodes checked against independently prefilled canonical S24 paths.

Retain the original numerical matchers and require exact hidden/KV bits too.
The extra hidden graph is also compared with eager over all120 physical rows.
This is a finite mathematical component, never serving or Goodput evidence.
"""
import ast
import copy
from pathlib import Path

import neural_oracle231 as original
from domain399 import cases
from oracle255 import prepare as prepare_original


def prepare(case, prefixes):
    return prepare_original(case, prefixes)


def specialized(name):
    source = Path(original.__file__).read_text()
    node = copy.deepcopy(next(n for n in ast.parse(source).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    for value in ast.walk(node):
        if isinstance(value, ast.Constant) and type(value.value) is int:
            if value.value == 80:
                value.value = 120
            elif name == 'run' and value.value == 16:
                value.value = 32
        elif isinstance(value, ast.ImportFrom):
            value.module = {'metadata231': 'metadata399',
                            'reference_schedule231': 'reference_schedule399',
                            'baseline234': 'baseline255',
                            'branch_tail231': 'branch_tail255'}.get(value.module, value.module)
        elif name == 'reference' and isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == 'canonical':
                value.func.id = '_checked_reference'
    if name == 'run':
        compare = next(n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.name == 'compare')
        index = next(i for i, item in enumerate(compare.body) if isinstance(item, ast.Return))
        compare.body[index:index] = ast.parse(
            "need(torch.equal(actual['hidden'], own['hidden']) and "
            "all(torch.equal(a,b) for a,b in zip(actual['KV'],own['KV'],strict=True)) "
            "and actual['graph_eager_KV_bitwise_equal'] and actual['allphysical_graph_eager_hidden_KV'], "
            "'ROOT399 exact cross-S32/S24 hidden and all128 KV plus fullphysical graph/eager')"
        ).body
        result = next(item.value for item in compare.body if isinstance(item, ast.Return))
        result.keywords.extend([
            ast.keyword(arg='actual_physical_S', value=ast.Constant(32)),
            ast.keyword(arg='actual_physical_M', value=ast.Constant(120)),
            ast.keyword(arg='independent_reference_S', value=ast.Constant(24)),
            ast.keyword(arg='allphysical_graph_eager_hidden_KV', value=ast.Subscript(
                value=ast.Name(id='actual', ctx=ast.Load()),
                slice=ast.Constant('allphysical_graph_eager_hidden_KV'), ctx=ast.Load())),
        ])
    from metadata399 import checked
    namespace = dict(vars(original), cases=cases, candidate=candidate, reference=reference,
                     _checked_reference=checked)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 __file__ + '::original-unchanged-matcher-' + name, 'exec'), namespace)
    return namespace[name]


def candidate(engine, case, bank, plans, rows):
    import torch
    from dual_bank_v146 import query_KV
    from projection397 import SEGMENT_CAPACITIES
    result = specialized('candidate')(engine, case, bank, plans, rows)
    current = engine._branch_tail_state.static_current
    if current is None or current['copy_enabled']:
        raise RuntimeError('ROOT399 actual finite S32 physical dispatch required')
    x, pos, metadata, inverse, mapping = current['physical']
    ends = []
    total = 0
    for n in SEGMENT_CAPACITIES:
        total += n
        ends.append(total)
    if not (x.numel() == 120 and len(metadata.sequence_lens) == 32
            and tuple(metadata.actual_seq_lengths_q) == tuple(ends)
            and tuple(metadata.tree_attention_mask.shape) == (32, 1, 10, 4096)
            and mapping['real_N'] == case['N']):
        raise RuntimeError('ROOT399 actual fixed M120/S32 q4-q3 metadata')
    graph = engine.graph_runner.run_tree_hidden(x, pos, metadata).clone()
    execution = engine.graph_runner.last_target_execution
    if not execution.used_aclgraph or execution.mode not in ('capture_replay', 'replay'):
        raise RuntimeError('ROOT399 naturally captured and validated actual hidden graph required')
    graph_KV = query_KV(engine, metadata.slot_mapping.long())
    eager = engine.model(x, pos, metadata).clone()
    eager_KV = query_KV(engine, metadata.slot_mapping.long())
    if not (bool(torch.isfinite(graph).all()) and bool(torch.isfinite(eager).all())
            and torch.equal(graph, eager)
            and len(graph_KV) == len(eager_KV) == 128
            and all(torch.equal(a, b) and bool(torch.isfinite(a).all())
                    for a, b in zip(graph_KV, eager_KV, strict=True))):
        raise RuntimeError('ROOT399 allphysical graph/eager hidden and all128 KV bits')
    engine._frugal399_reference_extent = metadata.sequence_lens[0]
    return dict(result, allphysical_graph_eager_hidden_KV=True,
                actual_physical_S=32, independent_reference_S=24,
                actual_q_partition=list(metadata.actual_seq_lengths_q),
                actual_visible_KV_extent=list(metadata.sequence_lens))


def reference(*args, **kwargs):
    return specialized('reference')(*args, **kwargs)


def run(engine, native, stage, bank, output):
    function = specialized('run')
    function.__globals__['prepare'] = lambda case: prepare(case, bank['prefixes'])
    return function(engine, native, stage, bank, output=output)
