"""HOST-only source/callback/corruption checks; no model or NPU claim."""
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent
for name in ('tree-static-q5-M120-SLO-first-full463-own-AR-v257',
        'tree-static-q5-M120-node-math-v255', 'tree-static-q5-M128-full463-own-AR-v251',
        'tree-static-q5-M128-full463-own-AR-v251', 'tree-static-q5-M128-projection-host-v249',
        'tree-canonical-contiguous-full463-own-AR-v235',
        'tree-contiguous-branch-tail-final-metadata-math-v234',
        'tree-causal80-small-cohort-role-aware-audit-full-v209'):
    sys.path.append(str(B / name))


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT398 HOST: ' + message)


def rejects(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return
    raise RuntimeError('ROOT398 HOST corruption unexpectedly accepted')


def main():
    from bucket398 import bind, extent, valid_record
    need([extent(v) for v in ((1,), (512,), (513,), (1535, 1536), (2049,), (2560,))]
         == [512, 512, 1024, 1536, 2560, 2560], 'visible-context boundary rounding')
    for bad in ((), (0,), (2561,), (True,), (1.0,)):
        rejects(lambda: extent(bad))

    @dataclass(frozen=True)
    class Meta:
        sequence_lens: tuple
        actual_seq_lengths_q: tuple = tuple(range(5, 121, 5))
        masks: object = None

    lengths = [606, 502] + [5] * 22
    token = SimpleNamespace(numel=lambda: 120)
    masks = object()
    state = SimpleNamespace(static_graph_calls=[], static_current=None)
    def prepare():
        original = Meta(tuple(lengths), masks=masks)
        physical = Meta((2560,) * 24, masks=masks)
        state.static_current = dict(original=(token, None, original),
            physical=(token, None, physical, object(), object()))
        return state.static_current['original']
    state.prepare = prepare
    def greedy():
        state.static_graph_calls.append(dict(sequence_lens=list(state.static_current['physical'][2].sequence_lens)))
        return token
    runner = SimpleNamespace(run_target_greedy=greedy)
    engine = SimpleNamespace(is_draft=False, _branch_tail_state=state, graph_runner=runner)
    bind(engine)
    state.prepare()
    need(state.static_current['physical'][2].sequence_lens == (2560,) * 24, 'original227 fixed extent first')
    state.uniform_KV_bucket398_enabled = True
    original = state.prepare()
    physical = state.static_current['physical']
    need(original[0] is token and physical[0] is token and physical[2].masks is masks
         and physical[2].actual_seq_lengths_q == tuple(range(5, 121, 5))
         and physical[2].sequence_lens == (1024,) * 24, 'only uniform extent changes')
    need(runner.run_target_greedy() is token, 'same output object and original greedy path')
    record = dict(fixed_KV_extent=[1024] * 24,
        uniform_KV_bucket398=state.static_graph_calls[-1]['uniform_KV_bucket398'])
    need(valid_record(record), 'actual prepared metadata and observer agree')
    damaged = copy.deepcopy(record)
    damaged['fixed_KV_extent'] = [512] * 24
    need(valid_record(damaged) is False, 'visible606-token context cannot truncate to512')
    rejects(lambda: bind(engine))
    lengths[0] = 2561
    rejects(state.prepare)

    from runtime398 import project_source
    source = (B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text()
    projected, changes = project_source(source)
    compile(projected, 'ROOT398.actual-source-projection', 'exec')
    need("extension_module='extension398'" in projected
         and "report['ROOT398_uniform_KV_math']" in projected
         and 'from scope398 import issue' in projected
         and 'from trace398 import run_windows' in projected
         and 'from audit398 import audit' in projected, 'actual worker uses declared new obligations')
    rejects(lambda: project_source(source.replace("extension_module='extension209'", "extension_module='changed'")))

    # Test the geometry auditor with actual retained227 dispatches. These are
    # HOST protocol fixtures only; numerical qualification is never inherited.
    proof_path = B / ('tree-causal80-small-cohort-role-aware-audit-full-v209/actual/'
        'static-M120-B5-tail3-root-v372-full463-ownAR-v1/canonical-independent-node-math/rank1-proof.json')
    proof = json.loads(proof_path.read_text())
    expected = [v['case'] for v in proof['math']['cases']]
    fixture = json.loads((B / 'tree-static-q5-M120-SLO-first-full463-own-AR-v257/FIXTURE257.json').read_text())
    prefixes = fixture['prefixes']
    calls = copy.deepcopy(proof['actual_static_graph_calls'])
    cursor = 0
    seen = set()
    for case in expected:
        capacity = 11 if 5 in case['budgets'] else 10
        count = 3 if capacity not in seen else 1
        visible = [len(prefixes[i]) + min(budget, 4) + 1 for i, budget in enumerate(case['budgets'])]
        selected = extent(tuple(visible))
        for call in calls[cursor:cursor + count]:
            call['sequence_lens'] = [selected] * 24
            call['uniform_KV_bucket398'] = dict(enabled=True, extent=selected, original_sequence_lens=visible)
        cursor += count
        seen.add(capacity)
    from dispatch398 import audit_calls
    result = audit_calls(expected, calls, prefixes)
    need(result['positive_node_cases'] == 227, 'all original227 permutation and graph checks')
    corrupt = copy.deepcopy(calls)
    corrupt[0]['sequence_lens'] = [512] * 24
    corrupt[0]['uniform_KV_bucket398'].update(extent=512, original_sequence_lens=[1])
    rejects(lambda: audit_calls(expected, corrupt, prefixes))
    warm_calls = [call for call in calls if call['execution_mode'] != 'capture_replay']
    # Cold groups have two extra replays. Retain just the final replay each.
    warm_calls = []
    cursor = 0
    seen = set()
    for case in expected:
        capacity = 11 if 5 in case['budgets'] else 10
        count = 3 if capacity not in seen else 1
        warm_calls.append(calls[cursor + count - 1])
        cursor += count
        seen.add(capacity)
    need(audit_calls(expected, warm_calls, prefixes)['actual_graph_dispatches'] == 227,
         'existing naturally qualified entries reuse, without flag assignments')
    from extension398 import projected_qualifier
    function = projected_qualifier()
    need(function.__globals__['_ROOT398_valid_record'] is valid_record,
         'actual additional extent gate bound to projected original qualifier')
    from oracle398 import specialized
    neural_body = specialized()
    def constants(code):
        values = []
        for value in code.co_consts:
            values.extend(constants(value) if hasattr(value, 'co_consts') else [value])
        return values
    need('ROOT398 strict hidden and all128 KV bits' in constants(neural_body.__code__),
         'actual nested neural compare retains additional strict bitwise gate')
    need(neural_body.__globals__['candidate'].__module__ == 'oracle255'
         and neural_body.__globals__['reference'].__module__ == 'oracle255',
         'actual original candidate and independent own-reference functions retained')
    from audit398 import projected_original_audit
    need(projected_original_audit().__globals__['_ROOT398_valid_record'] is valid_record,
         'actual full463 auditor changes only declared geometry extent condition')
    from prefixes398 import mid_prefixes, short_prefixes
    for rows in (mid_prefixes(prefixes), short_prefixes(prefixes)):
        need([len(p) % 128 for p in rows] == [len(p) % 128 for p in prefixes],
             'exact original mixed/mid/short context page residues')
    need('torch' not in sys.modules and 'torch_npu' not in sys.modules,
         'HOST protocol regression imports no model or NPU library')
    result = dict(status='PASS_HOST398_BOUNDARIES_CALLBACKS_RUNTIME_AND_227_DISPATCH_CORRUPTIONS',
        NPU_used=False, neural_qualification_inherited=False, production_qualified=False,
        original_worker_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        exact_reversible_runtime_changes=len(changes), actual_retained_dispatch_cases=227,
        all_original_draft_and_qualification_bodies_preserved=True)
    (P / 'HOST398_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
