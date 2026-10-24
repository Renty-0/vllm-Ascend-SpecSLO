"""Exercise actual cohort policy, trace audit, and full terminal interfaces."""
import ast
import copy
import datetime
import hashlib
import importlib.abc
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
from types import FunctionType, SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent


class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'acl'):
            raise RuntimeError('HOST regression cannot load model/device backends')


def main():
    sys.meta_path.insert(0, NoModel())
    shell = (P/'startup466.sh').read_text()
    line = next(s for s in shell.splitlines() if s.startswith('export PYTHONPATH='))
    sys.path[:0] = line[len('export PYTHONPATH='):-len('${PYTHONPATH:-}')].split(':')
    sys.path.insert(0, str(B/'budget-diagnostics'))
    import admission463, admission466, audit466, generation463, generation438, generation466
    import trace463, trace466, runtime465, runtime466, guard465, guard466, extension465, extension466
    from projection431 import patch as refill_patch
    from check_full_audit_interfaces_HOST import evaluate

    rng = random.Random(46648)
    checks = committed = 0
    for _ in range(1000):
        count = rng.choice((0, 1, 8, 16, 24, 32, 48))
        active = list(range(count))
        candidates = list(range(count, count+rng.randint(0, 64-count)))
        rng.shuffle(candidates)
        params = [SimpleNamespace(slo_tpot_ms=rng.choice((40., 50., 150.)), arrival_ts=0.)
                  for _ in range(count+len(candidates))]
        states = [SimpleNamespace(delivered_tokens=rng.randint(1, 256), decode_elapsed_ms=0.,
                                  decode_start_elapsed_ms=0.) for _ in params]
        for param, state in zip(params, states):
            state.decode_elapsed_ms = rng.random()*param.slo_tpot_ms*state.delivered_tokens
        lengths = [rng.randint(1, 500) for _ in params]
        engine = SimpleNamespace(_dynamic_tree_generation=19)
        chosen = admission466.choose(engine, candidates, maximum_new=48, active=active,
            request_params=params, states=states,
            local_states=[SimpleNamespace(prompt_length=n) for n in lengths],
            prefetched=set(), now=1., counters={})
        considered = candidates[:48]
        available = max(0., min((.95*params[i].slo_tpot_ms*(states[i].delivered_tokens-1)
                                -states[i].decode_elapsed_ms for i in active), default=0.))
        valid = [n for n in range(8, len(considered)+1)
                 if 1.1*(31.1820130301752+.13797625060353721*sum(lengths[i] for i in considered[:n])) <= available]
        expected = max(valid, default=0) if active else len(considered)
        assert chosen == considered[:expected] and len(active)+len(chosen) <= 64
        checks += 1
        if not chosen:
            continue
        witness = engine._root463_pending_admission
        elapsed = [states[i].decode_elapsed_ms for i in active]
        cost = witness['predicted_prefill_ms']
        trace = dict(source_generation=19, absolute_arrivals=[0.]*len(params),
            tokenized_inputs=[[0]*n for n in lengths],
            sampling=[dict(slo_tpot_ms=p.slo_tpot_ms) for p in params],
            ROOT432_refill_witnesses=[dict(ROOT463_admission=witness, admitted=chosen,
                incumbents=active, original_arrival_clock=1., incumbent_decode_before_ms=elapsed,
                prefill_ms=cost*1.2)],
            standalone_prefill_calls=[dict(sequence_ids=chosen, uncached_queries=sum(lengths[i] for i in chosen))])
        result = audit466.audit_trace(trace)
        assert result['maximum_new_cohort'] == 48 and len(result['all_prediction_underestimates']) == 1
        bad = copy.deepcopy(trace)
        bad['ROOT432_refill_witnesses'][0]['ROOT463_admission']['maximum_new'] = 32
        try:
            audit466.audit_trace(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Wrong actual cohort witness accepted')
        committed += 1

    source = runtime466.source()
    inverse = source
    for old, new in reversed(runtime466.ROUTES):
        inverse = inverse.replace(new, old)
    assert inverse == runtime465.source()
    interfaces = evaluate(source, guard466.constructed_audit())
    assert interfaces['both_interfaces_pass'], interfaces
    inspect.signature(audit466.constructed()).bind(Path('/synthetic'), (64,))

    original = B/'tree-parent4-canonical-tail-observer-binding-v425/HOST425.generated-LIVE.py'
    before = generation463.project(generation438.project(refill_patch(original.read_text())))
    after = generation466.project(before, 48)
    clocks = lambda s: [ast.unparse(n) for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Call)
                       and ast.unparse(n.func) in ('time.time', 'time.perf_counter', '_dt_cpu_broadcast')]
    assert clocks(after) == clocks(before)
    admit = next(node for node in ast.walk(ast.parse(after))
                 if isinstance(node, ast.FunctionDef) and node.name == 'admit_available')
    vacancy = next(node.value for node in ast.walk(admit) if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and ast.unparse(node.targets[0]) == 'vacant_capacity')
    expression = compile(ast.Expression(vacancy), '<actual LIVE vacancy>', 'eval')
    vacancy_cases = 0
    for active_count in range(65):
        for reserved in range(65-active_count):
            value = eval(expression, {}, dict(active=list(range(active_count)), initial_batch_size=64,
                _REFILL_MAX426=32, _v75_hooks=SimpleNamespace(reserved_slots=lambda: reserved)))
            assert value == min(64-active_count-reserved, 32 if active_count else 48)
            vacancy_cases += 1
    for name in ('make_trace', 'warm_to_fixedpoint', 'run_windows'):
        assert trace466.namespace()[name].__code__ is trace463.namespace()[name].__code__
    for name in ('prepare_install', 'install_entry', 'after_install', 'qualify_component',
                 'cache_snapshot', 'bucket_snapshot', 'metadata_snapshot', 'refill_snapshot',
                 'preflight_snapshot', 'postfull_AR', 'snapshot'):
        assert getattr(extension466, name) is getattr(extension465, name)
    previous, current = guard465.execution_namespace(), guard466.execution_namespace()
    a, z = previous['execute_locked'].__code__, current['execute_locked'].__code__
    assert a.co_code == z.co_code and a.co_names == z.co_names and a.co_varnames == z.co_varnames
    assert current['actual_audit'] is guard466.actual_audit
    assert current['SCOPE']['global_active_limit'] == 64 and current['SCOPE']['maximum_new_cohort'] == 48
    assert current['SCOPE']['maximum_new_prefill_cohort'] == 48
    assert current['SCOPE']['maximum_empty_prefill_cohort'] == 48
    assert current['SCOPE']['maximum_nonempty_prefill_cohort'] == 32
    assert current['SCOPE']['original463_algorithm_unchanged'] is False
    assert current['SCOPE']['original_credit_formula_and_nonempty_ceiling_unchanged'] is True
    for key in ('MAX_SECONDS', 'MAX_TAG_BYTES', 'MAX_AGGREGATE_BYTES'):
        assert current[key] == previous[key]
    # Exercise the final added cohort check using explicit in-memory evidence.
    for maximum in (48, 32):
        payload = {'ROOT463_actual_credit_cost_admission': [{'admission': {'maximum_new_cohort': maximum}}]}
        function = guard466.actual_audit
        reader = FunctionType(function.__code__, dict(function.__globals__,
            constructed_audit=lambda: lambda directory: payload), function.__name__)
        if maximum == 48:
            assert reader('/synthetic') is payload
        else:
            try:
                reader('/synthetic')
            except RuntimeError:
                pass
            else:
                raise AssertionError('Final guard accepted cohort32')
    for path in P.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    for path in P.glob('*.py'):
        ast.parse(path.read_bytes(), filename=str(path))
    assert not any(name in sys.modules for name in ('torch', 'torch_npu'))
    result = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='PASS_HOST466_COHORT48_GLOBAL64_AND_ACTUAL_TERMINAL_INTERFACES',
        independent_policy_cases=checks, actual_witness_audits_and_negative_controls=committed,
        actual_terminal_interfaces=interfaces, complete_worker_source_inverse=True,
        original_clocks_preserved=True, original_numerical_callbacks_identical=True,
        global_active_limit=64, maximum_new_cohort=48, physical_cards=[0,1,2,7],
        maximum_empty_cohort=48, maximum_nonempty_cohort=32,
        actual_LIVE_vacancy_expression_cases=vacancy_cases,
        NPU_used=False, NN_permission=False, performance_result=None,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in P.iterdir() if path.is_file() and path.suffix in ('.py','.sh')})
    (P/'HOST466_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','independent_policy_cases','actual_witness_audits_and_negative_controls','NPU_used')}))


if __name__ == '__main__':
    main()
