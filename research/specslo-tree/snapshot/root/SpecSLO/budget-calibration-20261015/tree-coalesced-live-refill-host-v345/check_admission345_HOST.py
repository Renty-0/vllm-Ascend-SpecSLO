"""Execute actual projected admission logic with isolated HOST dependencies."""
import ast
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B/name) for name in (
    'tree-static-q5-M120-SLO-first-full463-own-AR-v257',
    'tree-static-q5-M120-full463-own-AR-v256',
    'tree-causal80-small-cohort-role-aware-audit-full-v209',
    'tree-active-window32-16-local-v160',
    'tree-high-normal-cap4-native-families-v148')]
from generation257 import patch as original_generation
from window257 import adapt_source
from empty209 import patch as empty_patch
from admission345 import patch, descriptor


class Scalar:
    def __init__(self, values): self.value = values[0]
    def cpu(self): return self
    def item(self): return self.value


def source():
    base = (B/'tree-high-normal-cap4-native-families-v148/runtime_tree_decode.py').read_text()
    return original_generation(empty_patch(adapt_source(base, 48)))


def function(projected, prefix):
    fn = copy.deepcopy(next(n for n in ast.walk(ast.parse(projected))
                            if isinstance(n, ast.FunctionDef) and n.name == 'admit_available'))
    if prefix:
        end = next(i for i, n in enumerate(fn.body)
                   if isinstance(n, ast.If) and ast.unparse(n.test) == 'admitted and prefill_coalescing')
        fn.body = fn.body[:end] + ast.parse('return admitted').body
    return compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                   __file__+'::actual-projected-admit-available', 'exec')


def namespace(lengths, active, pending, arrivals=None, priority=None, prefetched=(), token_limit=131072):
    n = len(lengths)
    arrivals = [0.0]*n if arrivals is None else arrivals
    states = [SimpleNamespace(home_batch_id=i%2, arrival_wait_ms=0.0, delivered_tokens=0,
                              decode_start_elapsed_ms=0.0, decode_elapsed_ms=0.0) for i in range(n)]
    for state in states:
        state.add_decode_time = lambda elapsed, s=state: setattr(s, 'decode_elapsed_ms', s.decode_elapsed_ms+elapsed)
    def local(i):
        return SimpleNamespace(prompt_length=lengths[i], max_tokens=256,
                               token_ids=[i]*lengths[i], committed_length=lengths[i],
                               committed_completion_token_ids=[700+i])
    draft = [local(i) for i in range(n)]
    target = [local(i) for i in range(n)]
    effects = dict(activated=[], prefills=[], deliveries=[], releases=[], clocks=[])
    def prefill(tokens, local_states, owners):
        effects['prefills'].append(list(owners))
        return [700+i for i in owners]
    owner = SimpleNamespace(rank=1, topology=SimpleNamespace(target_leader_rank=1), eos_token_ids=[],
        config=SimpleNamespace(spec_rhythm_online_prefill=True, spec_rhythm_kv_ready_arrivals=False,
                               max_num_batched_tokens=token_limit),
        _prefill_and_sample_target_batch=prefill,
        _deliver_committed_tokens=lambda i, params, state: effects['deliveries'].append(i))
    times = iter([10.0, 10.05])
    ns = dict(self=owner, receive_live_admissions=lambda block=False: False,
              active=list(active), pending_admission=list(pending), initial_batch_size=48,
              torch=SimpleNamespace(tensor=lambda values, **kwargs: Scalar(values), float64='HOST'),
              time=SimpleNamespace(time=lambda: 100.0, perf_counter=lambda: next(times)),
              _dt_cpu_broadcast=lambda *args, **kwargs: effects['clocks'].append(kwargs['kind']),
              order_spec_rhythm_admission_params=lambda indices, **kwargs: list(priority) if priority is not None else list(indices),
              request_params=[SimpleNamespace(arrival_ts=a) for a in arrivals],
              has_slo_constraints=True, envs=SimpleNamespace(VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=True),
              prefetched=set(prefetched), local_states=target, draft_states=draft, target_states=target,
              states=states, counters=dict(spec_rhythm_prefill_batches=0, spec_rhythm_prefill_requests=0,
                  spec_rhythm_prefill_singleton_batches=0, spec_rhythm_prefill_pair_batches=0,
                  spec_rhythm_prefill_multi_batches=0, spec_rhythm_peak_active_requests=len(active)),
              prefill_coalescing=False,
              activate_online_cache=lambda owners: effects['activated'].append(list(owners)),
              release_online_cache=lambda owners: effects['releases'].append(list(owners)),
              _finished=lambda state, eos: False, completed={}, finished_decode_elapsed={})
    return ns, effects


def pages(length): return (length+256+8+127)//128


def main():
    base = source()
    projected = patch(base)
    prefix = function(projected, True)
    complete = function(projected, False)
    rows = []
    fixtures = [
        ('initial48', [128]*48, [], list(range(48)), {}, 48),
        ('initial-long', [2301]*48, [], list(range(48)), {}, 22),
        ('refill4', [128]*48, list(range(44)), list(range(44,48)), {}, 4),
        ('defer3-of-many', [128]*52, list(range(45)), list(range(45,52)), {}, 0),
        ('tail3', [128]*48, list(range(45)), list(range(45,48)), {}, 3),
        ('arrival-only3', [128]*52, list(range(40)), list(range(40,52)),
         dict(arrivals=[0]*43+[200]*9), 0),
        ('priority', [128]*48, list(range(40)), list(range(40,48)),
         dict(priority=list(reversed(range(40,48)))), 8),
        ('incumbent-pages', [2301]*24, list(range(20)), list(range(20,24)), {}, 2),
        ('cached-pending', [2301]*24, list(range(20)), list(range(20,24)),
         dict(prefetched=[20]), 2),
        ('token-limited', [128]*48, list(range(40)), list(range(40,48)),
         dict(token_limit=256), 2),
    ]
    for label, lengths, active, pending, options, count in fixtures:
        ns, _ = namespace(lengths, active, pending, **options)
        exec(prefix, ns)
        selected = ns['admit_available']()
        assert len(selected) == count, (label, selected)
        owners = set(active) | set(selected) | (set(options.get('prefetched', ())) & set(pending))
        reserved = 64+sum(pages(lengths[i]) for i in owners)
        assert reserved <= 544
        expected_order = options.get('priority', pending)
        expected_order = [i for i in expected_order if options.get('arrivals', [0]*len(lengths))[i] <= 100]
        assert selected == expected_order[:len(selected)]
        assert ns['pending_admission'] == pending and ns['active'] == active
        rows.append(dict(case=label, selected=selected, reserved_pages=reserved))

    # Run the entire original function, including actual home assignment,
    # prefill duration charge, draft/target root append and once publication.
    ns, effects = namespace([128]*48, list(range(44)), list(range(44,48)))
    incumbent_tokens = [list(s.token_ids) for s in ns['local_states'][:44]]
    incumbent_homes = [s.home_batch_id for s in ns['states'][:44]]
    exec(complete, ns)
    assert ns['admit_available']() is False
    assert ns['active'] == list(range(48)) and ns['pending_admission'] == []
    assert effects['prefills'] == effects['activated'] == [list(range(44,48))]
    assert effects['deliveries'] == list(range(44,48)) and effects['releases'] == []
    assert effects['clocks'] == ['arrival_clock', 'prefill_clock']
    assert [s.home_batch_id for s in ns['states'][:44]] == incumbent_homes
    assert [s.token_ids for s in ns['local_states'][:44]] == incumbent_tokens
    assert all(abs(s.decode_elapsed_ms-50.0) < 1e-8 for s in ns['states'][:44])
    assert [s.home_batch_id for s in ns['states'][44:]] == [0,1,0,1]
    for i in range(44,48):
        assert ns['target_states'][i].token_ids[-1] == ns['draft_states'][i].token_ids[-1] == 700+i
        assert ns['target_states'][i].committed_length == ns['draft_states'][i].committed_length == 129

    # Random actual loop inputs cover page-limited and vacancy-limited choices.
    rng = random.Random(345)
    randomized = 0
    for _ in range(200):
        lengths = [rng.choice((59,97,127,128,609,1505,2301)) for i in range(60)]
        active = list(range(rng.randrange(30)))
        if 64+sum(pages(lengths[i]) for i in active) > 544: continue
        pending = list(range(len(active), 60))
        rng.shuffle(pending)
        arrivals = [rng.choice((0,99,101,200)) for i in lengths]
        ns, _ = namespace(lengths, active, pending, arrivals=arrivals)
        exec(prefix, ns)
        selected = ns['admit_available']()
        assert len(active)+len(selected) <= 48
        assert 64+sum(pages(lengths[i]) for i in active+selected) <= 544
        eligible = [i for i in pending if arrivals[i] <= 100]
        assert selected == eligible[:len(selected)]
        randomized += 1

    for bad in (0, 9, True):
        try: patch(base, bad)
        except ValueError: pass
        else: raise AssertionError('invalid minimum accepted')
    try: patch(projected)
    except RuntimeError: pass
    else: raise AssertionError('rolling source patched twice')
    assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
    result = dict(status='passed_actual_admission_HOST', NPU_used=False, model_constructed=False,
        descriptor=descriptor(base), fixtures=rows, randomized_actual_loop_cases=randomized,
        whole_original_function_executed=True, original_prefill_time_charged_to_incumbents_ms=50,
        original_arrival_and_prefill_clocks_retained=True, original_once_root_publication=True,
        original_draft_target_root_append=True, incumbent_tokens_and_homes_unchanged=True,
        whole_source_AST_inverse_passed=True, no_model_or_performance_qualification=True)
    (P/'HOST345_ADMISSION_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
