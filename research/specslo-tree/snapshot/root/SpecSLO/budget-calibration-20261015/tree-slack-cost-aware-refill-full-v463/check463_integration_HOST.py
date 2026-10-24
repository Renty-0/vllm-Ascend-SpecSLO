"""CPU contracts for actual source routing and retained admission evidence."""

from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib.abc
import json
import sys

P = Path(__file__).resolve().parent
B = P.parent


class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'acl'):
            raise RuntimeError('HOST integration cannot import model/device backend')


sys.meta_path.insert(0, NoModel())
sys.path.insert(0, str(B / 'tree-adaptive-own-history-full-v460'))
from host_env460 import setup
setup()
sys.path.insert(0, str(P))
import admission463
import audit463
import extension463
import extension460
import generation463
import runtime463
import trace463
import trace460

callbacks = ('prepare_install', 'install_entry', 'after_install', 'qualify_component',
             'cache_snapshot', 'bucket_snapshot', 'metadata_snapshot', 'refill_snapshot',
             'preflight_snapshot', 'postfull_AR', 'snapshot')
for name in callbacks:
    assert getattr(extension463, name) is getattr(extension460, name)
source = runtime463.source()
ast.parse(source)
assert source.count("extension_module='extension463'") == 1
assert source.count('from trace463 import run_windows,warm_to_fixedpoint') == 1
assert source.count('from wait462 import receive_qualification') == 1
new, old = trace463.namespace(), trace460.namespace()
for name in ('make_trace', 'run_windows', 'warm_to_fixedpoint'):
    assert new[name].__code__ is old[name].__code__
assert new['canonical_patch'] is generation463.patch
assert new['trace_source'] is old['trace_source']

def example(old_count=16, new_count=32):
    size = old_count + new_count
    params = [SimpleNamespace(slo_tpot_ms=40.0, arrival_ts=0.0) for _ in range(size)]
    states = [SimpleNamespace(delivered_tokens=21, decode_elapsed_ms=560.0,
                              decode_start_elapsed_ms=0.0) for _ in params]
    local = [SimpleNamespace(prompt_length=100) for _ in params]
    engine = SimpleNamespace(_dynamic_tree_generation=7)
    selected = admission463.choose(engine, list(range(old_count, size)), active=list(range(old_count)),
        request_params=params, states=states, local_states=local, prefetched=set(), now=1.0, counters={})
    decision = engine._root463_pending_admission
    row = dict(incumbents=list(range(old_count)), admitted=selected, prefill_ms=180.0,
               incumbent_decode_before_ms=[560.0] * old_count, original_arrival_clock=1.0,
               ROOT463_admission=decision)
    prefill = dict(sequence_ids=selected, uncached_queries=100 * len(selected))
    return dict(source_generation=7, ROOT432_refill_witnesses=[row], standalone_prefill_calls=[prefill],
                absolute_arrivals=[0.0] * size, tokenized_inputs=[[0] * 100 for _ in params],
                sampling=[dict(slo_tpot_ms=40.0) for _ in params])

valid = example()
assert audit463.audit_trace(valid)['committed_refills'] == 1
underestimated = copy.deepcopy(valid)
underestimated['ROOT432_refill_witnesses'][0]['prefill_ms'] = 300.0
assert len(audit463.audit_trace(underestimated)['all_prediction_underestimates']) == 1
# Poor cost predictions must be retained as measurements, rather than used
# to discard an otherwise complete run or silently improve its SLO score.

rejected = []
for kind in ('missing_decision', 'predicted_cost', 'future', 'arrival_clock', 'credit',
             'prompt_queries', 'nonprefix', 'global_limit', 'generation', 'model_hash'):
    trial = copy.deepcopy(valid)
    row = trial['ROOT432_refill_witnesses'][0]
    d = row['ROOT463_admission']
    if kind == 'missing_decision':
        row.pop('ROOT463_admission')
    elif kind == 'predicted_cost':
        d['predicted_prefill_ms'] = 0.0
    elif kind == 'future':
        owner = d['candidates'][0]
        d['candidate_arrivals'][0] = trial['absolute_arrivals'][owner] = 2.0
    elif kind == 'arrival_clock':
        d['arrival_clock'] += 1.0
    elif kind == 'credit':
        d['incumbents'][0]['credit_ms'] += 1.0
    elif kind == 'prompt_queries':
        d['candidate_queries'][0] += 1
    elif kind == 'nonprefix':
        d['selected'].reverse()
    elif kind == 'global_limit':
        d['global_active_limit'] = 32
    elif kind == 'generation':
        d['generation'] = 8
    else:
        d['fixed_cost_model_SHA256'] = '0' * 64
    try:
        audit463.audit_trace(trial)
    except (RuntimeError, KeyError):
        rejected.append(kind)
    else:
        raise AssertionError('Invalid actual admission witness accepted: ' + kind)

assert not any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'acl') for name in sys.modules)
result = dict(status='PASS_HOST463_FULL_INTERFACE_AND_ADMISSION_AUDIT_CONTROLS',
    unchanged_numerical_callbacks=list(callbacks), all_original_trace_code_objects_retained=True,
    complete_worker_main_source_inverse=True, actual_wait462_route_retained=True,
    rejected_controls=rejected, prediction_underestimates_retained=True,
    synthetic_witnesses_only=True, NPU_used=False, NN_permission=False,
    full463=False, Raw=None, Goodput=None,
    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.glob('*.py'))})
with (P / 'HOST463_INTEGRATION_RESULT.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps({k: v for k, v in result.items() if k != 'sources'}))
