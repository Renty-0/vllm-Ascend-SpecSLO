#!/usr/bin/env python3
"""Independent, terminal-only HOST audit of V88's actual bounded observations.

Usage: retained Python3.11 -B analyze_v88_actual_candidate_capture.py --tag TAG
No runtime verification subprocess, model construction or NPU initialization.
Numerical differences are findings. Missing/invalid evidence is an audit error.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile

BASE = Path('/root/SpecSLO/budget-calibration-20261015')
P = BASE/'tree-candidate-hidden-capture-v88'
D = BASE/'budget-diagnostics'
CPU = P/'CPU_RESULT_V88.json'
FREEZE = P/'PREPARED_SOURCE_HASHES_V88.json'
ROOT_PIN = D/'ROOT_V88_FINAL_CPU_AND_SOURCE_PIN_AUDIT.json'
FIXTURE = D/'V85_NUMERICAL_MISMATCH_CASES.json'
CPU_SHA = '11548c5af1fdfe4ead696cead75ae8dea3f783e52a7046da364032cd979ba920'
FREEZE_SHA = '05a50b35001f3498b6fb8fe80c056cc43e6ad11ea84a7afd77b17f831a04fc2d'
FIXTURE_SHA = '14d8e406920b3614413cd98cab93d67aa20f7f0b1104f1fc7bbc75e89bb3d485'
LIMITS = {0: 251, 1: 35}
PAIRS = {0: (7196, 8365), 1: (525, 7629)}
FORBIDDEN = []


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def get_owner(mapping, owner):
    return mapping.get(str(owner), mapping.get(owner))


def no_initialization(frame, event, function):
    module, name = frame.f_globals.get('__name__', '') or '', frame.f_code.co_name
    if ((event == 'call' and module.startswith('torch_npu.npu') and name in ('init', '_lazy_init'))
            or (event == 'call' and module.endswith('.native_engine') and name == '__init__')
            or (event == 'c_call' and getattr(function, '__name__', '') == '_npu_init')):
        FORBIDDEN.append(module+'.'+name)
        raise RuntimeError('Independent HOST audit forbids model/device initialization')


class Audit:
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.observed = {}
        self.artifacts = {}
        self.errors = []
        self.tensor_cache = {}

    def note(self, path, *, artifact=False):
        path = Path(path).resolve()
        entry = dict(bytes=path.stat().st_size, sha256=file_sha(path))
        if str(path) in self.observed:
            require(entry == self.observed[str(path)], 'Evidence changed while reading: '+str(path))
        self.observed[str(path)] = entry
        if artifact:
            require(path.is_relative_to(self.directory), 'Artifact escaped this attempt: '+str(path))
            self.artifacts[str(path)] = dict(entry)
        return path

    def read(self, path, *, artifact=False):
        path = self.note(path, artifact=artifact)
        raw = path.read_bytes()
        require(sha_bytes(raw) == self.observed[str(path)]['sha256'], 'Evidence changed on JSON read')
        return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

    def reference(self, descriptor):
        path = self.note(descriptor['path'], artifact=True)
        entry = self.observed[str(path)]
        for key in ('sha256', 'bytes'):
            if key in descriptor:
                require(descriptor[key] == entry[key], 'Producer artifact '+key+' differs: '+str(path))
        if 'uncompressed_bytes' in descriptor:
            require(len(gzip.decompress(path.read_bytes())) == descriptor['uncompressed_bytes'],
                    'Artifact uncompressed size differs: '+str(path))
        return path

    def tensors(self, descriptor):
        import torch
        path = self.reference(descriptor)
        if str(path) not in self.tensor_cache:
            data = torch.load(path, map_location='cpu', weights_only=True)
            require(isinstance(data, dict) and data, 'Tensor artifact must be a nonempty mapping')
            require(all(isinstance(k, str) and isinstance(v, torch.Tensor)
                        and v.device.type == 'cpu' and v.numel() > 0
                        and bool(torch.isfinite(v).all()) for k, v in data.items()),
                    'Tensor artifact contains nonfinite/non-CPU/nontensor values')
            self.artifacts[str(path)].update(format='torch-CPU-tensor-mapping',
                tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype), numel=v.numel())
                         for k, v in data.items()},
                producer_sha256_present='sha256' in descriptor)
            self.tensor_cache[str(path)] = data
        return self.tensor_cache[str(path)]

    def verify_map(self, mapping):
        require(isinstance(mapping, dict) and mapping, 'Source closure map absent')
        for name, wanted in mapping.items():
            require(Path(name).is_absolute() and len(wanted) == 64, 'Malformed source map entry')
            path = self.note(name)
            require(self.observed[str(path)]['sha256'] == wanted, 'Frozen source changed: '+name)
        return dict(files=len(mapping), map_sha256=digest(mapping), exact=True)

    def archive(self, path, wanted_sha, mapping):
        path = self.note(path)
        require(file_sha(path) == wanted_sha, 'Source archive SHA differs')
        wanted = {Path(name).name: value for name, value in mapping.items()}
        require(len(wanted) == len(mapping), 'Archive basenames alias')
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            require(len(members) == len(wanted) and {m.name for m in members} == set(wanted)
                    and all(m.isfile() for m in members), 'Source archive membership differs')
            for member in members:
                require(sha_bytes(archive.extractfile(member).read()) == wanted[member.name],
                        'Source archive content differs: '+member.name)
        return dict(path=str(path), sha256=wanted_sha, members=len(wanted), exact=True)

    def check(self, label, callback):
        try:
            return callback()
        except Exception as error:
            self.errors.append(dict(check=label, error=type(error).__name__+': '+str(error)))
            return None

    def kv(self, value):
        if 'sample_artifact' in value:
            path = self.reference(value['sample_artifact'])
            data = self.read(path, artifact=True)
            require(all(data.get(k) == v for k, v in value.items() if k != 'sample_artifact'),
                    'Compressed KV summary differs from referring metadata')
            value = data
        layers = value['layers']
        require([layer['layer'] for layer in layers] == list(range(64)), 'Expected actual64 target KV layers')
        for layer in layers:
            for kind in ('key_cache', 'value_cache'):
                item = layer[kind]
                require(item['finite'] is True and item['numel'] > 0
                        and len(item['sample_indices']) == len(item['values'])
                        and all(0 <= n < item['numel'] for n in item['sample_indices'])
                        and all(math.isfinite(v) for v in item['values'])
                        and all(math.isfinite(item[k]) and item[k] >= 0 for k in ('max_abs', 'rms')),
                        'Malformed actual KV statistics')
        if 'full_current_position_KV' in value:
            full = self.tensors(value['full_current_position_KV'])
            require(set(full) == {f'{layer}:{kind}' for layer in range(64)
                                 for kind in ('key_cache', 'value_cache')}, 'Full KV vector layer/key set differs')
            require(value['full_vector_count'] == 128, 'Full KV vector count differs')
            for layer in layers:
                for kind in ('key_cache', 'value_cache'):
                    tensor = full[f'{layer["layer"]}:{kind}'].float().reshape(-1)
                    stats = layer[kind]
                    require(tensor.numel() == stats['numel']
                            and tensor[stats['sample_indices']].tolist() == stats['values'],
                            'Saved full KV disagrees with actual sparse sample')
                    require(math.isclose(float(tensor.abs().max()), stats['max_abs'], abs_tol=1e-5, rel_tol=1e-5)
                            and math.isclose(float(tensor.square().mean().sqrt()), stats['rms'], abs_tol=1e-5, rel_tol=1e-5),
                            'Saved full KV disagrees with actual full-row statistics')
        return value


def vector_comparison(left, right):
    import torch
    require(left.shape == right.shape, 'Compared actual vector shapes differ')
    a, b = left.float(), right.float()
    delta = a-b
    return dict(shape=list(a.shape), finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                exact_equal=bool(torch.equal(a, b)), max_abs=float(delta.abs().max()),
                rms=float(delta.square().mean().sqrt()), candidate_rms=float(a.square().mean().sqrt()),
                reference_rms=float(b.square().mean().sqrt()))


def compare_kv(audit, left, right):
    a, b = audit.kv(left), audit.kv(right)
    require(a['logical_position'] == b['logical_position'], 'KV comparison logical input position differs')
    sparse, first, full = [], None, []
    for aa, bb in zip(a['layers'], b['layers'], strict=True):
        item = dict(layer=aa['layer'])
        for kind in ('key_cache', 'value_cache'):
            av, bv = aa[kind], bb[kind]
            require(av['numel'] == bv['numel'] and av['sample_indices'] == bv['sample_indices'], 'KV channel geometry differs')
            delta = [x-y for x, y in zip(av['values'], bv['values'], strict=True)]
            item[kind] = dict(sample_max_abs=max(map(abs, delta)),
                sample_rms=math.sqrt(sum(x*x for x in delta)/len(delta)),
                full_row_rms_delta=av['rms']-bv['rms'], full_row_max_abs_delta=av['max_abs']-bv['max_abs'])
        if first is None and any(item[k]['sample_max_abs'] != 0 for k in ('key_cache', 'value_cache')):
            first = item['layer']
        sparse.append(item)
    if all('full_current_position_KV' in row for row in (a, b)):
        av, bv = audit.tensors(a['full_current_position_KV']), audit.tensors(b['full_current_position_KV'])
        for key in sorted(av, key=lambda k: (int(k.split(':')[0]), k)):
            full.append(dict(key=key, **vector_comparison(av[key], bv[key])))
    return dict(rank_local_comparison=True, logical_position=a['logical_position'],
                first_differing_sample_layer=first, sparse_layers=sparse, full_current_position_vectors=full,
                all_prefix_KV_equality_measured=False)


def snapshots_rows(audit, snapshots, rank, *, candidate, prompt_by_owner, candidate_outputs=None):
    rows = []
    for snapshot in snapshots:
        require(snapshot['rank'] == rank and snapshot['actual_projection'] is True, 'Snapshot role/projection differs')
        projections = {p['projection_index']: p for p in snapshot['projections']}
        context = snapshot['context']
        if candidate:
            plans = context['plans']
            offset = 0
            for plan in plans:
                require(plan['model_offset'] == offset and len(plan['candidate_token_ids']) == len(plan['parents'])
                        == plan['candidate_budget'], 'Full physical plan rows/budget mapping differs')
                require(len(plan['positions']) == len(plan['cache_positions']) == len(plan['parents'])+1,
                        'Physical plan positions omitted')
                require(all(type(parent) is int and -1 <= parent < i for i, parent in enumerate(plan['parents'])),
                        'Physical plan parent order differs')
                require(plan['depths'] == [p-plan['prefix_len'] for p in plan['positions']], 'Full plan depth mapping differs')
                offset += len(plan['parents'])+1
            require(offset == context['physical_M'] and len(plans) == context['physical_S'], 'Physical model M/S differs')
        for row in snapshot['rows']:
            owner = row['owner']
            require(owner in LIMITS, 'Unrequested owner capture')
            projection = projections[row.get('projection_index', 0)]
            require(row['actual_projection'] is row['hidden_finite'] is row['local_logits_finite'] is True,
                    'Actual finite projection evidence differs')
            require(row['actual_model_row'] == row['model_row'] and row['actual_position'] == row['logical_position']
                    and row['production_token'] == row['selected_token_id']
                    and 0 <= row['head_row'] < projection['head_rows'], 'Actual row/production token differs')
            execution = projection['execution']
            require(execution['mode'] in ('eager', 'eager_fallback', 'capture_replay', 'replay'), 'Execution mode absent')
            if execution['mode'] in ('capture_replay', 'replay'):
                require(isinstance(execution['entry_key'], str) and execution['entry_key']
                        and execution['replay_executed'] is True, 'Actual replay entry-key evidence absent')
            if candidate:
                require(context['head_indices'][row['head_row']] == row['model_row'], 'Head selection mapping differs')
                require(projection['model_rows'] == context['physical_M']
                        and projection['head_rows'] == len(context['head_indices']), 'Actual model/head shapes differ')
                plan = next(p for p in context['plans'] if p['owner'] == owner and p['real'])
                q = row['query_index']
                paths = [[plan['root_token_id']]]
                for i, parent in enumerate(plan['parents']):
                    paths.append(paths[parent+1]+[plan['candidate_token_ids'][i]])
                require(row['ancestor_token_ids'] == paths[q] and row['actual_input_token'] == paths[q][-1]
                        and row['model_row'] == plan['model_offset']+q
                        and row['logical_position'] == plan['positions'][q]
                        and row['cache_position'] == plan['cache_positions'][q], 'Actual full plan/query path differs')
                committed = prompt_by_owner[owner]+candidate_outputs[owner]
                require(0 <= plan['prefix_len'] < len(committed)
                        and committed[plan['prefix_len']] == plan['root_token_id'], 'Submitted root differs from actual candidate output recurrence')
                submitted = committed[:plan['prefix_len']]+paths[q]
                require(row['input_prefix_sha256'] == digest(submitted), 'Candidate prefix hash differs from independently reconstructed committed prefix plus ancestor path')
                if row.get('post_compaction_input_prefix_committed'):
                    require(committed[:len(submitted)] == submitted, 'Reported committed KV path differs from actual candidate completion')
                require(row['input_prefix_length'] == len(prompt_by_owner[owner])+row['output_index']
                        == row['logical_position']+1, 'Logical prefix length/output index differs')
            start, end = row['local_vocab_start'], row['effective_local_vocab_end']
            require(start < end <= row['local_vocab_end'] and projection['local_vocab_start'] == start
                    and projection['local_vocab_end'] == row['local_vocab_end'], 'Local vocabulary geometry differs')
            require([p['token_id'] for p in row['pair_scores']] == list(PAIRS[owner]), 'Specified pair ids differ')
            for pair in row['pair_scores']:
                owned = start <= pair['token_id'] < end
                require(pair['owned_by_rank'] is owned and ((owned and type(pair['score']) in (float, int)
                        and math.isfinite(pair['score'])) or (not owned and pair['score'] is None)),
                        'Missing shard score was fabricated or owned pair is invalid')
            top = row['local_top2']
            require(len(top) == 2 and all(start <= p['token_id'] < end and math.isfinite(p['score']) for p in top)
                    and top[0]['score'] >= top[1]['score'], 'Actual local top2 invalid')
            if row['hidden_saved']:
                require(rank == 1, 'Hidden vector written by a nonleader rank')
                tensor = audit.tensors(snapshot['hidden_artifact'])[row['hidden_key']]
                require(tensor.ndim == 1 and tensor.numel() == projection['hidden_width'], 'Saved hidden shape differs')
            if 'KV_sample' in row:
                audit.kv(row['KV_sample'])
                require(row['KV_sample']['logical_position'] == row['logical_position'], 'KV/query position differs')
                if candidate:
                    require(row['post_compaction_input_prefix_committed'] is True, 'Candidate KV sampled from an uncommitted path')
            rows.append(dict(row, _hidden_artifact=snapshot['hidden_artifact'], _context=context, _projection=projection))
    return rows


def logical_signature(rows):
    keys = ('owner', 'capture_id', 'query_index', 'model_row', 'head_row', 'proposal_id', 'logical_position',
            'cache_position', 'output_index', 'input_prefix_sha256', 'ancestor_token_ids', 'production_token')
    return [{key: row.get(key) for key in keys} for row in rows]


def global_scores(rows):
    require(len(rows) == 3, 'Three actual target local shards are required')
    ranges = sorted((r['local_vocab_start'], r['effective_local_vocab_end']) for r in rows)
    require(ranges[0][0] == 0 and ranges[-1][1] == 151936
            and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), 'TP local vocabulary coverage has a gap/overlap')
    tokens = {r['production_token'] for r in rows}
    require(len(tokens) == 1, 'Production greedy tokens disagree across target ranks')
    choices = [p for r in rows for p in r['local_top2']]
    best = max(p['score'] for p in choices)
    tied = sorted(p['token_id'] for p in choices if p['score'] == best)
    token = next(iter(tokens))
    pairs = []
    for target in rows[0]['pair_token_ids']:
        owned = [p for r in rows for p in r['pair_scores'] if p['token_id'] == target and p['owned_by_rank']]
        require(len(owned) == 1, 'Specified token does not have exactly one actual TP owner')
        pairs.append(owned[0])
    owner = next(r for r in rows if r['local_vocab_start'] <= token < r['effective_local_vocab_end'])
    token_scores = [p['score'] for p in owner['local_top2'] if p['token_id'] == token]
    token_scores += [p['score'] for p in pairs if p['token_id'] == token]
    identity_under_tie_unknown = not token_scores
    if token_scores:
        require(all(score == best for score in token_scores),
                'Actual retained production-token score is below a retained local shard maximum')
    else:
        # CPU topk's two indices need not include Native max's chosen index
        # when at least three BF16 logits share the maximum. We retain the
        # global maximum value, but cannot recover the omitted ID's value.
        require(owner['local_top2'][0]['score'] == owner['local_top2'][1]['score'] == best,
                'Production token is absent from its unique actual local maximum')
    return dict(production_token=token, local_shard_ranges=ranges, actual_global_max_score=best,
                observed_tied_max_token_ids=tied, pair_scores=pairs,
                production_token_score=token_scores[0] if token_scores else None,
                winner_identity_under_tie_not_independently_reconstructable=identity_under_tie_unknown,
                production_token_score_independently_verified=not identity_under_tie_unknown,
                ranking_source='actual local top2 and retained owned pair scores; no hidden reprojection')


def retained_token_scores(rows, tokens):
    """Read a dynamically observed competing pair only from retained scalars.

    V88 chose its two producer pair IDs before this run. A new first
    difference can concern other IDs; absent top2 values are unavailable,
    never zero and never recovered by a second projection.
    """
    result = []
    for token in tokens:
        owner = next(r for r in rows if r['local_vocab_start'] <= token < r['effective_local_vocab_end'])
        values = [(p['score'], 'actual_local_top2') for p in owner['local_top2'] if p['token_id'] == token]
        values += [(p['score'], 'actual_preselected_pair_score') for p in owner['pair_scores']
                   if p['token_id'] == token and p['owned_by_rank']]
        require(not values or all(v[0] == values[0][0] for v in values), 'Retained token scores disagree within one actual projection')
        result.append(dict(token_id=token, score=values[0][0] if values else None,
            measured=bool(values), sources=[v[1] for v in values],
            unavailable_reason=None if values else 'dynamic_competitor_not_retained_in_local_top2_or_preselected_pair'))
    return result


def own_ar(audit, reference, prompt, owner, rank):
    outputs = reference['completion_token_ids']
    require(len(outputs) == LIMITS[owner] and reference['fresh_allocation'] is reference['own_recurrence'] is True
            and reference['candidate_tokens_used_as_inputs'] is False, 'Bounded AR extent/allocation differs')
    require(reference['prompt_sha256'] == digest(prompt) and len(reference['calls']) == len(reference['captures']) == len(outputs),
            'Own-AR call/capture history incomplete')
    rows = snapshots_rows(audit, reference['captures'], rank, candidate=False, prompt_by_owner={owner: prompt})
    require(len(rows) == len(outputs), 'Expected one actually projected row per AR step')
    for step, (call, row) in enumerate(zip(reference['calls'], rows, strict=True)):
        inputs = list(prompt) if step == 0 else [outputs[step-1]]
        first_pos = 0 if step == 0 else len(prompt)+step-1
        end_pos = len(prompt)+step
        require(call['step'] == step and call['input_sha256'] == digest(inputs)
                and call['query_count'] == len(inputs) and call['positions'] == [first_pos, end_pos]
                and call['Native_WORLD_finite'] is call['hidden_logits_finite'] is True,
                'Own-AR input recurrence/finite call record differs')
        require(row['owner'] == owner and row['output_index'] == step
                and row['input_prefix_sha256'] == digest(list(prompt)+outputs[:step])
                and row['actual_input_token'] == inputs[-1] and row['production_token'] == outputs[step]
                and row['logical_position'] == len(prompt)+step-1
                and row['model_row'] == len(inputs)-1 and row['head_row'] == 0,
                'Own-AR actual input/position/production output differs')
    return rows


def source_checks(audit, receipt):
    require(file_sha(CPU) == CPU_SHA and file_sha(FREEZE) == FREEZE_SHA and file_sha(FIXTURE) == FIXTURE_SHA,
            'Pinned CPU/freeze/fixture differs')
    cpu, frozen, pin = audit.read(CPU), audit.read(FREEZE), audit.read(ROOT_PIN)
    require(pin['all_checks_passed'] is True and pin['identity']['CPU_sha256'] == CPU_SHA
            and pin['identity']['freeze_sha256'] == FREEZE_SHA, 'Original root CPU pin differs')
    require(cpu['tests_run'] == 265 and len(cpu['modules']) == 31 and cpu['status'] == 'passed'
            and cpu['failures'] == cpu['errors'] == 0 and cpu['NPU_initialized'] is False
            and cpu['model_initialization_attempts'] == cpu['NPU_initialization_attempts'] == [], 'Actual CPU265/31 evidence differs')
    require(cpu['input_hashes'] == cpu['input_hashes_after'] == frozen['files'] and len(frozen['files']) == 204
            and cpu['external_input_hashes'] == cpu['external_input_hashes_after']
            and len(cpu['external_input_hashes']) == 1825 and frozen['CPU_sha256'] == CPU_SHA,
            'Actual source closure count/content differs')
    result = dict(CPU_sha256=CPU_SHA, freeze_sha256=FREEZE_SHA,
        local=audit.verify_map(frozen['files']), external_CPU=audit.verify_map(cpu['external_input_hashes']),
        CPU_archive=audit.archive(cpu['actual_local_source_archive'], cpu['actual_local_source_archive_sha256'], frozen['files']))
    if 'source_hashes' in receipt:
        expected = {**frozen['files'], str(CPU): CPU_SHA, str(FREEZE): FREEZE_SHA}
        require(receipt['source_hashes'] == expected, 'Supervisor source archive closure differs')
        result['supervisor_archive'] = audit.archive(receipt['source_snapshot'], receipt['source_snapshot_sha256'], expected)
        result['external_supervisor'] = audit.verify_map(receipt['external_input_hashes'])
        require(all(receipt['external_input_hashes'].get(k) == v for k, v in cpu['external_input_hashes'].items()),
                'Supervisor external closure does not contain original CPU closure')
    env = audit.read(D/'V88_ACTUAL_SOURCED_ENV_SHELLS_PRE.json')
    aliases = audit.read(D/'V88_RESTORED_MISSING_RUNTIME_ALIASES.json')
    require(env['runtime_changed'] is False and len(env['files']) == 2, 'Actual sourced shell pre evidence differs')
    result['actual_sourced_env_shells_pre'] = audit.verify_map(env['files'])
    require(aliases['installed_or_updated_packages'] is aliases['changed_retained_runtime_bytes'] is False
            and aliases['failed_first_receipt_preserved'] is True and len(aliases['verified_before_aliases']) == 13,
            'Original missing-runtime-alias restoration scope differs')
    for name, entry in aliases['verified_before_aliases'].items():
        require(file_sha(entry['actual_retained_path']) == file_sha(name) == entry['sha256'], 'Retained runtime/alias bytes differ')
        audit.note(name)
        audit.note(entry['actual_retained_path'])
    for name, target in aliases['created_symlinks'].items():
        require(Path(name).is_symlink() and Path(name).resolve() == Path(target).resolve(), 'Restored runtime alias destination differs')
    result['aliases'] = dict(record=aliases, verified_retained_files=13, aliases_verified=2)
    first = P/'local-receipt-candidate-hidden-kv48-v1.json'
    if first.exists():
        original = audit.read(first)
        require(original['status'] == 'failed' and 'child' not in original and original['errors'], 'First pre-verifier failure was not preserved')
        result['original_failed_pre_verifier_attempt'] = dict(path=str(first), status='failed', child_was_not_started=True,
                                                            errors=original['errors'])
    sys.path.insert(0, str(P))
    checker = importlib.import_module('check_v75')
    require(checker.candidate_files() == frozen['files'] and checker.external_CPU_files() == cpu['external_input_hashes'],
            'Current input-file closure differs from final CPU closure')
    return result


def numerical_findings(audit, inners):
    fixture = audit.read(FIXTURE)
    cases = {case['request_index']: case for case in fixture['cases']}
    leader = inners.get(1)
    require(leader is not None and leader.get('generate_results') is not None, 'Actual leader candidate outputs unavailable')
    results = leader['generate_results']
    require(len(results) == 48 and all(len(r['completion_token_ids']) == 256 for r in results), 'Candidate did not complete48x256')
    streamed = defaultdict(list)
    for event in leader['stream_events']:
        streamed[event['request_index']].extend(event['token_ids'])
    require(all(streamed[i] == r['completion_token_ids'] for i, r in enumerate(results)), 'Actual publication stream differs from final outputs')
    prompts = {owner: cases[owner]['prompt_tokens'] for owner in LIMITS}
    candidates, references, ar_rows = {}, {}, {}
    for rank, report in inners.items():
        require(report['rank'] == rank and report['schema'] == 'V88.actual-candidate-hidden-capture.v1', 'Actual report rank/schema differs')
        if rank == 0:
            require(report['candidate_captures'] == [] and report.get('generate_results') is None, 'Draft has target observations')
            continue
        require(all(get_owner(report['prompt_token_ids'], o) == prompts[o] for o in LIMITS), 'Actual original prompt tokenization differs')
        candidates[rank] = snapshots_rows(audit, report['candidate_captures'], rank, candidate=True, prompt_by_owner=prompts,
                                         candidate_outputs={owner: results[owner]['completion_token_ids'] for owner in LIMITS})
        references[rank] = {r['request_index']: r for r in report.get('bounded_AR_reference') or []}
        for owner, reference in references[rank].items():
            ar_rows[rank, owner] = own_ar(audit, reference, prompts[owner], owner, rank)
    all_targets = set(candidates) == {1, 2, 3}
    if all_targets:
        require(logical_signature(candidates[1]) == logical_signature(candidates[2]) == logical_signature(candidates[3]),
                'Actual candidate logical/production mappings differ across TP ranks')
    output = []
    for owner in LIMITS:
        available = [rank for rank in (1, 2, 3) if owner in references.get(rank, {})]
        if not all_targets or available != [1, 2, 3]:
            output.append(dict(owner=owner, status='partial-reference-evidence', candidate_tokens=results[owner]['completion_token_ids'],
                               available_reference_ranks=available, first_difference=None,
                               first_difference_evaluated=False))
            continue
        refs = [references[r][owner] for r in (1, 2, 3)]
        require(all(ref['completion_token_ids'] == refs[0]['completion_token_ids'] and ref['calls'] == refs[0]['calls'] for ref in refs),
                'Independent own-AR tokens/calls disagree across targets')
        actual, reference = results[owner]['completion_token_ids'], refs[0]['completion_token_ids']
        first = next((i for i, token in enumerate(reference) if token != actual[i]), None)
        findings = dict(owner=owner, candidate_tokens=actual, independent_AR_tokens=reference,
            independent_AR_steps=len(reference), three_target_own_AR_agree=True,
            first_difference=first, pinned_reference_prefix_reproduced=reference[:-1] == cases[owner]['ref_output_prefix'],
            pinned_reference_terminal_reproduced=reference[-1] == cases[owner]['reference_diff_id'],
            same_prefix_observations=[], pinned_position_observations=[])
        require(all(ref['alignment']['first_difference'] == first for ref in refs), 'Recorded first difference differs from actual token sequences')
        for step, destination in ((first, 'same_prefix_observations'), (len(reference)-1, 'pinned_position_observations')):
            if step is None:
                continue
            prefix = prompts[owner]+reference[:step]
            key = digest(prefix)
            matched = [[r for r in candidates[rank] if r['owner'] == owner and r['output_index'] == step
                        and r['input_prefix_sha256'] == key and r['production_token'] == actual[step]] for rank in (1, 2, 3)]
            require([r['capture_id'] for r in matched[0]] == [r['capture_id'] for r in matched[1]]
                    == [r['capture_id'] for r in matched[2]], 'Same-prefix capture identity differs across TP ranks')
            if destination == 'same_prefix_observations':
                require(all(ref['alignment']['same_prefix_capture_ids'] == [r['capture_id'] for r in matched[0]] for ref in refs),
                        'Recorded same-prefix capture ids differ from independently located actual rows')
            for n in range(len(matched[0])):
                rows = [group[n] for group in matched]
                refrows = [ar_rows[r, owner][step] for r in (1, 2, 3)]
                require(all(row['logical_position'] == len(prefix)-1 and row['actual_input_token'] == prefix[-1] for row in rows),
                        'First-difference candidate actual input is not the independently reconstructed shared prefix frontier')
                observation = dict(capture_id=rows[0]['capture_id'], step=step, input_prefix_sha256=key,
                    logical_position=len(prefix)-1, candidate_global=global_scores(rows), reference_global=global_scores(refrows),
                    rank_local=[], hidden=dict(measured=False, reason='actual leader hidden not retained'))
                competing = list(dict.fromkeys((actual[step], reference[step])))
                candidate_scores, reference_scores = retained_token_scores(rows, competing), retained_token_scores(refrows, competing)
                observation['dynamic_actual_competitors'] = dict(token_ids=competing,
                    candidate_scores=candidate_scores, reference_scores=reference_scores,
                    both_competitors_measured_on_both_sides=all(p['measured'] for p in candidate_scores+reference_scores),
                    preselected_producer_pair_ids=list(PAIRS[owner]), no_missing_score_fabricated=True)
                for rank, crow, rrow in zip((1, 2, 3), rows, refrows):
                    item = dict(rank=rank, candidate_pair_scores=crow['pair_scores'], reference_pair_scores=rrow['pair_scores'],
                                candidate_context=crow['_context'], candidate_projection=crow['_projection'],
                                reference_projection=rrow['_projection'],
                                actual_candidate_mapping={key: crow.get(key) for key in
                                    ('head_row', 'model_row', 'actual_input_token', 'logical_position', 'cache_position', 'actual_slot')},
                                actual_reference_mapping={key: rrow.get(key) for key in
                                    ('head_row', 'model_row', 'actual_input_token', 'logical_position', 'actual_slot')},
                                owned_pair_score_deltas=[dict(token_id=a['token_id'], candidate_minus_reference=a['score']-b['score'])
                                    for a, b in zip(crow['pair_scores'], rrow['pair_scores'], strict=True) if a['owned_by_rank']])
                    if 'KV_sample' in crow and 'KV_sample' in rrow:
                        item['KV'] = compare_kv(audit, crow['KV_sample'], rrow['KV_sample'])
                        if destination == 'same_prefix_observations':
                            producer = next(x['comparison'] for x in refs[rank-1]['alignment']['KV_sample_comparisons']
                                            if x['capture_id'] == crow['capture_id'])
                            require(producer['first_differing_sample_layer'] == item['KV']['first_differing_sample_layer'],
                                    'Producer first differing sparse KV layer differs')
                            for reported, computed in zip(producer['layers'], item['KV']['sparse_layers'], strict=True):
                                require(reported['layer'] == computed['layer'], 'Producer sparse KV layer mapping differs')
                                for kind in ('key_cache', 'value_cache'):
                                    require(reported[kind]['finite'] is True
                                            and all(reported[kind][k] == v for k, v in computed[kind].items()),
                                            'Producer sparse KV comparison differs from independent recomputation')
                        if destination == 'pinned_position_observations' and item['KV']['full_current_position_vectors']:
                            producer = next(x['comparison'] for x in refs[rank-1]['alignment']['pinned_full_position_KV_comparisons']
                                            if x['capture_id'] == crow['capture_id'])
                            computed = {row['key']: row for row in item['KV']['full_current_position_vectors']}
                            require(len(producer['vectors']) == 128
                                    and all(all(computed[row['key']][k] == v for k, v in row.items())
                                            for row in producer['vectors']),
                                    'Producer full-position KV comparison differs from independent tensor recomputation')
                    observation['rank_local'].append(item)
                crow, rrow = rows[0], refrows[0]
                if crow['hidden_saved'] and rrow['hidden_saved']:
                    left = audit.tensors(crow['_hidden_artifact'])[crow['hidden_key']]
                    right = audit.tensors(rrow['_hidden_artifact'])[rrow['hidden_key']]
                    observation['hidden'] = dict(measured=True, rank=1, **vector_comparison(left, right))
                if destination == 'same_prefix_observations':
                    for rank, ref, crow, rrow in zip((1, 2, 3), refs, rows, refrows):
                        stored = next(x for x in ref['alignment']['same_prefix_numerics'] if x['capture_id'] == crow['capture_id'])
                        stored_local = stored['actual_local_logits']
                        require(stored_local['candidate_production_token'] == crow['production_token']
                                and stored_local['reference_production_token'] == rrow['production_token']
                                and stored_local['candidate_local_top2'] == crow['local_top2']
                                and stored_local['reference_local_top2'] == rrow['local_top2'],
                                'Producer same-prefix logits summary differs from raw actual shard observations')
                        if rank == 1 and observation['hidden']['measured']:
                            require(all(observation['hidden'].get(k) == v for k, v in stored['hidden'].items()),
                                    'Producer hidden comparison differs from independent CPU tensor comparison')
                findings[destination].append(observation)
        findings['same_prefix_capture_available'] = bool(findings['same_prefix_observations']) if first is not None else None
        findings['preceding_tokens_equal'] = first is None or actual[:first] == reference[:first]
        output.append(findings)
    return dict(candidate_requests=48, candidate_output_tokens=12288,
                actual_candidate_logical_mappings_agree=True if all_targets else None,
                available_candidate_target_ranks=sorted(candidates),
                independent_AR_steps_by_rank={str(rank): sum(len(ref['completion_token_ids']) for ref in refs.values())
                                              for rank, refs in references.items()},
                cases=output, cross_rank_hidden_or_KV_equality_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag), 'Invalid tag')
    require(sys.version_info[:2] == (3, 11) and sys.dont_write_bytecode, 'Use retained Python3.11 -B')
    receipt_path = P/('local-receipt-'+args.tag+'.json')
    raw_receipt = json.loads(receipt_path.read_bytes())
    if raw_receipt.get('status') not in ('complete', 'failed'):
        print(json.dumps(dict(status='not-run', reason='supervisor-not-terminal', tag=args.tag)))
        return 2
    output = args.output or D/('ROOT_V88_ACTUAL_CANDIDATE_CAPTURE_'+args.tag+'_AUDIT.json')
    require(output.parent.resolve() == D.resolve() and not output.exists(), 'Output must be new and external in budget-diagnostics')
    directory = P/'local-output'/args.tag
    audit = Audit(directory)
    receipt = audit.read(receipt_path)
    require(receipt.get('action') == 'candidate-capture-probe', 'This tag is not the actual V88 capture action')
    result = dict(schema='ROOT.V88.actual-candidate-hidden-projection-KV-host-audit.v1',
        audited_UTC=datetime.now(timezone.utc).isoformat(), tag=args.tag,
        original_job_status=receipt['status'], original_job_status_modified=False,
        original_receipt=dict(path=str(receipt_path), sha256=file_sha(receipt_path), errors=receipt.get('errors'), child=receipt.get('child')),
        analysis_script=dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__)),
        full48_AR=False, full463_qualified=False, all65_qualified=False, formal_Goodput=False,
        online_qualified=False, numerical_qualification_inferred=False, root_cause_inferred=False,
        latency_or_Goodput_inferred=False, cross_rank_hidden_or_KV_equality_measured=False)
    result['original_terminal_evidence'] = {key: receipt.get(key) for key in
        ('pre_verifier', 'post_verifier', 'devices_before', 'devices_after', 'owned_remaining',
         'experimental_sources_unchanged', 'external_sources_unchanged', 'local_execution_complete')}
    verifier_logs = {}
    for label in ('pre', 'post'):
        path = P/(args.tag+'-'+label+'-verifier.log')
        if path.exists():
            audit.note(path)
            lines = [line for line in path.read_text().splitlines() if line.startswith('{')]
            if lines:
                verifier_logs[label] = audit.check('actual '+label+' verifier log', lambda raw=lines[-1]: json.loads(raw))
    result['actual_verifier_logs'] = verifier_logs
    previous = sys.getprofile()
    sys.setprofile(no_initialization)
    try:
        result['integrity_checks'] = audit.check('CPU/source/archive/environment', lambda: source_checks(audit, receipt))
        inners, wrappers, recovery, raw_ar = {}, [], {}, {}
        for rank in range(4):
            path = directory/f'rank{rank}-local-diagnostic.json'
            if path.exists():
                wrapper = audit.check('rank'+str(rank)+' report JSON', lambda p=path: audit.read(p, artifact=True))
                if wrapper:
                    wrappers.append(wrapper)
                    if isinstance(wrapper.get('candidate_capture_probe'), dict):
                        inners[rank] = wrapper['candidate_capture_probe']
                        def recovery_descriptor(r=rank):
                            descriptor = inners[r]['candidate_recovery_artifact']
                            path = audit.reference(descriptor)
                            require(path == (directory/f'rank{r}-candidate-actual-evidence.json.gz').resolve(),
                                    'Final report refers to another candidate attempt/checkpoint')
                        audit.check('rank'+str(rank)+' candidate recovery descriptor', recovery_descriptor)
                    else:
                        audit.errors.append(dict(check='rank'+str(rank)+' actual candidate payload', error='Actual payload missing'))
            candidate_path = directory/f'rank{rank}-candidate-actual-evidence.json.gz'
            if candidate_path.exists():
                recovered = audit.check('rank'+str(rank)+' candidate checkpoint', lambda p=candidate_path: audit.read(p, artifact=True))
                if recovered:
                    recovery[rank] = recovered
                    def candidate_checkpoint(c=recovered, r=rank):
                        require(c['rank'] == r and c['schema'] == 'V88.actual-candidate-hidden-capture.v1'
                                and c['status'] == 'actual-candidate-complete-bounded-reference-pending'
                                and c['candidate_capture_count'] == len(c['candidate_captures'])
                                and c['wrappers_restored'] is c['class_delegate_restored']
                                is c['candidate_cache_released'] is True,
                                'Actual candidate checkpoint identity/completion/release differs')
                        require(all(c[k] is False for k in ('full48_AR', 'all65_qualified', 'full463_qualified',
                                'numerical_qualification_inferred', 'online_qualified', 'cost_or_Goodput')),
                                'Recovery checkpoint unexpectedly asserts qualification')
                        prompts = {case['request_index']: case['prompt_tokens'] for case in audit.read(FIXTURE)['cases']}
                        tokenized = {item['request_index']: item for item in c['tokenized_inputs']}
                        require(len(tokenized) == 48 and all(tokenized[o]['token_count'] == len(tokens)
                                and tokenized[o]['token_sha256'] == digest(tokens) for o, tokens in prompts.items()),
                                'Actual candidate tokenization does not bind to independently pinned prompt tokens')
                        return dict(rank=r, actual_candidate_checkpoint_complete=True,
                                    actual_capture_calls=len(c['candidate_captures']), cache_released=True)
                    audit.check('rank'+str(rank)+' candidate checkpoint identity/completion', candidate_checkpoint)
                    if rank not in inners:
                        # Analysis-only view of original recovery objects. This is never
                        # persisted as, or passed to a validator as, an original report.
                        inners[rank] = dict(recovered, prompt_token_ids={o: c['prompt_tokens'] for o, c in
                            {case['request_index']: case for case in audit.read(FIXTURE)['cases']}.items()}, bounded_AR_reference=[])
                    else:
                        audit.check('rank'+str(rank)+' candidate checkpoint content', lambda r=rank, c=recovered:
                            require(all(inners[r].get(k) == c.get(k) for k in ('candidate_captures', 'generate_results', 'stream_events',
                                'normal_target_calls', 'logical_events', 'native_before', 'native_after', 'native_delta')),
                                'Full candidate recovery differs from actual final report'))
            for owner in LIMITS:
                ar_path = directory/f'rank{rank}-AR-owner{owner}-actual-evidence.json.gz'
                if ar_path.exists():
                    checkpoint = audit.check('rank'+str(rank)+' AR'+str(owner)+' checkpoint', lambda p=ar_path: audit.read(p, artifact=True))
                    if checkpoint:
                        raw_ar[rank, owner] = checkpoint
                        def ar_checkpoint(c=checkpoint, r=rank, o=owner):
                            require(c['rank'] == r and c['owner'] == o and c['cache_released'] is True
                                    and c['online_qualified'] is c['cost_or_Goodput'] is False,
                                    'AR recovery rank/owner/release/scope differs')
                            require((r == 0 and c['target_report'] is None) or (r != 0
                                    and c['target_report']['request_index'] == o
                                    and len(c['target_report']['completion_token_ids']) == LIMITS[o]),
                                    'AR checkpoint target role/actual bounded extent differs')
                        audit.check('AR checkpoint identity', ar_checkpoint)
                        if rank in inners and rank != 0:
                            refs = inners[rank].get('bounded_AR_reference') or []
                            existing = next((r for r in refs if r['request_index'] == owner), None)
                            if existing is not None:
                                audit.check('AR recovery content', lambda a=existing, b=checkpoint['target_report']:
                                    require(a == b, 'Full AR recovery differs from actual final report'))
                            elif checkpoint['target_report']:
                                inners[rank]['bounded_AR_reference'] = refs+[checkpoint['target_report']]
        result['available_evidence'] = dict(final_report_ranks=[r['rank'] for r in wrappers],
            candidate_recovery_ranks=sorted(recovery),
            AR_recovery_rank_owner_pairs=[list(key) for key in sorted(raw_ar)],
            analysis_view_ranks=sorted(inners),
            analysis_input_kind='original final payloads when present; original recovery checkpoints otherwise',
            original_worker_reports_reconstructed=False)
        result['actual_findings'] = audit.check('independent actual recurrence/rows/numerics', lambda: numerical_findings(audit, inners))
        if len(wrappers) == 4 and result['integrity_checks'] is not None:
            def frozen_audit():
                module = importlib.import_module('local_mixed_qualification')
                require(Path(module.__file__).resolve() == P/'local_mixed_qualification.py', 'Wrong frozen auditor module')
                return module.audit_completed_candidate_capture_worker_reports(wrappers)
            result['frozen_actual_four_rank_auditor'] = audit.check('frozen actual four-rank auditor', frozen_audit)
        else:
            audit.errors.append(dict(check='final four-rank reports', error='Incomplete; recovery evidence retained without asserting completion'))
        def cleanup():
            require(receipt.get('owned_remaining') == [] and receipt.get('devices_after', {}).get('all_idle') is True,
                    'Actual supervisor owned-process/device cleanup incomplete')
            require(all(receipt.get(k, {}).get('exit_code') == 0 for k in ('pre_verifier', 'post_verifier')),
                    'Actual original pre/post runtime verification did not both pass')
            require(all(verifier_logs.get(k, {}).get('status') == 'pass'
                        and verifier_logs[k].get('baseline_id') == 'nano-pearl-ascend-v1'
                        and verifier_logs[k].get('errors') == []
                        and verifier_logs[k].get('checked_external_files') == 15 for k in ('pre', 'post')),
                    'Actual original pre/post verifier log content did not both pass')
            require(receipt.get('experimental_sources_unchanged') is True and receipt.get('external_sources_unchanged') is True,
                    'Supervisor source preservation evidence absent/false')
            exited = audit.read(directory/'worker-exit.json', artifact=True)
            require(exited.get('closed') is True and exited.get('all_exited') is True, 'Original worker-exit incomplete')
            if receipt['status'] == 'complete':
                require(receipt.get('errors') == [] and receipt.get('child', {}).get('exit_code') == 0
                        and receipt.get('local_execution_complete') is True, 'Complete supervisor is inconsistent')
            return dict(owned_remaining=[], devices_after=receipt['devices_after'], worker_exit=exited,
                        pre_verifier=receipt['pre_verifier'], post_verifier=receipt['post_verifier'])
        result['cleanup'] = audit.check('actual terminal cleanup', cleanup)
        completed_path = directory/'completed.json'
        result['original_completion_record_present'] = completed_path.exists()
        if completed_path.exists():
            def completion():
                completed = audit.read(completed_path, artifact=True)
                require(completed['action'] == 'candidate-capture-probe' and completed['online_qualified'] is False
                        and completed['replies'] == [['calibrated', r, {'rows': 0, 'timings': 0}] for r in range(4)],
                        'Actual completion replies differ')
                return completed
            result['original_completion_record'] = audit.check('actual original completion replies', completion)
        elif receipt['status'] == 'complete':
            audit.errors.append(dict(check='actual original completion replies', error='Complete receipt has no completed.json'))
        def budgets():
            files = [p for p in directory.rglob('*') if p.is_file()]
            total = sum(p.stat().st_size for p in files)
            tensors = sum(p.stat().st_size for p in files if p.suffix == '.pt')
            require(total < 32*1024**2 and tensors <= 8*1024**2, 'Actual32MiB/8MiB aggregate evidence budget exceeded')
            for path in files:
                audit.note(path, artifact=True)
                if path.suffix == '.pt':
                    audit.tensors(dict(path=str(path)))
                elif path.name.endswith('.json.gz'):
                    value = audit.read(path, artifact=True)
                    if isinstance(value, dict) and 'layers' in value:
                        audit.kv(value)
            return dict(private_bytes=total, private_limit_bytes=32*1024**2,
                        tensor_bytes=tensors, tensor_limit_bytes=8*1024**2, files=len(files))
        result['actual_artifact_budgets'] = audit.check('all private artifacts and aggregate budgets', budgets)
        def unchanged():
            value = all(file_sha(path) == entry['sha256'] and Path(path).stat().st_size == entry['bytes']
                        for path, entry in audit.observed.items())
            require(value, 'Observed evidence changed during independent analysis')
            return value
        result['evidence_unchanged_after_analysis'] = audit.check('read-only evidence preservation', unchanged)
        require(not FORBIDDEN, 'Forbidden initialization attempted')
        require(not bool(getattr(sys.modules.get('torch_npu.npu'), '_initialized', False)),
                'NPU was initialized inside the HOST audit process')
    finally:
        sys.setprofile(previous)
    result.update(artifacts=audit.artifacts, observed_files=audit.observed, errors=audit.errors,
                  forbidden_initialization_attempts=FORBIDDEN, NPU_initialized=False)
    findings = result.get('actual_findings') or {}
    result['actual_raw_candidate_and_bounded_AR_evidence_complete'] = (
        set(recovery) == set(range(4)) and set(raw_ar) == {(r, o) for r in range(4) for o in LIMITS}
        and findings.get('candidate_output_tokens') == 12288
        and findings.get('independent_AR_steps_by_rank') == {'1': 286, '2': 286, '3': 286}
        and all(c.get('three_target_own_AR_agree') is True for c in findings.get('cases', []))
        and len(findings.get('cases', [])) == 2
        and not any(e['check'] != 'final four-rank reports' for e in audit.errors))
    result['all_checks_passed'] = receipt['status'] == 'complete' and not audit.errors and result.get('evidence_unchanged_after_analysis') is True
    result['status'] = 'audited-complete-evidence' if result['all_checks_passed'] else 'audited-terminal-partial-or-invalid-evidence'
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(output), status=result['status'], original_job_status=receipt['status'],
        candidate_output_tokens=findings.get('candidate_output_tokens'),
        first_differences={str(c['owner']): c['first_difference'] for c in findings.get('cases', [])},
        errors=len(audit.errors), artifacts=len(audit.artifacts), budgets=result.get('actual_artifact_budgets'))))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
