"""Read only completed, explicitly enumerated evidence; no backend imports."""
from pathlib import Path
import ast, datetime, gzip, hashlib, json, shlex

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
OLD = B / 'tree-live-prefix-page-protection-v85'
BASE = B / 'tree-high-normal-cap4-native-families-v148'
N = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
PINS = {}

def pin(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    value = hashlib.sha256(raw).hexdigest()
    assert expected is None or value == expected, str(path)
    PINS[str(path)] = dict(sha256=value, bytes=len(raw))
    return raw

def read(path, expected=None):
    return json.loads(pin(path, expected))

def digest(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()

def definitions(path, names):
    tree = ast.parse(pin(path).decode())
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names}

def config(path):
    tree = ast.parse(pin(path).decode())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'PEARLConfig']
    assert len(calls) == 1
    return {k.arg: ast.unparse(k.value) for k in calls[0].keywords}

def exports(path):
    output = {}
    for line in pin(path).decode().splitlines():
        if line.startswith('export '):
            for item in shlex.split(line[7:]):
                if '=' in item:
                    key, value = item.split('=', 1)
                    output[key] = value
    return output

def main():
    pin(__file__)
    old_audit = read(D / 'ROOT_V85_ACTUAL_CANDIDATE_AR_MISMATCH_AUDIT.json')
    receipt = read(OLD / 'local-receipt-live-poisson48-page-prefix-v1.json',
                   old_audit['original_supervisor_receipt']['sha256'])
    assert receipt['physical_cards'] == [2, 5, 6, 7]
    assert all(receipt[k]['exit_code'] == 0 for k in ('pre_verifier', 'post_verifier'))
    assert receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True
    lock_path = N / 'lock.json'
    lock = read(lock_path, receipt['external_input_hashes'][str(lock_path)])
    old_candidate_path = OLD / 'live-progress/rank1-candidate-before-AR.json'
    old_candidate = read(old_candidate_path, old_audit['actual_raw_inputs'][str(old_candidate_path)])
    refs, reference_checks = [], []
    for i in (0, 1):
        rows = []
        for rank in (1, 2, 3):
            path = OLD / f'live-progress/rank{rank}-AR-request{i}.json'
            checkpoint = read(path, old_audit['actual_raw_inputs'][str(path)])
            assert checkpoint['rank'] == rank and checkpoint['request_index'] == i
            assert checkpoint['completed_steps'] == 256
            rows.append(checkpoint['reference_row'])
        assert rows[0] == rows[1] == rows[2]
        row = rows[0]
        prompt = old_candidate['tokenized_inputs'][i]
        assert row['prompt_sha256'] == prompt['token_sha256']
        assert len(row['completion_token_ids']) == len(row['calls']) == 256
        assert row['execution'] == 'direct Native eager PA; fresh allocation; own previous predictions'
        for step, call in enumerate(row['calls']):
            assert call['step'] == step
            assert call['query_count'] == (prompt['token_count'] if step == 0 else 1)
            assert call['start'] == (0 if step == 0 else prompt['token_count'] + step - 1)
            assert call['end'] == (prompt['token_count'] if step == 0 else prompt['token_count'] + step)
            assert call['input_sha256'] == (row['prompt_sha256'] if step == 0
                                            else digest([row['completion_token_ids'][step - 1]]))
            assert call['hidden_logits_and_Native_vote_finite'] is True
        refs.append(row)
        reference_checks.append(dict(request_index=i, request_id=row['request_id'],
            target_ranks=[1, 2, 3], all256_own_recurrence_calls_verified=True,
            prompt_token_count=prompt['token_count'], prompt_sha256=row['prompt_sha256'],
            completion_sha256=digest(row['completion_token_ids']), execution=row['execution']))

    names = ('tokenize_prefix', 'sampling_specifications', '_digest')
    assert definitions(OLD / 'poisson_coverage_probe.py', names) == definitions(BASE / 'poisson_coverage_probe.py', names)
    old_cfg, new_cfg = config(OLD / 'startup_config_v75.py'), config(BASE / 'startup_config_v75.py')
    cfg_diff = {k: dict(V85=old_cfg.get(k), P148_parent=new_cfg.get(k))
                for k in old_cfg.keys() | new_cfg.keys() if old_cfg.get(k) != new_cfg.get(k)}
    assert set(cfg_diff) == {'max_num_seqs', 'worker_timeout_seconds'}
    model_fields = {k: old_cfg[k] for k in ('draft_model_path', 'target_model_path',
        'draft_tensor_parallel_size', 'target_tensor_parallel_size', 'draft_dtype',
        'target_dtype', 'enable_mc2', 'mc2_profile')}
    for path in (N / 'runtime/mc2-profile.json',
                 N / 'source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py',
                 N / 'source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_model.py'):
        pin(path, receipt['external_input_hashes'][str(path)])
    model_cfg = read('/root/data/shared-models/Qwen3-32B/config.json')
    assert model_cfg['torch_dtype'] == 'bfloat16' and model_cfg['vocab_size'] == 151936
    old_exports = exports(OLD / 'local-startup-body.sh')
    numeric_names = sorted(k for k in old_exports if k.startswith('VLLM_ASCEND_PEARL_')
        or k in ('HCCL_DETERMINISTIC', 'HCCL_OP_EXPANSION_MODE', 'TASK_QUEUE_ENABLE',
                 'ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'PYTORCH_NPU_ALLOC_CONF'))
    comparisons, current_identity = [], []
    for run in (173, 190):
        audit = read(D / f'V{run}_INDEPENDENT_TERMINAL_FULL463_AUDIT.json')
        current_receipt = read(audit['receipt']['path'], audit['receipt']['sha256'])
        assert current_receipt['status'] == 'complete' and current_receipt['errors'] == []
        assert current_receipt['actual_execution_complete'] is True
        assert current_receipt['physical_cards'] == [2, 5, 6, 7]
        assert current_receipt['sources_unchanged'] is True
        for phase in ('pre_verifier', 'post_verifier'):
            verifier = current_receipt[phase]
            result = verifier['actual_result']
            assert verifier['exit_code'] == 0 and result['status'] == 'pass'
            assert (result['checked_frozen_files'], result['checked_external_files'], result['checked_packages']) == (10496, 15, 212)
            pin(verifier['path'], verifier['sha256'])
        env = exports(D / f'v{run}-root-execution/startup{run}.sh')
        assert all(env.get(k) == old_exports[k] for k in numeric_names)
        current_identity.append(dict(run=run, receipt=audit['receipt'], status=current_receipt['status'],
            baseline_prepost_10496_15_212_passed=True, matching_explicit_numeric_exports=True,
            additional_scheduler_exports={k: env[k] for k in env if k.startswith('VLLM_ASCEND_SPECRHYTHM_') and k not in old_exports}))
        manifest = read(D / f'V{run}_INDEPENDENT_TERMINAL_FULL463_AUDIT_RAW_MANIFEST.json')
        path = next(Path(s) for s in manifest if s.endswith('/windows/window32/full463/rank1-full463-actual-trace.json.gz'))
        data = json.loads(gzip.decompress(pin(path, manifest[str(path)]['sha256'])))
        assert data['rank'] == 1 and data['scope_count'] == len(data['generate_results']) == 463
        assert data['manifest_sha256'] == old_candidate['source']['original_input_and_decode']['manifest_sha256']
        for i, ref in enumerate(refs):
            prompt, result, sampling = data['tokenized_inputs'][i], data['generate_results'][i], data['sampling'][i]
            assert digest(prompt) == ref['prompt_sha256'] and result['request_id'] == ref['request_id']
            assert sampling == old_candidate['sampling'][i]
            controls = {k: result[k] for k in ('temperature', 'draft_temperature', 'top_p', 'top_k',
                'draft_top_p', 'draft_top_k', 'max_tokens', 'ignore_eos', 'finish_reason')}
            assert controls == dict(temperature=0., draft_temperature=0., top_p=1., top_k=0,
                draft_top_p=1., draft_top_k=0, max_tokens=256, ignore_eos=True, finish_reason='length')
            actual, oracle = result['completion_token_ids'], ref['completion_token_ids']
            assert len(actual) == len(oracle) == 256
            differences = [n for n, (x, y) in enumerate(zip(actual, oracle, strict=True)) if x != y]
            first = differences[0] if differences else None
            start, end = (max(0, first - 5), min(256, first + 6)) if first is not None else (0, 0)
            comparisons.append(dict(run=run, request_index=i, request_id=result['request_id'],
                same_original_prompt=True, prompt_tokens=len(prompt), prompt_sha256=digest(prompt),
                candidate_artifact=str(path), candidate_selector=f'generate_results[{i}].completion_token_ids',
                reference_artifact=str(OLD / f'live-progress/rank1-AR-request{i}.json'),
                reference_selector='reference_row.completion_token_ids', matching_sampling=sampling,
                output_controls=controls, token_count_each=256, exact_equal=not differences,
                candidate_completion_sha256=digest(actual), reference_completion_sha256=digest(oracle),
                first_mismatch_index_0based=first, common_prefix_tokens=256 if first is None else first,
                first_pair_candidate_reference=None if first is None else [actual[first], oracle[first]],
                total_mismatch_count=len(differences), context=dict(range_0based_halfopen=[start, end],
                    candidate=actual[start:end], reference=oracle[start:end])))
    assert all(row['exact_equal'] is False for row in comparisons)
    output = dict(schema='V85-two-own-AR-references-vs-completed173190-readonly-audit-v1',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='completed-bounded-readonly-evidence-check',
        conclusion=dict(reusable_as_pinned_original_direct_eager_PA_AR_sequences=True,
            sameM_reference_for_candidates=False, core_model_prompt_sampling_identity_matched=True,
            comparisons=4, exact_matches=0, pointwise_non_equivalence_to_these_canonical_AR_sequences=True,
            current_candidate_engine_own_AR_recomputed=False, whole463_AR_qualified=False,
            root_cause_or_one_numerical_mode_more_accurate_proved=False),
        reference_checks=reference_checks,
        identity=dict(original_baseline_lock=PINS[str(lock_path)], original_model_fields=model_fields,
            actual_template_sampling_digest_functions_AST_equal=list(names),
            config_differences_fully_enumerated=cfg_diff,
            config_difference_scope='Only service capacity32→64 and worker timeout1800→6000 differ. Not identical complete configs; neither field changes the declared one-owner direct-AR recurrence.',
            target_config_dtype='bfloat16', target_semantic_vocab=151936,
            reference_head='V85 full model logits sliced to actual shared vocab, then argmax. V86 only rechecks bounded reference-side head/margin points; it is not the source of these complete256-token rows.',
            numeric_launch_exports={k: old_exports[k] for k in numeric_names},
            current_receipts=current_identity, original_weight_identity_scope=lock['model_hash_scope'],
            weight_check_limit='Reused original lock SHA and retained pre/post verifier records; no new multi-GiB weights were hashed. Original lock records33 SHA-verified public model/tokenizer files; launch checks resolved path/size/mtime and212 packages.'),
        comparisons=comparisons,
        interpretation='Each completed candidate differs from each stored original canonical direct-eager-PA own-AR sequence for exactly matching prompt/control and pinned core model/baseline identities. Consequently its retained whole463 output cannot equal a canonical reference set containing those sequences. This is a bounded exact-output counterexample, not a same-M numerical failure or root-cause diagnosis.',
        limitations=[
            'Reference whole-prompt eager PA then one-query PA is not the candidate batched/graph causal/FULL M80/128/etc numerical ABI. Different physical shapes/history and head rows are explicit; these are not silently treated as same-M.',
            'No complete historical inherited environment dump is retained in these rows. Matching explicit numeric exports, unchanged core source/profile/lock and saved package/model pre/post checks support the bounded same-baseline comparison, not portable golden outputs across arbitrary environments.',
            'Only two complete reference requests/512 logical tokens exist. Their disagreement refutes exact equality to those sequences, but provides no complete463 reference, arbitrary-input proof or assurance about unexamined rows.',
            'Candidate hidden/KV and branch acceptance cause were not reconstructed. Reference-only margins cannot establish a BF16/MC2/NZ14/attention/protocol root cause. Candidate KV was never used as oracle.',
            'No existing failed V85 status, current whole_AR=False flag, frozen source or actual state was modified.'
        ],
        performed_now=dict(completed_evidence_only=True, current192_raw_or_receipt_read=False,
            Torch_imported=False, NPU_queried=False, models_or_old_suites_run=False,
            existing_sources_reports_or_states_modified=False), inputs=PINS)
    path = D / 'V85_TWO_OWN_AR_REFERENCES_VS_COMPLETED173190_BOUNDED_REUSE_AUDIT.json'
    with path.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size, comparisons=[{k:r[k] for k in ('run','request_index','first_mismatch_index_0based','first_pair_candidate_reference')} for r in comparisons])))

if __name__ == '__main__':
    main()
