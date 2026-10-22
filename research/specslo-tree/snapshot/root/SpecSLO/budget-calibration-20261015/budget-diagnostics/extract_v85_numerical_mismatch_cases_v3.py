#!/usr/bin/env python3
"""HOST source fixtures only: retained public tokenizer, no weights/devices/network."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-live-prefix-page-protection-v85'
D = B/'budget-diagnostics'
OUT = D/'V85_NUMERICAL_MISMATCH_CASES.json'
AUDIT = D/'ROOT_V85_ACTUAL_CANDIDATE_AR_MISMATCH_AUDIT.json'
AUDIT_SHA = 'c034f7ec813456d550b854a3cec0c201bd168283adf67e26579fa6f0a9a26e9b'
TOKENIZER = Path('/root/data/shared-models/Qwen3-0.6B')
RETAINED = Path('/root/miniconda3/envs/vllm-hust-dev/bin/python3.11')
opened_tokenizer_files, forbidden_attempts = set(), []


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(tokens):
    return sha(json.dumps(list(tokens), separators=(',', ':')).encode())


def read(path):
    return json.loads(Path(path).read_bytes())


def guard(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
        forbidden_attempts.append(event)
        raise RuntimeError('Extractor forbids network/subprocess: '+event)
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt'):
            forbidden_attempts.append('weight-file:'+str(path))
            raise RuntimeError('Extractor forbids weight files')
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(char in mode for char in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and path != OUT and (path.is_relative_to(P) or path.is_relative_to(TOKENIZER.resolve())):
            forbidden_attempts.append('write:'+str(path))
            raise RuntimeError('Extractor permits writing only its exclusive fixture')
        if path.is_relative_to(TOKENIZER.resolve()):
            opened_tokenizer_files.add(str(path))


def profile(frame, event, function):
    module, name = (frame.f_globals.get('__name__', '') or ''), frame.f_code.co_name
    if event == 'call' and ((module.startswith(('torch_npu.npu', 'torch.cuda')) and name in ('init', '_lazy_init'))
            or (module.endswith('.native_engine') and name == '__init__')
            or (module.startswith('transformers.modeling_utils') and name == 'from_pretrained')
            or (module == 'torch.serialization' and name == 'load')):
        forbidden_attempts.append(module+'.'+name)
        raise RuntimeError('Extractor forbids model/device/weight initialization')
    if event == 'c_call' and getattr(function, '__name__', '') == '_npu_init':
        forbidden_attempts.append('_npu_init')
        raise RuntimeError('Extractor forbids device initialization')


def actual_raw(path, audit, hashes):
    raw = path.read_bytes()
    expected = audit['actual_raw_inputs'][str(path)]
    require(sha(raw) == expected, 'Actual retained raw differs: '+str(path))
    hashes[str(path)] = expected
    return json.loads(raw)


def extract_case(owner, step, snapshots, refs, prompts):
    snap = snapshots[1]
    result = snap['generate_results'][owner]
    output = result['completion_token_ids']
    reference = refs[owner]
    ar = reference['completion_token_ids']
    require(output[:step] == ar[:step] and output[step] != ar[step], 'Case is not the exact first mismatch')
    published = {i:[] for i in range(48)}
    found = None
    for stream_index, event in enumerate(snap['stream_events']):
        request = event['request_index']
        start = len(published[request])
        if request == owner and start <= step < start+len(event['token_ids']):
            found = (stream_index, event, start, start+len(event['token_ids']))
            break
        published[request].extend(event['token_ids'])
    require(found is not None, 'Mismatch stream event is missing')
    stream_index, event, start, end = found
    context = len(prompts[owner])+start-1
    matches = [(i,row) for i,row in enumerate(snap['begin_inputs'])
        if owner in row['target_ids'] and row['contexts'][row['target_ids'].index(owner)] == context]
    require(len(matches) == 1, 'Actual begin inputs do not uniquely identify this context')
    begin_index, begin = matches[0]
    for rank in (0,2,3):
        require(snapshots[rank]['begin_inputs'][begin_index] == begin, 'All4 actual begin inputs differ')
    normal = [(i,row) for i,row in enumerate(snap['normal_target_calls'])
        if owner in row['sequence_ids'][:row['N']]
        and row['contexts'][row['sequence_ids'].index(owner)] == context]
    plans = [row for row in snap['logical_events'] if row['kind'] == 'plan' and row['cycle_id'] == begin['cycle_id']]
    require((len(normal),len(plans)) in ((1,0),(0,1)), 'Actual normal/mixed source is ambiguous')
    histories = []
    for target, budget, prefix in zip(begin['target_ids'],begin['budgets'],begin['contexts'],strict=True):
        count = prefix+1-len(prompts[target])
        completed = snap['generate_results'][target]['completion_token_ids'][:count]
        require(count >= 1 and len(completed) == count and published[target] == completed,
            'Actual context does not match already published candidate prefix: '+str(target))
        history = prompts[target]+completed
        require(len(history) == prefix+1, 'Actual root position differs')
        histories.append(dict(owner=target, B=budget, context=prefix,
            prompt_tokens=prompts[target], already_published_candidate_prefix=completed,
            prompt_plus_published_candidate_prefix_and_root=history,
            protected_prefix_token_count=prefix, root_position=prefix, root_token_id=history[-1],
            history_sha256=digest(history)))
    pending = begin['owner']
    pending_prompt = dict(owner=pending, identity=begin['identity'], committed_prompt_cursor=begin['cursor'],
        prompt_tokens=prompts[pending], original_prompt_prefix=prompts[pending][:begin['cursor']],
        prompt_sha256=digest(prompts[pending]))
    missing = ['full selected draft tree token rows for every target owner',
        'exact selected parent_indices for every target owner',
        'actual plan positions/cache_positions and accepted-node indices',
        'candidate target hidden/logits and original KV bytes']
    if normal:
        normal_index, metadata = normal[0]
        for rank in (2,3):
            require(snapshots[rank]['normal_target_calls'][normal_index] == metadata, 'Target3 actual normal metadata differ')
        mode, proposals = 'normal-target', None
        missing.append('normal-path proposal IDs: not retained in begin_inputs or normal_target_calls')
        runtime = dict(normal_target_call_index=normal_index, actual_normal_target_call=metadata,
            pending_prompt_did_not_execute_mixed_this_cycle=True)
    else:
        plan = plans[0]
        commits = [row for row in snap['logical_events'] if row['kind'] == 'commit' and row['cycle_id'] == begin['cycle_id']]
        require(len(commits) == 1 and plan['start'] == begin['cursor'] and plan['budgets'] == begin['budgets']
            and plan['target_ids'] == begin['target_ids'], 'Actual mixed commit metadata differs')
        mode, proposals = 'mixed-target', plan['proposal_ids']
        partition = plan['query_partition']
        require(sum(partition) == 197 and len(partition) == 19, 'Actual mixed extent differs')
        pending_prompt.update(actual_chunk_token_ids=prompts[pending][plan['start']:plan['end']],
            chunk_start=plan['start'],chunk_end=plan['end'],final=plan['final'])
        runtime = dict(actual_mixed_plan=plan,actual_mixed_commit=commits[0],
            physical_M=sum(partition),physical_S=len(partition),physical_query_partition=partition)
        missing.append('actual scratch sequence IDs/pages: query partition alone does not retain allocator identities')
    require((owner,start,context,begin['cycle_id']) == ((0,246,331,156) if owner == 0 else (1,34,298,21)),
        'Independent case localization differs from the requested actual case')
    own_call = reference['calls'][step]
    require(own_call['input_sha256'] == digest([ar[step-1]])
        and own_call['start'] == len(prompts[owner])+step-1, 'AR first-mismatch input differs')
    shared = prompts[owner]+ar[:step]
    return dict(case_id=f'owner{owner}-step{step}',request_index=owner,request_id=result['request_id'],
        step=step,prompt_tokens=prompts[owner],ref_output_prefix=ar[:step],
        candidate_diff_id=output[step],reference_diff_id=ar[step],
        first_difference_verified=True,matching_output_prefix_tokens=step,
        reference_output_256_sha256=digest(ar),candidate_output_256_sha256=digest(output),
        shared_original_input_history_through_previous_token=shared,
        shared_original_input_last_position=len(shared)-1,actual_AR_first_mismatch_call=own_call,
        stream_event_global_index=stream_index,stream_output_start=start,stream_output_end_exclusive=end,
        actual_stream_event=event,actual_begin_input_index=begin_index,actual_begin_input=begin,
        execution_path=mode,target_ids=begin['target_ids'],budgets=begin['budgets'],contexts=begin['contexts'],
        proposal_ids=proposals,actual_forward_metadata=runtime,target_owner_histories=histories,
        pending_prompt=pending_prompt,missing_for_exact_original_tree_replay=missing,
        source_fixture_only=True,numerically_qualified=False)


def main():
    require(len(sys.argv) == 1 and sys.version_info[:2] == (3,11)
        and Path(sys.executable).resolve() == RETAINED.resolve() and sys.dont_write_bytecode,
        'Use retained Python3.11 -B, no arguments')
    require(not OUT.exists(), 'Refusing to overwrite the retained fixture')
    require(sha(AUDIT.read_bytes()) == AUDIT_SHA, 'Pinned actual failure audit differs')
    audit = read(AUDIT)
    require(audit['all_checks_passed'] is True and audit['status'] == 'failure-evidence-verified'
        and audit['original_job_status'] == 'failed', 'Actual failure evidence was not accepted')
    pin = read(D/'ROOT_V85_FINAL_CPU_AND_SOURCE_PIN_AUDIT.json')
    freeze_path = P/'PREPARED_SOURCE_HASHES_V85.json'
    require(sha(freeze_path.read_bytes()) == pin['identity']['freeze_sha256'], 'Actual source freeze differs')
    sources = read(freeze_path)['files']
    require(len(sources) == 198 and all(sha(Path(path).read_bytes()) == wanted for path,wanted in sources.items()),
        'Actual frozen source bytes differ')
    hashes = {}
    snapshots = [actual_raw(P/'live-progress'/f'rank{rank}-candidate-before-AR.json',audit,hashes) for rank in range(4)]
    refs = {}
    for owner in (0,1):
        records = [actual_raw(P/'live-progress'/f'rank{rank}-AR-request{owner}.json',audit,hashes)
            for rank in (1,2,3)]
        require(records[0]['reference_row'] == records[1]['reference_row'] == records[2]['reference_row'],
            'Actual independent AR target3 records differ')
        refs[owner] = records[0]['reference_row']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    sys.addaudithook(guard)
    sys.setprofile(profile)
    sys.path.insert(0,str(P))
    original = importlib.import_module('poisson_coverage_probe')
    require(Path(original.__file__).resolve() == P/'poisson_coverage_probe.py', 'Tokenizer formatter source differs')
    from transformers import AutoTokenizer, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
    rows = original._manifest_rows(original.MANIFEST.read_bytes())
    prompts, verified = original.tokenize_prefix(tokenizer,rows)
    require(len(prompts) == 48 and all(verified == snapshot['tokenized_inputs'] for snapshot in snapshots),
        'Actual48 tokenizer hashes/lengths differ from the retained run')
    cases = [extract_case(owner,step,snapshots,refs,prompts) for owner,step in ((0,250),(1,34))]
    sys.setprofile(None)
    resources = {}
    for name in ('config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt'):
        path = TOKENIZER/name
        require(path.is_file(), 'Missing local tokenizer resource: '+name)
        resources[str(path.resolve())] = sha(path.read_bytes())
    require(not forbidden_attempts and all(sha(Path(path).read_bytes()) == wanted for path,wanted in hashes.items())
        and sha(AUDIT.read_bytes()) == AUDIT_SHA, 'Original inputs changed or forbidden operation attempted')
    report = dict(schema='V85.numerical-mismatch-source-fixtures.v1',created_UTC=datetime.now(timezone.utc).isoformat(),
        source_fixture_only=True,numerically_qualified=False,independent_AR_qualified=False,
        full48_qualified=False,all65_qualified=False,full463_qualified=False,online_qualified=False,formal_Goodput=False,
        NPU_or_model_work_performed=False,forbidden_operation_attempts=forbidden_attempts,
        extraction_script=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__).read_bytes())),
        failure_audit=dict(path=str(AUDIT),sha256=AUDIT_SHA,original_job_status='failed'),
        source_sha256=sources,raw_input_sha256=hashes,
        tokenizer=dict(path=str(TOKENIZER),resolved_path=str(TOKENIZER.resolve()),
            class_name=type(tokenizer).__name__,transformers_version=transformers_version,
            local_files_only=True,format_function='poisson_coverage_probe.tokenize_prefix',
            original_Native_tokenizer_source='native_engine.py:4156 uses config.draft_model',
            resource_sha256=resources,observed_opened_local_files=sorted(opened_tokenizer_files)),
        all48_prompt_hashes_and_lengths_verified=verified,cases=cases,
        exact_original_tree_replay_possible_from_this_fixture_alone=False)
    raw = (json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
    descriptor = os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'wb') as stream:
        stream.write(raw)
    print(str(OUT))
    print('SHA256 '+sha(raw))
    print('All48 original prompt hashes/lengths exact; source fixtures only; no model/weights/NPU/network.')


if __name__ == '__main__':
    main()
