"""Unscored evidence checker. Never read actual/receipt without ROOT terminal SHA.

Standard library only; this does not launch or qualify any model or device.
"""
import argparse, ast, collections, copy, datetime, hashlib, importlib.abc
import json, math, re, struct, sys, tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'canonical-ar-common-prefix-diagnostic-v1'
ACTION = P.name
SCHEMA = ACTION + '.v1'
FREEZE_SHA = '5fe879b5f8cb7f8fe452ff64bb6479dea481229be780147a174fe6895f631df3'
ROOT_SHA = '8f1b0652b533387de0f3f7d80e943528e931e19470e9b8f093ba929c630ed62a'
PEER_SHA = 'b4ca32be4137c301d572645f2fb6b377de696d830059f2aa65d3cde78671dbc6'
ARCHIVE_SHA = '3e6c378415a2635f635cb56a37e049a41441a934b2bfa7a850aa7743470330b2'
FLAGS = dict(production_qualified=False, whole463_AR_qualified=False,
             numerical_losslessness_proved=False, formal_Goodput=False, adopted=False)
BACKENDS = ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'acl', 'safetensors')

def need(value, text):
    if not value: raise ValueError(text)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def desc(path):
    p = Path(path)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

def digest(values):
    return hashlib.sha256(json.dumps(list(values), separators=(',', ':')).encode()).hexdigest()

def function(path, name, namespace):
    nodes = [n for n in ast.walk(ast.parse(Path(path).read_text()))
             if isinstance(n, ast.FunctionDef) and n.name == name]
    need(len(nodes) == 1, 'one actual pure original function ' + name)
    node = copy.deepcopy(nodes[0]); node.decorator_list = []
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path) + '::independent-HOST-only', 'exec'), namespace)
    return namespace[name]

class Firewall(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BACKENDS:
            self.attempts.append(name)
            raise ImportError('Common-prefix terminal HOST forbids ' + name)

def source_evidence():
    freeze_path = P / 'FROZEN_DIAGNOSTIC.json'
    need(sha(freeze_path) == FREEZE_SHA, 'exact ROOT freeze')
    f = read(freeze_path); root_path = Path(f['root_review']['path'])
    peer_path = Path(f['peer_review']['path']); ready_path = Path(f['ready']['path'])
    need(sha(root_path) == f['root_review']['sha256'] == ROOT_SHA and
         sha(peer_path) == f['peer_review']['sha256'] == PEER_SHA and
         sha(ready_path) == f['ready']['sha256'], 'exact ROOT/peer/ready pins')
    root, ready = read(root_path), read(ready_path)
    need(root['status'] == 'passed_source_and_retained_HOST_review' and
         f['ROOT_source_reviewed'] is True and all(f[k] is False for k in FLAGS), 'source-only approval scope')
    need(len(f['sources']['local']) == 10 and len(f['sources']['external']) == 52, 'actual revised10/52 source schema')
    flat = {**f['sources']['local'], **f['sources']['external']}
    cpu = read(P / 'CPU_RESULT.json'); prepared = read(P / 'PREPARED.json')
    need(cpu['source_before'] == cpu['source_after'] == f['sources'] == prepared['sources'] and
         sha(P / 'CPU_RESULT.json') == f['CPU_sha256'] == prepared['CPU_sha256'], 'exact current CPU source bindings')
    need(cpu['status'] == 'passed' and cpu['tests_run'] == 3 and cpu['errors'] == cpu['failures'] == 0 and
         cpu['no_Torch_import'] is True and cpu['NPU_used'] is cpu['models_run'] is cpu['old_suites_rerun'] is False,
         'actual revised signal HOST schema')
    old = cpu['inherited_original4_HOST']
    need(sha(old['path']) == old['sha256'] and old['tests_run'] == 4 and old['rerun'] is False and
         old['unchanged_core_except_guardian_verified'] is True and read(old['path'])['status'] == 'passed', 'retained original4 HOST')
    pins = root['exact_verified_pins']
    need(pins == ready['pins'] and len(pins) == 111 and all(pins[k] == v for k, v in flat.items()), 'ROOT exact111 closure')
    need(all(sha(k) == v for k, v in pins.items()), 'all current selected source/evidence bytes')
    archive = Path(f['source_archive']['path'])
    need(sha(archive) == f['source_archive']['sha256'] == ARCHIVE_SHA, 'exact114-member archive binary')
    selected = {**pins, str(root_path): ROOT_SHA, str(peer_path): PEER_SHA, str(ready_path): sha(ready_path)}
    need(len(selected) == f['source_members'] == 114, 'exact114 projected member map')
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        need(len(members) == 114 and len({m.name for m in members}) == 114, 'archive unique regular inventory')
        for m in members:
            key = '/' + m.name.lstrip('/')
            need(m.isfile() and key in selected and hashlib.sha256(bundle.extractfile(m).read()).hexdigest() == selected[key], 'regular archive bytes ' + m.name)
    return dict(freeze=desc(freeze_path), ROOT_review=desc(root_path), peer=desc(peer_path),
                ready=desc(ready_path), CPU=desc(P / 'CPU_RESULT.json'), archive=dict(**desc(archive), members=114),
                selected_SHA_checked=111, no_original_HOST_suite_or_verifier_rerun=True)

def saved_baseline(row):
    need(row['exit_code'] == 0 and sha(row['path']) == row['sha256'], 'saved original baseline log SHA/exit')
    text = Path(row['path']).read_text(); decoder = json.JSONDecoder(); values = []
    for i, character in enumerate(text):
        if character != '{': continue
        try: value, end = decoder.raw_decode(text[i:])
        except ValueError: continue
        if type(value) is dict and 'checked_frozen_files' in value: values.append(value)
    need(len(values) == 1, 'one actual original verifier result')
    value = values[0]
    need(value['status'] == 'pass' and value['errors'] == [] and
         (value['checked_frozen_files'], value['checked_external_files'], value['checked_packages']) == (10496, 15, 212), 'original10496/15/212 prepost')
    return dict(**desc(row['path']), actual_result=value)

def terminal_evidence(receipt):
    need(receipt['schema'] == 'canonical-ar-common-prefix-diagnostic.receipt.v1' and
         receipt['physical_cards'] == [2, 5, 6, 7] and receipt['execution_lock'] == str(B / '.ROOT_NPU_EXECUTION.lock'), 'actual self-lock/card identity')
    need(receipt['sources'] == read(P / 'FROZEN_DIAGNOSTIC.json')['sources'] and
         receipt['sources_unchanged'] is True and receipt['owned_remaining'] == [], 'source continuity and owned cleanup')
    baseline = {k: saved_baseline(receipt[k]) for k in ('pre_verifier', 'post_verifier')}
    for key in ('devices_before', 'devices_after'):
        v = receipt[key]
        need(v['all_idle'] is v['four_910B2'] is True and set(v['processes']) == {'2', '5', '6', '7'} and
             sorted(map(tuple, v['parsed_hardware'])) == [(2, '910B2'), (5, '910B2'), (6, '910B2'), (7, '910B2')], 'saved all4 hardware/idle')
        need(v['hardware']['exit_code'] == 0 and v['hardware']['timed_out'] is False and
             sorted((int(device), name) for device, name in re.findall(
                 r'^\|\s*(\d+)\s+(\S+)\s*\|', v['hardware']['output'], re.M) if not name.isdecimal()) ==
             [(2, '910B2'), (5, '910B2'), (6, '910B2'), (7, '910B2')], 'saved original hardware output independently parsed')
        for process in v['processes'].values():
            need(process['exit_code'] == 0 and process['timed_out'] is False and process['idle'] is True and
                 re.search(r'No process in device\.|No running processes found', process['output'], re.I) is not None, 'saved original idle probe')
    need(all(receipt[k] is False for k in FLAGS) and receipt['termination_signals'] == [], 'unscored scope and no termination')
    need(type(receipt['private_bytes']) is int and 0 <= receipt['private_bytes'] <= 64 * 1024 ** 2 and
         math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds'] > 0, 'actual storage polling and elapsed schema')
    return dict(baseline=baseline, owned_remaining=[], all4_idle=True,
                lock_policy='frozen guardian obtains ROOT LOCK_EX|LOCK_NB before devices; no outer lock',
                declared_polling_limits=dict(seconds=1800, private_bytes=64 * 1024 ** 2),
                cleanup_included_in_terminal_elapsed=True, exact_peak_private_bytes_not_retained=True)

def input_record(row, tokens, positions, *, fia):
    need(type(row) is dict and row['input_sha256'] == digest(tokens) and row['position_sha256'] == digest(positions) and row['M'] == len(tokens), 'actual input/position digest and M')
    if fia:
        need(row['M'] == 80 and row['S'] == 16 and row['query_cumulative'] == list(range(5, 81, 5)), 'actual FIA80/S16')
        need(row['causal_mask_dtype'] == 'torch.int8' and row['causal_mask_shape'] == [2048, 2048], 'original cached causal mask ABI')
    else:
        need(row['S'] == 1 and row['query_cumulative'] == [len(tokens)] and row['causal_mask_dtype'] is None and row['causal_mask_shape'] is None, 'actual direct PA one-segment metadata')
    need(row['use_fused_infer_attention'] is fia and row['tree_attention'] is False and row['context_lengths'] == [p + 1 for p in positions], 'actual causal/PA contexts')
    slots = row['cache_slots']; need(len(slots) == len(tokens) and len(set(slots)) == len(slots), 'actual unique query slots')
    pages = {}; owners = [i // 5 for i in range(80)] if fia else [0] * len(tokens)
    for owner, position, slot in zip(owners, positions, slots):
        need(type(slot) is int and slot >= 0 and slot % 128 == position % 128, 'original128-page physical offset')
        key = (owner, position // 128)
        need(key not in pages or pages[key] == slot // 128, 'same logical page stable inside actual call')
        pages[key] = slot // 128
    need(len(set(pages.values())) == len(pages), 'observed owner/logical-page query slots distinct')

def rounded(value, dtype):
    if dtype == 'torch.float64': return value
    if dtype == 'torch.float32': return struct.unpack('f', struct.pack('f', value))[0]
    if dtype == 'torch.float16': return struct.unpack('e', struct.pack('e', value))[0]
    need(dtype == 'torch.bfloat16', 'explicit retained logits dtype')
    bits = struct.unpack('I', struct.pack('f', value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('f', struct.pack('I', bits))[0]

def head_record(snapshot, tokens, positions, *, fia):
    count = 16 if fia else 1
    need(snapshot['physical_model_rows'] == len(tokens) and snapshot['physical_head_rows'] == (80 if fia else 1) and
         snapshot['restricted_vocabulary'] == 151936 and type(snapshot['LM_head_pad_multiple']) is int and snapshot['LM_head_pad_multiple'] > 0,
         'actual full80-head-before-root or selected PA-head shape')
    need(snapshot['hidden_logits_finite'] is snapshot['original_WORLD_finite'] is True, 'retained actual finite gates')
    need(snapshot['hidden_dtype'] in ('torch.bfloat16', 'torch.float16', 'torch.float32', 'torch.float64'), 'actual hidden dtype')
    need(snapshot['logits_dtype'] in ('torch.bfloat16', 'torch.float16', 'torch.float32'),
         'retained float32 top2 permits exact arithmetic only for original logits no wider than float32')
    input_record(snapshot['actual_input'], tokens, positions, fia=fia)
    need(all(len(snapshot[k]) == count for k in ('tokens', 'top2_ids', 'top2_logits', 'margins')), 'actual chosen-root signal lengths')
    ties = 0
    for token, ids, scores, margin in zip(snapshot['tokens'], snapshot['top2_ids'], snapshot['top2_logits'], snapshot['margins']):
        need(type(token) is int and 0 <= token < 151936 and len(ids) == len(scores) == 2 and
             all(type(v) is int and 0 <= v < 151936 for v in ids) and ids[0] != ids[1], 'top2 vocabulary and identity')
        need(all(type(v) in (int, float) and math.isfinite(v) for v in (*scores, margin)) and scores[0] >= scores[1] and margin >= 0, 'finite ordered root scores/margin')
        need(margin == rounded(scores[0] - scores[1], snapshot['logits_dtype']), 'recorded dtype-rounded top2 margin arithmetic')
        if margin > 0: need(token == ids[0], 'unique argmax agrees with top2 maximum')
        else: ties += 1
    return dict(chosen_rows=count, exact_top2_margin_arithmetic=True, exact_tie_rows=ties,
                tie_argmax_not_inferred_from_only_two_retained_logits=True)

def graph_state(state, *, after):
    need(type(state) is dict and set(state) == {'execution', 'actual_entries', 'disabled_keys'}, 'actual Native state schema')
    ex = state['execution']; need(set(ex) == {'mode', 'fallback_reason', 'capture_attempted', 'replay_executed'} and
        ex['mode'] in ('not_run', 'eager', 'replay', 'capture_replay') and type(ex['capture_attempted']) is type(ex['replay_executed']) is bool and
        (ex['fallback_reason'] is None or type(ex['fallback_reason']) is str), 'actual four-field NativeGraphExecution')
    if after: need(ex['mode'] != 'not_run', 'actual graph request outcome recorded')
    if ex['mode'] in ('replay', 'capture_replay'): need(ex['replay_executed'] is True and ex['fallback_reason'] is None, 'returned graph outcome semantics')
    if ex['mode'] == 'capture_replay': need(ex['capture_attempted'] is True, 'natural capture flag')
    entries = state['actual_entries']; need(len({v['key'] for v in entries}) == len(entries) and
        all(type(v['key']) is str and type(v['runtime_validated']) is bool for v in entries), 'actual entry keys/validation Booleans')
    need(type(state['disabled_keys']) is list and all(type(v) is str for v in state['disabled_keys']) and
         len(set(state['disabled_keys'])) == len(state['disabled_keys']), 'actual disabled-key inventory')
    return ex

def report_case(report, case, rank):
    need(report['schema'] == SCHEMA and report['rank'] == rank and report['fixture_sha256'] == sha(P / 'FIXTURE.json') and
         report['request_index'] == case['request_index'] and report['request_id'] == case['request_id'], 'actual rank/case/fixture identity')
    fn = function(P / 'ar_margin_actor.py', 'audit_case_report', dict(require=need, FLAGS=FLAGS))
    fn(report, case)
    own = report['canonical_own_AR']; outputs, calls = own['outputs'], own['calls']
    need(len(outputs) == len(calls) == case['step'] + 1 and all(type(v) is int and 0 <= v < 151936 for v in outputs), 'actual own-AR length/vocabulary')
    prompt = case['prompt_tokens']
    for step, row in enumerate(calls):
        inp = prompt if step == 0 else [outputs[step - 1]]
        positions = list(range(len(prompt))) if step == 0 else [len(prompt) + step - 1]
        need(row == dict(step=step, input_sha256=digest(inp), positions=[positions[0], positions[-1] + 1]), 'independent actual self-recurrence hash/positions')
    routes = report['independent_teacher_forced_common_prefix_routes']; need(len(routes) == 4, 'four independently allocated route records')
    prefix = case['common_prefix_tokens']; length = len(prefix); root_tokens = [prefix[-1], 0, 0, 0, 0] * 16
    root_positions = list(range(length - 1, length + 4)) * 16
    result = dict(case_id=case['case_id'], rank=rank, own_steps=len(outputs), original_audit_case_report_reexecuted=True)
    if rank == 0:
        need(own['terminal'] is None, 'draft role does not claim target logits')
    else:
        inp, pos = [outputs[-2]], [len(prompt) + case['step'] - 1]
        result['own_terminal_signals'] = head_record(own['terminal'], inp, pos, fia=False)
        need(own['terminal']['tokens'] == [outputs[-1]], 'actual own last head drives its own next output')
        diff = next((i for i, (a, b) in enumerate(zip(outputs, case['reference_outputs'])) if a != b), None)
        need(own['first_reference_difference'] == diff and own['pinned_reference_prefix_reproduced'] is (diff is None), 'independent retained-reference first difference')
        result['own_reference_first_difference'] = diff
    for i, route in enumerate(routes):
        need(route['own_fresh_cache'] is True and route['input_prefix_sha256'] == digest(prefix), 'actual fresh common-prefix route identity')
        if i < 2:
            need(route['route'] == 'direct eager PA' and route['prefix_rebuild'] == ('whole-prefix' if i == 0 else 'whole-prefix-minus-root thenroot'), 'actual PA reconstruction route')
            if rank == 0: need(route['snapshot'] is None, 'draft role PA joins only')
            else: head_record(route['snapshot'], prefix if i == 0 else [prefix[-1]], list(range(length)) if i == 0 else [length - 1], fia=False)
            continue
        need(route['cohort'] == dict(N=16, M=80, S=16, prefix=length - 1, root=prefix[-1], candidates=[0]*4,
             owners=list(range(16)), roots_at=list(range(0, 80, 5)), cohort='reconstructed legal N16 B4, identical common prefix per independent owner') and
             route['all16_prefixes_independently_computed'] is route['logits_from_full_physical_head'] is route['graph_fallback_or_natural_flags_not_overridden'] is True and
             route['historical_packed_call_replayed'] is False, 'actual reconstructed legal cohort, not historical candidate-KV reuse')
        need(route['route'] == ('original direct eager FIA' if i == 2 else 'original natural hidden andgreedy graph requests'),
             'actual eager and natural graph route identity')
        phases = ['eager'] if i == 2 else ['first', 'changed', 'restored']; rows = route['hidden_counterpart_logit_snapshots']
        need([v['phase'] for v in rows] == phases, 'actual eager or hidden F/T/F phase order')
        for row in rows:
            tokens = list(root_tokens)
            if row['phase'] == 'changed': tokens[0] = (tokens[0] + 1) % 151936
            if rank == 0: need(row['actual_input'] is row['snapshot'] is row['before'] is row['after'] is None, 'draft role FIA joins only')
            else:
                input_record(row['actual_input'], tokens, root_positions, fia=True)
                head_record(row['snapshot'], tokens, root_positions, fia=True)
                need(row['snapshot']['actual_input'] == row['actual_input'], 'same actual model and signal input')
                if i == 3: graph_state(row['before'], after=False); graph_state(row['after'], after=True)
                else: need(row['before'] is row['after'] is None, 'eager route does not invent graph state')
        greedy = route['greedy_graph_results']; need(len(greedy) == (3 if i == 3 else 0), 'actual natural greedy phase count')
        if i == 3:
            if rank == 0: need(greedy == [None, None, None], 'draft role greedy joins only')
            else:
                need([v['phase'] for v in greedy] == phases, 'actual greedy F/T/F order')
                for row in greedy:
                    tokens = list(root_tokens)
                    if row['phase'] == 'changed': tokens[0] = (tokens[0] + 1) % 151936
                    need(len(row['physical_tokens']) == 80 and all(type(v) is int and 0 <= v < 151936 for v in row['physical_tokens']) and row['original_WORLD_finite'] is True, 'all80 retained greedy tokens')
                    input_record(row['actual_input'], tokens, root_positions, fia=True); graph_state(row['actual'], after=True)
                result['hidden_first_restored_root_tokens_equal'] = rows[0]['snapshot']['tokens'] == rows[2]['snapshot']['tokens']
                result['greedy_first_restored_all80_equal'] = greedy[0]['physical_tokens'] == greedy[2]['physical_tokens']
                result['hidden_graph_execution_modes'] = [v['after']['execution']['mode'] for v in rows]
                result['greedy_graph_execution_modes'] = [v['actual']['execution']['mode'] for v in greedy]
                result['hidden_root_vs_greedy_root_first_differences'] = [next((j for j, (a, b) in enumerate(zip(h['snapshot']['tokens'], g['physical_tokens'][::5])) if a != b), None) for h, g in zip(rows, greedy)]
    return result

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--terminal-authorized', action='store_true', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); need(args.output.parent.resolve() == D and not args.output.exists(), 'fresh privateD output')
    receipt_path = P / 'receipt.json'; need(sha(receipt_path) == args.expected_receipt_sha256, 'ROOT explicit terminal receipt SHA')
    receipt = read(receipt_path); need(receipt['status'] in ('complete', 'failed'), 'running/future receipt forbidden')
    sys.meta_path.insert(0, Firewall())
    result = dict(schema='canonical-common-prefix.independent-terminal-HOST.v1', job_status=receipt['status'],
                  receipt=desc(receipt_path), script=desc(__file__), errors=[], **FLAGS)
    try:
        result['provenance'] = source_evidence(); result['terminal'] = terminal_evidence(receipt)
        need(receipt['status'] == 'complete' and receipt['errors'] == [] and receipt['child_exit_code'] == 0 and receipt['diagnostic_execution_complete'] is True, 'actual successful diagnostic, no reclassification of failed job')
        actual = P / 'actual'; paths = sorted(actual.rglob('*'))
        need(actual.resolve() == actual and not actual.is_symlink() and not any(p.is_symlink() for p in paths), 'private actual regular files')
        result['actual_files'] = {str(p): desc(p) for p in paths if p.is_file()}
        fixture = read(P / 'FIXTURE.json')
        validate = function(P / 'contract.py', 'validate_fixture', dict(require=need, digest=digest, SCHEMA=SCHEMA))
        cases = validate(fixture); summaries = [read(actual / f'rank{r}-margin-summary.json') for r in range(4)]
        complete = read(actual / 'completed.json'); exited = read(actual / 'worker-exit.json')
        need(exited['closed'] is exited['all_exited'] is True and all(exited[k] is False for k in FLAGS), 'all workers exited')
        need(complete['status'] == 'unscored-diagnostic-complete' and complete['teacher_forced_routes_are_not_own_AR'] is True and complete['case_ids'] == [c['case_id'] for c in cases] and all(complete[k] is False for k in FLAGS), 'actual bounded completion scope')
        replies = complete['replies']; need(sorted(v[1] for v in replies) == list(range(4)) and all(v[0] == 'calibrated' for v in replies), 'all4 completion replies')
        result['case_audits'] = []
        for rank, summary in enumerate(summaries):
            need(summary['schema'] == SCHEMA and summary['rank'] == rank and summary['action'] == ACTION and summary['fixture_sha256'] == sha(P / 'FIXTURE.json') and summary['enforce_eager'] is summary['precompile_decode_graphs'] is summary['precompile_serial_draft_graphs'] is False, 'actual original worker constructor/summary')
            need(all(summary[k] is False for k in ('candidate_logits_measured', 'candidate_KV_measured', 'numerical_qualification', 'online_qualified', 'cost_or_Goodput')), 'original worker does not issue qualification')
            need(summary['before']['aclgraph_entries'] == summary['before']['aclgraph_target_entries'] == 0, 'no inherited Native inventory')
            need(summary['cases'] == [dict(request_index=c['request_index'], status='diagnostic_complete') for c in cases],
                 'all four bounded cases actually completed in original worker order')
            need(next(v[2] for v in replies if v[1] == rank) == summary, 'actual IPC summary equals saved summary')
        for case in cases:
            reports = [read(actual / f'rank{r}-{case["case_id"]}.json') for r in range(4)]
            for rank, report in enumerate(reports): result['case_audits'].append(report_case(report, case, rank))
            need(all(r['canonical_own_AR']['outputs'] == reports[0]['canonical_own_AR']['outputs'] for r in reports), 'all4 true own-input recurrence')
            for key in ('canonical_own_AR', 'independent_teacher_forced_common_prefix_routes'):
                need(reports[1][key] == reports[2][key] == reports[3][key], 'three target ranks full observation agreement')
        result['original_audit_case_report'] = dict(source=desc(P / 'ar_margin_actor.py'), calls=16,
            scope='Original function checks case/status/common-prefix identity and unqualified flags; independent recurrence/metadata/signal checks above are separate.')
        result['diagnostic_log'] = desc(P / 'model-diagnostic.log')
    except Exception as error: result['errors'].append(dict(type=type(error).__name__, message=str(error)))
    need(not Firewall.attempts and not any(n.split('.')[0] in BACKENDS for n in sys.modules), 'no backend imported')
    result['status'] = 'passed_unscored_terminal_evidence_audit' if not result['errors'] else 'terminal_evidence_with_audit_errors'
    result['limitations'] = ['Own-AR outputs/calls prove retained self-input recurrence only through the four bounded steps, not whole463 equivalence.',
        'FIA80 logits were projected at full80 before selecting16 root rows. Only16 root top2/logits and all80 natural-greedy IDs are retained, not all80 logits or hidden/KV tensors.',
        'HOST recomputes stored integers/hashes/metadata and dtype-rounded margin arithmetic, not model logits, hidden/KV math or finite predicates. These remain explicitly retained actual runtime gates.',
        'Fresh teacher-forced routes use pinned common prefixes even if current own-AR first diverges earlier. Do not call them own-AR continuations or historical packed candidate replay.',
        'First/restored token differences, graph fallback and cross-route differences are diagnostic findings, not hidden corruption or new production qualification. Exact-tie argmax cannot be reconstructed from only top2 values.',
        'Source/cleanup/lock/limits bind only this diagnostic. No elapsed time is a full-workload Goodput measurement or an external client clock.']
    result['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(**desc(args.output), status=result['status'], errors=result['errors'])))
    return int(bool(result['errors']))

if __name__ == '__main__': raise SystemExit(main())
