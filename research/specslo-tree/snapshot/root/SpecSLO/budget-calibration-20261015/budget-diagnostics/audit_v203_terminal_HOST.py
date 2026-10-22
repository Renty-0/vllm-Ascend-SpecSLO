"""Independent future203 terminal evidence audit. Never launch model/device work.

ROOT must authorize a terminal receipt by its exact SHA before any actual read.
Only standard-library helpers from the completed173/189 auditors are AST-reused.
"""
import argparse, ast, collections, copy, datetime, gzip, hashlib, importlib.abc
import json, math, re, statistics, sys, tarfile, types
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-empty-active-slo-admission1-causal80-active32-full-v203'
ACTION = 'empty-active-slo-admission1-causal80-active32-full463'
FREEZE_SHA = '7ff2a3d5c3d6b7bca985bf843bb46b29b71d1cb05428ccf2b17ba86741893d75'
ROOT_PREPARED = D / 'ROOT_V203_PREPARED_SOURCE_HOST_AUDIT.json'
ROOT_PREPARED_SHA = '652a2a188e5f2c664f762983425e717d31c02df43dd9336980ab7939171a3bad'
HELPERS = D / 'audit_v173_terminal_full463_HOST.py'
HELPER_SHA = 'ed8aef51646ccf60c4812782d080d119fa406d7e53138ce24d6b470bce74c6f5'
CLOCKS = D / 'analyze_completed_tree_full463_cost.py'
CLOCK_SHA = 'd3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
BACKENDS = ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'safetensors', 'acl')

if hashlib.sha256(HELPERS.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Exact completed173 stdlib helper source changed')
_names = {'need', 'sha', 'desc', 'read', 'archive', 'Firewall', 'extracted', 'Value', 'pct'}
_nodes = [copy.deepcopy(n) for n in ast.parse(HELPERS.read_text()).body
          if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _names]
if {n.name for n in _nodes} != _names:
    raise RuntimeError('Complete selected173 stdlib helper AST')
exec(compile(ast.fix_missing_locations(ast.Module(body=_nodes, type_ignores=[])),
             str(HELPERS) + '::isolated203-stdlib-helpers', 'exec'), globals())


DERIVED_FROM = D / 'audit_v197_terminal_HOST.py'
DERIVED_FROM_SHA = '5118f3e4f2da7ad3de169dfce9e48f310df8097c41c7e0b350df8852eedc81b7'
STAT_SOURCE = D / 'audit_v189_terminal_HOST.py'
STAT_SHA = '3935ae12f3b7983bed7fc0349ced8befd2c59f31fe103c2e263de174f1631920'
ARCHIVE_SHA = '1f39cfc60c6c0d64c464c33fa146fb9ac1d91b080c6d07c9fec4aaa59002ce0a'

def provenance(receipt):
    freeze_path=P/'FROZEN_SOURCE203.json'
    need(sha(freeze_path)==FREEZE_SHA and sha(ROOT_PREPARED)==ROOT_PREPARED_SHA,
         'exact frozen203 and ROOT prepared proof')
    need(sha(DERIVED_FROM)==DERIVED_FROM_SHA,'unchanged completed197 independent auditor source')
    need(sha(STAT_SOURCE)==STAT_SHA and sha(CLOCKS)==CLOCK_SHA,'unchanged stdlib statistics/clock sources')
    f=read(freeze_path);pins={}
    for key in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
        for path,value in f[key].items():
            need(path not in pins or pins[path]==value,'nonconflicting203 selected pin');pins[path]=value
    pins.update({str(P/'CPU_RESULT203.json'):f['CPU_sha256'],str(freeze_path):FREEZE_SHA,
                 f['root_review']['path']:f['root_review']['sha256']})
    need(len(pins)==332 and all(sha(p)==v for p,v in pins.items()),'all332 current source/evidence hashes')
    c=read(P/'CPU_RESULT203.json');prepared=read(P/'PREPARED_HOST203.json')
    need(c['schema']=='V203.actual-empty-active-admission-HOST.v1' and c['status']=='passed' and c['tests_run']==3
         and c['failures']==c['errors']==0 and all(c[k] is False for k in
         ('Torch_imported','NPU_used','model_constructed','parent_suites_rerun','device_qualification_inherited','freeze_created')),
         'actual203 three-HOST producer schema and false backend/model/inheritance fields')
    need(c['local_before']==c['local_after']==f['files']==prepared['files'] and
         c['external_before']==c['external_after']==f['external_files']==prepared['external_files'] and
         f['CPU_sha256']==prepared['CPU_sha256']==sha(P/'CPU_RESULT203.json'),'CPU/source/prepared/freeze identity')
    need(set(prepared['actual_HOST_attempts'])=={str(p) for p in P.glob('HOST_ATTEMPT203_*.json')}
         and len(prepared['actual_HOST_attempts'])==3 and len(prepared['retained_HOST_files'])==84,
         'all3 actualHOST attempts plus84 source/log/result snapshots')
    ev={str(P/'PREPARED_HOST203.json'):sha(P/'PREPARED_HOST203.json'),**prepared['actual_HOST_attempts'],**prepared['retained_HOST_files']}
    need(ev==f['retained_HOST_evidence'] and len(ev)==88,'all88 retained HOST evidence distinct from84 files')
    expected=dict(empty_active_only=True,current_multiowner_eager_math=True,original_parent173_cap4=True,
                  causal80_domain=[16,80,'causal'],active_windows_requested=[32],configured_capacity=64,queued=463,storage=527,
                  SLO_AWARE_ADMISSION=True,TARGET_SLO_FIRST=False)
    need(f['ROOT_source_reviewed'] is True and all(f[k]==v and receipt[k]==v for k,v in expected.items()),
         'actual current empty-active173 production scope plus only original SLO1/TARGET0')
    root=read(f['root_review']['path']);reviewed={k:v for k,v in pins.items() if k not in(str(freeze_path),f['root_review']['path'])}
    need(root['status']=='passed_source_review' and root['reviewed_inputs']==reviewed and len(reviewed)==330,
         'exact ROOT330 reviewed inputs')
    readiness=receipt['readiness']
    need(readiness['freeze_sha256']==FREEZE_SHA and readiness['CPU_sha256']==f['CPU_sha256'] and
         readiness['root_review']==f['root_review'] and readiness['peers']==f['peer_reviews'] and
         readiness['device_qualification_inherited'] is False,'actual original203 readiness')
    ar=readiness['archive']
    need(Path(ar['path'])==P/'PREPARED_SOURCE_ARCHIVE203.tar.gz' and ar['members']==332 and
         sha(ar['path'])==ar['sha256']==ARCHIVE_SHA,'actual332 own regular archive/binary')
    own_archive=archive(ar['path'],pins)
    import check203
    deps=check203.require_parent(archive=True)
    need(deps==f['dependency_archives_verified_in_place']==readiness['dependency_archives'] and
         deps['parent']==str(B/'tree-empty-active-cohort-causal80-active32-full-v197') and deps['members']==294 and
         deps['archive_verified_in_place'] is True and deps['device_qualification_inherited'] is False,
         'sole197 parent294 and its P1486390/P16042/P173158 in-place dependency producer schema')
    safe=D/'v177-root-safe-storage-overlay/safe_storage177.py'
    need(f['external_files'][str(safe)]==sha(safe),'frozen177 direct scoped storage source')
    for key in ('pre_verifier','post_verifier'):
        row=receipt[key];value=read(row['path'])
        need(sha(row['path'])==row['sha256'] and row['exit_code']==0 and value==row['actual_result'] and
             value['status']=='pass' and value['errors']==[] and
             (value['checked_frozen_files'],value['checked_external_files'],value['checked_packages'])==(10496,15,212),
             'saved original baseline pre/post verifier '+key)
    for key in ('devices_before','devices_after'):
        value=receipt[key]
        need(value['all_idle'] is value['four_910B2'] is True and set(value['processes'])=={'2','5','6','7'},'saved4 idle '+key)
        need(all(v['exit_code']==0 and v['idle'] is True and not v.get('timed_out',False) and
                 re.search(r'No process in device\.|No running processes found',v['output'],re.I)
                 for v in value['processes'].values()),'saved original device outputs '+key)
    need(receipt['physical_cards']==[2,5,6,7] and receipt['sources_unchanged'] is True and receipt['owned_remaining']==[]
         and receipt['execution_lock']['path']==str(B/'.ROOT_NPU_EXECUTION.lock') and
         receipt['execution_lock']['nonblocking_exclusive'] is True,'original selfROOTlock/cleanup/source continuity')
    limits=dict(child_execution_seconds=7200,actual_tag_bytes=256*1024**2,aggregate_private_bytes=512*1024**2)
    need(receipt['limits']==limits and math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds']>0,
         'actual original finite elapsed and limits')
    if receipt['status']=='complete':need(receipt['elapsed_seconds']<7200,'completed bounded execution')
    storage=receipt['private_storage_final']
    need(0<=storage['actual_tag_bytes']<=limits['actual_tag_bytes'] and
         0<=storage['aggregate_private_bytes']<=limits['aggregate_private_bytes'],'saved private final storage')
    return dict(freeze=desc(freeze_path),CPU=desc(P/'CPU_RESULT203.json'),ROOT_prepared=desc(ROOT_PREPARED),
                selected_current_SHA_checked=332,archive=own_archive,parent_archives=deps,storage177_source=desc(safe),
                saved_prepost='10496/15/212 PASS; not rerun',saved_all4_idle=True,owned_remaining=[],
                inherited_cost_measured_flag_is_not_numeric_or_performance_evidence=True,
                inherited_new_target_keys2_is_original173_M80_not_extra_prefill_or_policy_keys=True,
                complete_environment_import_closure_claimed=False)


def component_integer_verdicts(directory):
    """Run unchanged Native integer verifier against saved heads, not model math."""
    ht = types.SimpleNamespace(Tensor=Value, tensor=lambda v, **kw: Value(v), long='long')
    ns = dict(torch=ht, TreeVerificationOutput=lambda a, b: types.SimpleNamespace(token_ids=a, accepted_node_indices=b))
    verify = extracted(NATIVE / 'native_engine.py', '_verify_tree_outputs_host', ns)
    counts = collections.Counter(); cross = []
    for folder_name, passes_per_rank in (('variable-B4-own-full-process', 6), ('causal80-own-current-process', 3)):
        folder = directory / folder_name
        for rank in range(4):
            report = read(folder / f'rank{rank}-component.json.gz')
            need(report['rank'] == rank and len(report['traces']) == passes_per_rank, 'actual role/pass inventory')
            for trace in report['traces']:
                value = trace['Native']; before, after, delta = (value[k] for k in ('before', 'after', 'delta'))
                need(len(delta) == 21 and set(before) == set(after) == set(delta) and
                     all(type(before[k]) is type(after[k]) is type(delta[k]) is int
                         and 0 <= before[k] <= after[k] and after[k] - before[k] == delta[k] for k in delta),
                     'independent actual21 monotone subtraction')
                if trace['zero_replay']:
                    need(not any(delta.values()), 'entire protocol no-new-work zero')
                    counts[folder_name + '_zero_subtractions'] += 21
            if rank == 0:
                continue
            for phase in range(3):
                cases = range(2) if folder_name.startswith('variable') else (None,)
                for case in cases:
                    for cycle in range(4):
                        suffix = f'case{case}-' if case is not None else ''
                        row = read(folder / f'rank{rank}-pass{phase}-{suffix}cycle{cycle}.json.gz')
                        budgets = [len(v) for v in row['rows']]
                        plans = [types.SimpleNamespace(candidate_budget=v, parent_indices=Value([-1, 0, 1, 2][:v])) for v in budgets]
                        arms = ('old', 'new') if case is None else ('reference',)
                        for arm in arms:
                            head = row['crossM'][arm + '_real80'] if case is None else row['reference']['real_query_tokens']
                            out = verify(row['rows'], plans, Value(head), 4, 'explicit-HOST-values')
                            tokens = [[v for v in r if v >= 0] for r in out.token_ids.values]
                            expected_tokens = row['crossM'][arm + '_verdict_tokens'] if case is None else row['commit']['committed_tokens']
                            expected_paths = row['crossM'][arm + '_accepted'] if case is None else row['commit']['accepted_paths']
                            need(tokens == [[v for v in r if v >= 0] for r in expected_tokens]
                                 and out.accepted_node_indices.values == expected_paths,
                                 'unchanged Native verifier on actual saved ' + arm + ' heads')
                            counts[folder_name + '_integer_verifier_calls'] += 1
                            counts[folder_name + '_owner_verdicts'] += len(budgets)
                        if case is None:
                            cm = row['crossM']; different = [i for i, (a, b) in enumerate(zip(cm['old_real80'], cm['new_real80'], strict=True)) if a != b]
                            need(len(cm['old_real80']) == len(cm['new_real80']) == 80 and different == cm['different_head_positions']
                                 and cm['head_equal'] == (not different), 'retained cross-M complete80 positions')
                            cross.append(dict(rank=rank, phase=phase, cycle=cycle, different_head_positions=different,
                                              verdict_equal=cm['verdict_equal'], accepted_equal=cm['accepted_equal']))
    need(counts['variable-B4-own-full-process_zero_subtractions'] == 336 and
         counts['causal80-own-current-process_zero_subtractions'] == 168 and
         counts['variable-B4-own-full-process_integer_verifier_calls'] == 72 and
         counts['causal80-own-current-process_integer_verifier_calls'] == 72 and len(cross) == 36,
         'all504 component zeros plus144 integer verifier calls')
    need([v for v in cross if v['rank'] == 1] == [dict(v, rank=1) for v in cross if v['rank'] == 2] ==
         [dict(v, rank=1) for v in cross if v['rank'] == 3], 'three target ranks cross-M outcomes agree')
    return dict(counts=counts, current80_crossM_records=cross,
                all80_heads_equal=all(not v['different_head_positions'] for v in cross),
                all_verdicts_equal=all(v['verdict_equal'] and v['accepted_equal'] for v in cross),
                hidden_head_KV_tensors_recomputed_on_HOST=False)

def common_full_statistics(traces, reports):
    t = traces[1]
    ns = dict(math=math, statistics=statistics, CLASSES=('tight', 'normal', 'loose'))
    extracted(CLOCKS, 'need', ns)
    metrics = extracted(CLOCKS, 'clocks', ns)(t)
    first = {}
    for e in t['stream_events']:
        owner = e['request_index']
        if owner not in first:
            need('first_token_wall_seconds160' in e, 'actual first worker-submission timestamp')
            first[owner] = e['first_token_wall_seconds160']
        else:
            need('first_token_wall_seconds160' not in e, 'one first timestamp per original request')
    need(set(first) == set(range(463)) and all((row['arrival_ts'] == t['absolute_arrivals'][i] for i, row in enumerate(t['generate_results']))), 'all463 original arrival clocks and first callbacks')
    ttft = [(first[i] - row['arrival_ts']) * 1000 for i, row in enumerate(t['generate_results'])]
    need(all((math.isfinite(v) and v >= 0 for v in ttft)), 'finite nonnegative worker TTFT')
    metrics['TTFT_worker_submission'] = dict(samples=463, **pct(ttft), by_class={cl: pct([ttft[i] for i, row in enumerate(t['generate_results']) if row['slo_class'] == cl]) for cl in ('tight', 'normal', 'loose')})
    retained = next((v['output'] for v in reports[1]['windows']['per_rank_full_metrics'] if v['window'] == 32))
    need(all((retained[k] == v for k, v in metrics.items() if k != 'TTFT_worker_submission')) and all((retained['TTFT160'][k] == metrics['TTFT_worker_submission'][k] for k in ('median_ms', 'P90_ms'))), 'independent original clocks/TTFT match actual runtime metrics')
    calls = t['normal_target_calls']
    owners = sum((c['N'] for c in calls))
    nodes = sum((sum(c['budgets']) for c in calls))
    real = sum((c['real_query_count'] for c in calls))
    physical = sum((c['physical_M'] for c in calls))
    dc = t['decode_counters']
    need(nodes == dc['spec_rhythm_verified_tokens'] and physical == dc['spec_rhythm_target_query_tokens'] and (physical - real == dc['spec_rhythm_target_graph_padding_queries']), 'real target geometry counters')
    selected80 = [c for c in calls if (c['N'], c['real_query_count'], c['family']) == (16, 80, 'causal')]
    need(all(((c['physical_M'], c['physical_S'], c['physical_query_partition']) == (80, 16, [v + 1 for v in c['budgets']]) for c in selected80)), 'actual current80 mapper usage')
    prefill_ids = [owner for c in t['standalone_prefill_calls'] for owner in c['sequence_ids']]
    need(sorted(prefill_ids) == list(range(463)) and sum((c['uncached_queries'] for c in t['standalone_prefill_calls'])) == sum((len(v) for v in t['tokenized_inputs'])) and (t['logical_events'] == []), 'all463 genuine standalone whole prompts, no drop or mixed claim')
    queues = collections.defaultdict(collections.deque)
    for c in calls:
        for owner, budget in zip(c['sequence_ids'][:c['N']], c['budgets'], strict=True):
            queues[owner].append(budget)
    initial = set()
    delivered = visible_bonus = 0
    align_error = None
    for e in t['stream_events']:
        owner = e['request_index']
        if owner not in initial:
            need(len(e['token_ids']) == 1, 'one original prefill/root token')
            initial.add(owner)
            continue
        delivered += len(e['token_ids'])
        if queues[owner]:
            visible_bonus += int(len(e['token_ids']) == min(queues[owner].popleft(), 4) + 1)
        else:
            align_error = 'A post-initial callback has no retained target owner occurrence.'
    if any(queues.values()):
        align_error = 'Some retained target owner occurrences have no callback.'
    bonus = dict(owner_rounds=owners, accepted_nodes=dc['spec_rhythm_tree_accepted_nodes'], Native_full_primary_accepts=dc['spec_rhythm_tree_full_accepts'], delivered_decode_tokens=delivered, verified_event_alignment_exact=align_error is None, alignment_limitation=align_error)
    if align_error is None:
        accepted = bonus['accepted_nodes']
        full = bonus['Native_full_primary_accepts']
        discarded = accepted + owners - delivered
        need(discarded >= full - visible_bonus >= 0, 'primary bonus/truncation accounting')
        bonus.update(accepted_nodes_per_owner=accepted / owners, delivered_decode_per_owner=delivered / owners, visible_primary_bonus=visible_bonus, discarded_primary_bonus=full - visible_bonus, total_output_limit_discarded=discarded, subtraction_is_not_a_rerun_without_bonus=True)
    pages = {}
    for trace in traces:
        a = trace['allocator']
        need(a['pool_blocks'] == 544 and a['allocation_storage_rows'] == [527] and (0 <= a['referenced_page_highwater'] <= 544) and (a['final_referenced_pages'] == 0) and (a['acquired_pages'] == a['released_pages']), 'actual page highwater/release, not HBM bytes')
        pages[str(trace['rank'])] = a
    return dict(metrics=metrics, target_calls=len(calls), owner_rounds=owners, mean_N=owners / len(calls), budget_owner_rounds=dict(sorted(collections.Counter((v for c in calls for v in c['budgets'])).items())), N_histogram=dict(sorted(collections.Counter((c['N'] for c in calls)).items())), current80_calls=len(selected80), physical_queries=physical, real_queries=real, padding_queries=physical - real, standalone_prefill_calls=len(t['standalone_prefill_calls']), all463_prefill_owner_ids_exact=True, actual_empty_active_only_coalesce1_0=True, bonus_accounting=bonus, allocator_by_rank=pages, HOST_phase_seconds_by_role={str(v['rank']): v['phase_seconds'] for v in traces}, clock_scope='Native charged decode and worker-submitted stream events, not external client clocks')

def role_counter_fields():
    path=B/'tree-high-normal-cap4-native-families-v148/scope_v146.py';ns={}
    nodes=[copy.deepcopy(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id in ('NATIVE_COUNTERS','FAILURES') for t in n.targets)]
    need(len(nodes)==2,'actual two original Native counter declarations')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure-counter-declarations','exec'),ns)
    return ns['NATIVE_COUNTERS'],ns['FAILURES']


def independent_multiowner(directory,traces,reports):
    folder=directory/'multiowner-eager-prefill-own-current-process';zeros=0;roots={};cases=[]
    names,failures=role_counter_fields()
    need(len(names)==21,'actual21 counter schema')
    tok=traces[1]['tokenized_inputs'];results=traces[1]['generate_results']
    digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    for rank in range(4):
        report=read(folder/f'rank{rank}-component.json.gz')
        actual=reports[rank]['component']['multiowner_prefill']
        need(all(report[k] is True for k in ('passed_current_multiowner_math','cache_released','actual_entry_objects_preserved'))
             and report['Native_flags_assigned'] is False and report['rank']==rank and report['errors']==[],
             'current complete multiowner report')
        need(actual==dict(report,audit=actual['audit']),'worker retains exact current extra component and raw audit')
        base=read(directory/'causal80-own-current-process'/f'rank{rank}-component.json.gz')
        need(report['actual_Native_inventory']==base['inventory']['after'],
             'no new keys after actual same-instance173 M80 qualification; object identity is separately actual source-gated')
        need([(c['pass_index'],c['N']) for c in report['cases']]==[(p,n)for p in range(3)for n in(2,32)],'exact6 role cases')
        for phase in range(3):
            for n in (2,32):
                row=read(folder/f'rank{rank}-N{n}-pass{phase}.json.gz')
                summary=next(c for c in report['cases'] if(c['pass_index'],c['N'])==(phase,n))
                need(row['rank']==rank and row['N']==n and row['pass_index']==phase,'actual role/N/pass raw')
                want=[dict(request_id=results[i]['request_id'],length=len(tok[i]),sha256=digest(tok[i])) for i in range(n)]
                need(row['prompt_requests']==want,'genuine original first2/32 prompts bound to actual full463 source')
                lens=[len(v)for v in tok[:n]];need(len(set(lens))>1,'mixed actual prompt lengths')
                lease=row['lease'];tables=lease['block_tables'];pages=[v for t in tables for v in t if v>=0]
                need(len(tables)==527 and lease['storage_rows']==527 and lease['pages']==pages and
                     len(pages)==len(set(pages))==lease['actual_unique_pages']<=544 and
                     lease['candidate_owners']==list(range(n)) and lease['reference_owners']==list(range(n,2*n)) and
                     lease['incumbent_owner']==2*n and lease['candidate_prefix_copied'] is False,'actual disjoint owner/page lease')
                for key,off in (('candidate_metadata',0),('independent_metadata',n)):
                    meta=row[key];ids=[i+off for i,l in enumerate(lens) for _ in range(l)]
                    positions=[j for l in lens for j in range(l)];ends=[];total=0
                    for l in lens:total+=l;ends.append(total)
                    rt=[tables[i+off]for i in range(n)]
                    slots=[rt[i][j//128]*128+j%128 for i,l in enumerate(lens)for j in range(l)]
                    need(meta['inputs']==[v for ts in tok[:n]for v in ts] and meta['sequence_ids']==ids and
                         meta['owners']==list(range(off,n+off)) and meta['positions']==positions and
                         meta['slots']==slots and meta['contexts']==[j+1 for j in positions] and
                         meta['actual_seq_lengths_q']==ends and meta['sequence_lens']==lens and
                         meta['block_tables']==meta['request_block_tables']==rt and
                         meta['mask_shape']==[2048,2048] and meta['mask_dtype']=='torch.int8' and
                         meta['cached_mask_identity'] is meta['fused'] is True and
                         meta['field_dtypes']==dict(context_lens='torch.int32',slot_mapping='torch.int32',block_tables='torch.int32'),
                         'independent actual source/owner/slot/table/Q/KV/FIA mask ABI arithmetic')
                need(all(row[k] is True for k in ('all_physical_hidden_original_matcher','all_query_KV_original_matcher','incumbent_prefix_protected'))
                     and row['all_query_KV_tensors']==(56 if rank==0 else 128) and row['candidate_KV_copied'] is False and
                     row['singleton201_authorization_used'] is False and row['full32_numerical_exhaustive'] is False,
                     'actual sameM fullphysical hidden/roleKV gates bound; tensors unavailable to HOST')
                pub=row['publication'];root=pub['root_tokens'];need(len(root)==n and all(type(t)is int and t>=0 for t in root),'actual n root tokens')
                need(pub['original_append_and_publish_AST'] is pub['original_callback'] is True and
                     pub['active']==list(range(n)) and pub['logical_tokens']==[p+[t]for p,t in zip(tok[:n],root,strict=True)] and
                     len(pub['callbacks'])==(n if rank==1 else 0),'original once root append/callback')
                if rank==1:
                    need([v['request_index']for v in pub['callbacks']]==list(range(n)) and
                         all(v['token_ids']==[root[v['request_index']]] for v in pub['callbacks']),'immediate ordered single-root callbacks')
                if rank:need(row['independent_same_N_heads']==root,'same n original target lasttoken head exact')
                key=(phase,n)
                if key in roots:need(roots[key]==root,'all4 actual original root broadcast agreement')
                else:roots[key]=root
                nxt=row['next_draft'];plan=nxt['planner'];pn=1 if n==2 else 16;ids=list(range(0,n,2));refs=[i+n for i in ids]
                need(plan['actual_next_owners']==plan['planner_normal']==plan['target_ready_admission']==ids and
                     plan['actual_reference_owners']==refs and plan['actual_next_B']==[4]*pn and
                     plan['active_owners']==n and plan['target_candidates']==4*pn<=96 and
                     plan['target_max16'] is plan['untouched_owner_KV_protected'] is True and
                     plan['N32_target_scope_added'] is plan['actual_target_submit_in_this_extra_component'] is False,
                     'original natural home1/16 and typed ready<=16/C96; no extra target submission')
                need(nxt['selected']==[list(range(4))]*pn and len(nxt['rows'])==pn and all(len(v)==4 for v in nxt['rows']),
                     'actual4 primary next rows')
                if rank==0:
                    d=nxt['draft_proof'];need(all(d[k] is True for k in ('all_physical_hidden_original_matcher','all28_query_KV_original_matcher','exact_full8_encoded','independent_history')) and
                         d['actual_draft_model_stages']==4 and d['raw_encoded_shape']==[pn,4,2] and d['raw_pool_nodes']==8*pn and
                         d['selected_C']==4*pn and d['variable_num_draft_tokens']==[4]*pn and d['independent_owners']==refs and
                         d['source_KV_copied'] is False and d['natural_missing_suffix_observed'] is False and d['natural_missing_suffix_rows']==0,
                         'real fourstep independent sameM full8/56KV/ledger next, not B3 relabel')
                    need(nxt['rows']==[v[:4]for v in d['raw_pool']] and d['roots']==[root[i]for i in ids],
                         'actual published root is current nextdraft input and primary4 selection')
                else:need(nxt['draft_proof'] is None,'only draft owns draft model proof')
                native=row['Native'];need(native==summary['Native'] and summary['root_tokens']==root and
                     set(native['before'])==set(native['after'])==set(native['delta'])==set(names) and
                     all(type(native[s][k])is int for s in ('before','after','delta')for k in names) and
                     all(native['after'][k]-native['before'][k]==native['delta'][k]>=0 for k in names),
                     'actual21 current monotone deltas and summaries')
                need(all(native['delta'][k]==0 for k in failures),'no failed/fallback discovery')
                if phase:need(not any(native['delta'].values()),'two entire new component zero sweeps');zeros+=21
                cases.append(dict(rank=rank,N=n,phase=phase,actual_next_N=pn,query_KV_tensors=row['all_query_KV_tensors'],actual_pages=len(pages)))
    need(zeros==336 and len(cases)==24,'all336 added zero scalars/all24 role cases')
    return dict(cases=cases,Native_extra_zero_subtractions=zeros,current_Native_inventory_no_new_keys=True,
                root_and_next_reference_source_reconstructed=True,N32_target_added=False,extra_component_submits_target=False,
                hidden_KV_tensors_recomputed_on_HOST=False,all_prompt_tuples_exhaustive=False)


def admission_and_full_statistics(traces,reports):
    from empty203 import audit_trace
    from admission203 import audit_trace as audit_policy
    for rank,trace in enumerate(traces):
        policy=audit_policy(trace,463)
        need(policy['SLO_AWARE_ADMISSION'] is True and policy['TARGET_SLO_FIRST'] is False,
             'original producer actual all4 full policy1/0')
        need(trace['sealed_before'] is trace['sealed_after'] is trace['cache_released'] is True,'all4 sealed/released full')
        need(trace['configured_capacity']==64 and trace['admission_window160']==32 and 0<trace['active_peak']<=32 and
             trace['actual_capacity64_scope_completed'] is False,'truthful actual32/config64')
        config=reports[rank]['configuration_before_constructor']
        need(config==reports[rank]['configuration_after_constructor'] and config['spec_rhythm_prefill_coalesce_min_requests']==1 and
             config['spec_rhythm_prefill_coalesce_max_wait_ms']==0 and config['spec_rhythm_verification_budget']==96,
             'actual original173 constructor/coalesce1/0/budget')
        actual=reports[rank]['admission_policy']
        need(actual['environment']=={'VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION':'1','VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST':'0'} and
             actual['SLO_AWARE_ADMISSION'] is True and actual['TARGET_SLO_FIRST'] is False and
             actual['arrival_filter_unchanged'] is True and actual['drop_or_abort_added'] is False,
             'actual current Native original environment resolution and unchanged arrival filter')
        c=trace['decode_counters']
        need(type(c['spec_rhythm_slo_aware_admission_enabled']) is int and c['spec_rhythm_slo_aware_admission_enabled']==1 and
             type(c['spec_rhythm_target_slo_first_enabled']) is int and c['spec_rhythm_target_slo_first_enabled']==0 and
             type(c['spec_rhythm_prefill_coalesce_enabled']) is int and c['spec_rhythm_prefill_coalesce_enabled']==0,
             'actual strict SLO1/TARGET0/coalesce1 trace counters')
    for t in traces[1:]:need(t['standalone_prefill_calls']==traces[0]['standalone_prefill_calls'],'all4 prefill cohorts identical')
    cohorts=audit_trace(traces[1]);value=common_full_statistics(traces,reports)
    t=traces[1];calls=t['standalone_prefill_calls'];c=t['decode_counters'];sizes=collections.Counter(len(v['sequence_ids'])for v in calls)
    need(cohorts['cohort_count']==len(calls) and cohorts['all_arrivals_retained'] is cohorts['no_active_refill'] is True,
         'actual closed cohort stream replay all463 drain')
    need(type(c['spec_rhythm_prefill_coalesce_deferred_polls']) is int and c['spec_rhythm_prefill_coalesce_deferred_polls']==0,
         'no vacancy/coalesce waiting policy added')
    counts=[0]*463;finished=set();first={};last={}
    for e in t['stream_events']:
        i=e['request_index'];need(type(i)is int and 0<=i<463 and i not in finished,'no callback after original finish')
        need(isinstance(e['token_ids'],list) and all(type(v)is int and v>=0 for v in e['token_ids']),'real original output tokens')
        if i not in first:first[i]=e['first_token_wall_seconds160']
        counts[i]+=len(e['token_ids']);last[i]=e
        if e['finished']:finished.add(i)
    need(counts==[256]*463 and finished==set(range(463)),'all463 original exactly256 lossless publication outputs/finished')
    results=t['generate_results'];classes=('tight','normal','loose');by_class={};mix=[]
    for cl in classes:
        ids=[i for i,row in enumerate(results)if row['slo_class']==cl]
        ttft=[(first[i]-results[i]['arrival_ts'])*1000 for i in ids]
        need(ids and all(math.isfinite(v)and v>=0 for v in ttft),'finite original-arrival TTFT by class')
        by_class[cl]=dict(requests=len(ids),TTFT_worker_publication=dict(**pct(ttft),min_ms=min(ttft),max_ms=max(ttft)),
                          original_request_ids=[results[i]['request_id']for i in ids])
    for row in cohorts['cohorts']:
        ids=row['owners'];composition=collections.Counter(results[i]['slo_class']for i in ids)
        mix.append(dict(ordinal=row['ordinal'],owners=ids,original_request_ids=[results[i]['request_id']for i in ids],
                        SLO_class_counts={cl:composition[cl]for cl in classes},incumbent_count=row['incumbent_count'],
                        original_arrivals=[results[i]['arrival_ts']for i in ids],
                        first_worker_publications=[first[i]for i in ids]))
    value.update(empty_active_cohort_replay=cohorts,prefill_cohort_size_histogram=dict(sorted(sizes.items())),
                 deferred_polls=c['spec_rhythm_prefill_coalesce_deferred_polls'],no_prefill_pause_memberships_for_incumbents=True,
                 actual_admission_policy=dict(SLO_AWARE_ADMISSION=True,TARGET_SLO_FIRST=False,coalesce_min_requests=1,coalesce_max_wait_ms=0.0),
                 cohort_composition=mix,TTFT_fairness_by_original_SLO_class=by_class,
                 all463_lossless_output_counts256=True,all463_finished=True,
                 admission_ready_set_or_poll_timestamps_reconstructed=False,
                 not_an_unchanged_trajectory_or_performance_counterfactual=True)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-authorized',action='store_true',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)is not None,'safe exact tag')
    need(re.fullmatch(r'[0-9a-f]{64}',a.expected_receipt_sha256)is not None,'explicit exact terminal receipt SHA256')
    need(a.output.parent.resolve()==D and not a.output.exists(),'fresh D output')
    receipt_path=P/('receipt-'+a.tag+'.json')
    # No current actual or receipt is read until ROOT explicitly supplies exact terminal identity.
    need(sha(receipt_path)==a.expected_receipt_sha256,'ROOT exact terminal203 receipt SHA')
    receipt=read(receipt_path)
    need(receipt['status']in('complete','failed') and receipt['action']==ACTION,'terminal203 only; running rejected before actual read')
    actual=P/'actual'/a.tag
    need(not actual.is_symlink() and actual.resolve()==actual,'private exact terminal actual path')
    actual_present=actual.is_dir()
    if receipt['status']=='complete':need(actual_present,'complete job must retain actual directory')
    sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P));errors=[]
    result=dict(schema='V203.independent-terminal-HOST-audit.v1',candidate=str(P),tag=a.tag,job_status=receipt['status'],
                receipt=desc(receipt_path),script=desc(__file__),errors=errors,Torch_imported=False,NPU_used=False,
                whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
    try:
        result['provenance']=provenance(receipt)
        if receipt['status']=='complete':need(read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'all4 actual workers closed')
    except Exception as e:errors.append(dict(stage='source_or_terminal_provenance',error=repr(e)))
    paths=sorted(actual.rglob('*')) if actual_present else []
    need(not any(p.is_symlink()for p in paths),'no raw aliases')
    manifest={str(p):desc(p)for p in paths if p.is_file()};mp=a.output.with_name(a.output.stem+'_RAW_MANIFEST.json')
    need(not mp.exists(),'fresh manifest');mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    result['raw_manifest']=dict(**desc(mp),files=len(manifest));result['worker_errors']=[read(p)for p in sorted(actual.glob('rank*-last-error.json'))]
    if receipt['status']=='complete' and 'provenance'in result:
        try:
            import audit203
            original=audit203.audit(actual,(32,));complete=read(actual/'completed.json')
            need(original['passed']is True and original==complete['audit']==receipt['actual_full_audit'] and
                 complete['action']==ACTION and complete['replies']==[['calibrated',r,dict(windows=[32],full463=True)]for r in range(4)],
                 'unmodified audit203 ==completed==original guard')
            need(receipt['errors']==[] and receipt['child']['exit_code']==0 and receipt['actual_execution_complete']is True and
                 not result['worker_errors'] and receipt['actual_full_status']=='passed_full','actual original successful job')
            need(original['native_zero_subtractions']==252 and original['math_independent_qualified']is True and
                 original['current_multiowner_eager_component']['Native_zero_scalar_subtractions']==336,
                 'base/full252 and extra336 natural zero gates')
            result['original_unmodified_audit']=dict(source=desc(P/'audit203.py'),result=original)
            result['actual_compatibility_classification']=dict(old_padding_behavior_equal=original['old_padding_behavior_equal'],
                 numerical_variant_only=original['numerical_variant_only'],guard_passed_full_is_not_crossM_or_whole_AR=True)
            result['additional_integer_component']=component_integer_verdicts(actual)
        except Exception as e:errors.append(dict(stage='original_current_components_full_audit_and_job',error=repr(e)))
        try:
            traces=[read(actual/'windows/window32/full463'/f'rank{r}-full463-actual-trace.json.gz')for r in range(4)]
            reports=[read(actual/f'rank{r}-local-diagnostic.json')for r in range(4)]
            result['independent_multiowner']=independent_multiowner(actual,traces,reports)
            result['retained_full463']=admission_and_full_statistics(traces,reports)
        except Exception as e:errors.append(dict(stage='independent_multiowner_or_full463_arithmetic',error=repr(e)))
    elif receipt['status']=='failed':
        result['failure_only']=dict(child=receipt['child'],outer_errors=receipt['errors'],completed_file_present=(actual/'completed.json').is_file(),
             actual_directory_present=actual_present,worker_exit=(read(actual/'worker-exit.json') if (actual/'worker-exit.json').is_file() else None),
             full_qualification_issued=False,GP_assigned=False,within7200_text_alone_is_not_timeout_proof=True)
    need(not Firewall.attempts and not any(n.split('.')[0]in BACKENDS for n in sys.modules),'no backend imports')
    result['status']=('passed_terminal_component_and_full463_audit' if receipt['status']=='complete' else
                      'terminal_failed_job_evidence_only') if not errors else'terminal_evidence_with_audit_errors'
    result['limitations']=[
        'HOST reconstructs source,integer verdict,metadata/slot/table,counter,root/packet and clock arithmetic. Hidden/logit/KV tensors/finite predicates are unavailable; actual sameM/allphysical/128or56KV/operator gates are source-and-raw-bound, never claimed CPU recomputation.',
        'N32 is only current multiowner prefill; original controller chooses home16 next (N2 gives1),no extra target submit or newN32target issuer. All original same-engine173B4/M80 actual target verdict/commit/ownnext remain prerequisite.',
        'Native charged TPOT/256 and worker/engine-publication first-last/255 are distinct attainment clocks. Existing GP functions retain original generate-wall denominators; no final9 manifest-origin denominator or external client-delivery observation is fabricated. Source name TTFT_worker_submission means retained worker publication callback, not main/client receive.',
        'Original SLO admission1 changes ready-request priority only, with target-SLO0/coalesce1-0. Empty-active groups still drain before another wholeprompt prefill; classes may remain mixed and lower-priority TTFT may increase. Report each class and exact cohort composition; no unchanged-trajectory GP/fairness counterfactual is inferred.',
        'Inherited cost_measured=True/new_target_keys=2 template flags alone prove neither performance nor added prefkeys. Whole463 independent AR,adoption/formalGoal remainFalse.',
        'Failed jobs stay failed and receive no GP0 or full qualification. Generic within7200 message does not by itself identify timeout.'
    ]
    result['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with a.output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(**desc(a.output),status=result['status'],job_status=result['job_status'],errors=errors)))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
