"""Source construction only; never imports a candidate or reads current197 actual."""
import ast,copy,hashlib,json
from pathlib import Path
D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
S=D/'audit_v189_terminal_HOST.py';text=S.read_text();assert hashlib.sha256(S.read_bytes()).hexdigest()=='3935ae12f3b7983bed7fc0349ced8befd2c59f31fe103c2e263de174f1631920'
tree=ast.parse(text)
def source(name):
 n=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name);return '\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])+'\n'
header=text[:text.index('\ndef provenance(')]
for a,b in [('Independent future189','Independent future197'),('completed173 auditor','completed173/189 auditors'),("tree-slo-admission1-causal80-active32-full-v189","tree-empty-active-cohort-causal80-active32-full-v197"),("slo-admission1-causal80-active32-full463","empty-active-cohort-causal80-active32-full463"),('ac0233ffafcc99b8303c8b88c4911cf4c2d0cc7d4129ff1b6ac6ba879c9777eb','9368614839d11e4100b812cf173d2c55fb5d249a729a422d2958be9bc8109324'),('ROOT_V189_PREPARED_SOURCE_HOST_AUDIT.json','ROOT_V197_PREPARED_SOURCE_HOST_AUDIT.json'),('f9b9032cfad346a7e45c1ee0db6f84386060c1da1cc86182f1154d18fe7b91c1','43b0a32d0b9bdc0df23de184ac43ad6cb659006542a4035fc447387253a3c2b1'),('isolated189-stdlib-helpers','isolated197-stdlib-helpers')]:
 assert header.count(a)==1,a;header=header.replace(a,b)
header+='\nSTAT_SOURCE = D / \'audit_v189_terminal_HOST.py\'\nSTAT_SHA = \'3935ae12f3b7983bed7fc0349ced8befd2c59f31fe103c2e263de174f1631920\'\nARCHIVE_SHA = \'a5e7eae79f81980bfdca4084e748fc3cc16a2dc5739b7020417a5b729b2fcb5c\'\n\n'
provenance=r'''def provenance(receipt):
    freeze_path=P/'FROZEN_SOURCE197.json'
    need(sha(freeze_path)==FREEZE_SHA and sha(ROOT_PREPARED)==ROOT_PREPARED_SHA,
         'exact frozen197 and ROOT prepared proof')
    need(sha(STAT_SOURCE)==STAT_SHA and sha(CLOCKS)==CLOCK_SHA,'unchanged stdlib statistics/clock sources')
    f=read(freeze_path);pins={}
    for key in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
        for path,value in f[key].items():
            need(path not in pins or pins[path]==value,'nonconflicting197 selected pin');pins[path]=value
    pins.update({str(P/'CPU_RESULT197.json'):f['CPU_sha256'],str(freeze_path):FREEZE_SHA,
                 f['root_review']['path']:f['root_review']['sha256']})
    need(len(pins)==294 and all(sha(p)==v for p,v in pins.items()),'all294 current source/evidence hashes')
    c=read(P/'CPU_RESULT197.json');prepared=read(P/'PREPARED_HOST197.json')
    need(c['schema']=='V197.full-source-HOST.v1' and c['status']=='passed' and c['tests_run']==4
         and c['failures']==c['errors']==0 and all(c[k] is False for k in
         ('Torch_imported','NPU_used','model_constructed','model_math_qualified','old_suites_rerun','full463_executed','freeze_created')),
         'actual197 CPU producer4 checks, no invented189 policy/backend fields')
    need(c['local_before']==c['local_after']==f['files']==prepared['files'] and
         c['external_before']==c['external_after']==f['external_files']==prepared['external_files'] and
         f['CPU_sha256']==prepared['CPU_sha256']==sha(P/'CPU_RESULT197.json'),'CPU/source/prepared/freeze identity')
    need(set(prepared['actual_HOST_attempts'])=={str(p) for p in P.glob('HOST_ATTEMPT197_*.json')}
         and len(prepared['actual_HOST_attempts'])==3 and len(prepared['retained_HOST_files'])==82,
         'all3 actualHOST attempts plus82 source/log/result snapshots')
    ev={str(P/'PREPARED_HOST197.json'):sha(P/'PREPARED_HOST197.json'),**prepared['actual_HOST_attempts'],**prepared['retained_HOST_files']}
    need(ev==f['retained_HOST_evidence'] and len(ev)==86,'all86 retained HOST evidence distinct from82 files')
    expected=dict(empty_active_only=True,current_multiowner_eager_math=True,original_parent173_cap4=True,
                  causal80_domain=[16,80,'causal'],active_windows_requested=[32],configured_capacity=64,queued=463,storage=527)
    need(f['ROOT_source_reviewed'] is True and all(f[k]==v and receipt[k]==v for k,v in expected.items()),
         'actual current empty-active173 production scope')
    root=read(f['root_review']['path']);reviewed={k:v for k,v in pins.items() if k not in(str(freeze_path),f['root_review']['path'])}
    need(root['status']=='passed_source_review' and root['reviewed_inputs']==reviewed,'exact ROOT292 reviewed inputs')
    readiness=receipt['readiness']
    need(readiness['freeze_sha256']==FREEZE_SHA and readiness['CPU_sha256']==f['CPU_sha256'] and
         readiness['root_review']==f['root_review'] and readiness['peers']==f['peer_reviews'] and
         readiness['device_qualification_inherited'] is False,'actual original197 readiness')
    ar=readiness['archive']
    need(Path(ar['path'])==P/'PREPARED_SOURCE_ARCHIVE197.tar.gz' and ar['members']==294 and
         sha(ar['path'])==ar['sha256']==ARCHIVE_SHA,'actual294 own regular archive/binary')
    own_archive=archive(ar['path'],pins)
    import base197,check197
    deps=dict(original=base197.parent_archive(),P173=check197.require_parent(archive=True))
    need(deps==f['dependency_archives_verified_in_place']==readiness['dependency_archives'],
         'P1486390/P16042/P173158 exact in-place source dependencies')
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
    return dict(freeze=desc(freeze_path),CPU=desc(P/'CPU_RESULT197.json'),ROOT_prepared=desc(ROOT_PREPARED),
                selected_current_SHA_checked=294,archive=own_archive,parent_archives=deps,storage177_source=desc(safe),
                saved_prepost='10496/15/212 PASS; not rerun',saved_all4_idle=True,owned_remaining=[],
                inherited_cost_measured_flag_is_not_numeric_or_performance_evidence=True,
                inherited_new_target_keys2_is_original173_M80_not_extra_prefill_keys=True,
                complete_environment_import_closure_claimed=False)

'''
fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='admission_and_full_statistics'))
start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='t'for t in n.targets))
assert start==2
fn.name='common_full_statistics';fn.body=fn.body[start:]
ret=fn.body[-1];assert isinstance(ret,ast.Return)
k=next(v for v in ret.value.keywords if v.arg=='actual_SLO_admission1_target0_coalesce1_0');k.arg='actual_empty_active_only_coalesce1_0'
common=ast.unparse(ast.fix_missing_locations(fn))+'\n\n'
extra=r'''def role_counter_fields():
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
                     and row['all_query_KV_tensors']==(56 if rank==0 else128) and row['candidate_KV_copied'] is False and
                     row['singleton201_authorization_used'] is False and row['full32_numerical_exhaustive'] is False,
                     'actual sameM fullphysical hidden/roleKV gates bound; tensors unavailable to HOST')
                pub=row['publication'];root=pub['root_tokens'];need(len(root)==n and all(type(t)is int and t>=0 for t in root),'actual n root tokens')
                need(pub['original_append_and_publish_AST'] is pub['original_callback'] is True and
                     pub['active']==list(range(n)) and pub['logical_tokens']==[p+[t]for p,t in zip(tok[:n],root,strict=True)] and
                     len(pub['callbacks'])==(n if rank==1 else0),'original once root append/callback')
                if rank==1:
                    need([v['request_index']for v in pub['callbacks']]==list(range(n)) and
                         all(v['token_ids']==[root[v['request_index']]] for v in pub['callbacks']),'immediate ordered single-root callbacks')
                if rank:need(row['independent_same_N_heads']==root,'same n original target lasttoken head exact')
                key=(phase,n)
                if key in roots:need(roots[key]==root,'all4 actual original root broadcast agreement')
                else:roots[key]=root
                nxt=row['next_draft'];plan=nxt['planner'];pn=1 if n==2 else16;ids=list(range(0,n,2));refs=[i+n for i in ids]
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
    from empty197 import audit_trace
    for rank,trace in enumerate(traces):
        need(trace['sealed_before'] is trace['sealed_after'] is trace['cache_released'] is True,'all4 sealed/released full')
        need(trace['configured_capacity']==64 and trace['admission_window160']==32 and 0<trace['active_peak']<=32 and
             trace['actual_capacity64_scope_completed'] is False,'truthful actual32/config64')
        config=reports[rank]['configuration_before_constructor']
        need(config==reports[rank]['configuration_after_constructor'] and config['spec_rhythm_prefill_coalesce_min_requests']==1 and
             config['spec_rhythm_prefill_coalesce_max_wait_ms']==0 and config['spec_rhythm_verification_budget']==96,
             'actual original173 constructor/coalesce1/0/budget')
        c=trace['decode_counters']
        need(c['spec_rhythm_slo_aware_admission_enabled']==c['spec_rhythm_target_slo_first_enabled']==c['spec_rhythm_prefill_coalesce_enabled']==0,
             'actual SLO0/TARGET0/original coalesce1')
    for t in traces[1:]:need(t['standalone_prefill_calls']==traces[0]['standalone_prefill_calls'],'all4 prefill cohorts identical')
    cohorts=audit_trace(traces[1]);value=common_full_statistics(traces,reports)
    calls=traces[1]['standalone_prefill_calls'];c=traces[1]['decode_counters'];sizes=collections.Counter(len(v['sequence_ids'])for v in calls)
    need(cohorts['cohort_count']==len(calls) and cohorts['all_arrivals_retained'] is cohorts['no_active_refill'] is True,
         'actual closed cohort stream replay all463 drain')
    need(c['spec_rhythm_prefill_coalesce_deferred_polls']==0,'no vacancy/coalesce waiting policy added')
    value.update(empty_active_cohort_replay=cohorts,prefill_cohort_size_histogram=dict(sorted(sizes.items())),
                 deferred_polls=c['spec_rhythm_prefill_coalesce_deferred_polls'],no_prefill_pause_memberships_for_incumbents=True,
                 not_an_unchanged_trajectory_or_performance_counterfactual=True)
    return value

'''
main=r'''def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-authorized',action='store_true',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)is not None,'safe exact tag')
    need(a.output.parent.resolve()==D and not a.output.exists(),'fresh D output')
    receipt_path=P/('receipt-'+a.tag+'.json')
    # No current actual or receipt is read until ROOT explicitly supplies exact terminal identity.
    need(sha(receipt_path)==a.expected_receipt_sha256,'ROOT exact terminal197 receipt SHA')
    receipt=read(receipt_path)
    need(receipt['status']in('complete','failed') and receipt['action']==ACTION,'terminal197 only; running rejected before actual read')
    actual=P/'actual'/a.tag;need(actual.is_dir()and actual.resolve()==actual and not actual.is_symlink(),'private terminal actual')
    sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P));errors=[]
    result=dict(schema='V197.independent-terminal-HOST-audit.v1',candidate=str(P),tag=a.tag,job_status=receipt['status'],
                receipt=desc(receipt_path),script=desc(__file__),errors=errors,Torch_imported=False,NPU_used=False,
                whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
    try:
        result['provenance']=provenance(receipt)
        need(read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'all4 actual workers closed')
    except Exception as e:errors.append(dict(stage='source_or_terminal_provenance',error=repr(e)))
    paths=sorted(actual.rglob('*'));need(not any(p.is_symlink()for p in paths),'no raw aliases')
    manifest={str(p):desc(p)for p in paths if p.is_file()};mp=a.output.with_name(a.output.stem+'_RAW_MANIFEST.json')
    need(not mp.exists(),'fresh manifest');mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    result['raw_manifest']=dict(**desc(mp),files=len(manifest));result['worker_errors']=[read(p)for p in sorted(actual.glob('rank*-last-error.json'))]
    if receipt['status']=='complete' and 'provenance'in result:
        try:
            import audit197
            original=audit197.audit(actual,(32,));complete=read(actual/'completed.json')
            need(original['passed']is True and original==complete['audit']==receipt['actual_full_audit'] and
                 complete['action']==ACTION and complete['replies']==[['calibrated',r,dict(windows=[32],full463=True)]for r in range(4)],
                 'unmodified audit197 ==completed==original guard')
            need(receipt['errors']==[] and receipt['child']['exit_code']==0 and receipt['actual_execution_complete']is True and
                 not result['worker_errors'] and receipt['actual_full_status']=='passed_full','actual original successful job')
            need(original['native_zero_subtractions']==252 and original['math_independent_qualified']is True and
                 original['current_multiowner_eager_component']['Native_zero_scalar_subtractions']==336,
                 'base/full252 and extra336 natural zero gates')
            result['original_unmodified_audit']=dict(source=desc(P/'audit197.py'),result=original)
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
             full_qualification_issued=False,GP_assigned=False,within7200_text_alone_is_not_timeout_proof=True)
    need(not Firewall.attempts and not any(n.split('.')[0]in BACKENDS for n in sys.modules),'no backend imports')
    result['status']=('passed_terminal_component_and_full463_audit' if receipt['status']=='complete' else
                      'terminal_failed_job_evidence_only') if not errors else'terminal_evidence_with_audit_errors'
    result['limitations']=[
        'HOST reconstructs source,integer verdict,metadata/slot/table,counter,root/packet and clock arithmetic. Hidden/logit/KV tensors/finite predicates are unavailable; actual sameM/allphysical/128or56KV/operator gates are source-and-raw-bound, never claimed CPU recomputation.',
        'N32 is only current multiowner prefill; original controller chooses home16 next (N2 gives1),no extra target submit or newN32target issuer. All original same-engine173B4/M80 actual target verdict/commit/ownnext remain prerequisite.',
        'Native charged TPOT/256 and worker-publication first-last/255 are separate attainment clocks. Existing GP functions use original generate wall denominator; no final9 manifest-origin scoring or external client delivery clock is fabricated.',
        'Empty-active admission may raise queue/TTFT or reduce utilization. Cohort/pause arithmetic is actual trajectory only,not a no-pause speedup estimate or rewritten arrivals.',
        'Inherited cost_measured=True/new_target_keys=2 template flags alone prove neither performance nor added prefkeys. Whole463 independent AR,adoption/formalGoal remainFalse.',
        'Failed jobs stay failed and receive no GP0 or full qualification. Generic within7200 message does not by itself identify timeout.'
    ]
    result['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with a.output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(**desc(a.output),status=result['status'],job_status=result['job_status'],errors=errors)))
    return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
'''
# Python lexical spacing in three conditional expressions is explicit; no imported backend.
extra=extra.replace('else128','else 128').replace('else0','else 0').replace('else16','else 16')
output=header+provenance+source('component_integer_verdicts')+'\n'+common+extra+main
p=D/'audit_v197_terminal_HOST.py';assert not p.exists();p.write_text(output)
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}))
