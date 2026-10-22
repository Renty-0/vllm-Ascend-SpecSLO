"""Future exact-authorized ROOT224 HOST audit. No actual reads before terminal CLI."""
import argparse,datetime,hashlib,importlib.util,json,re,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-FP32-tie-policy-full463-own-AR-v224'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
FREEZE='584e810f056c4f5c7539638b9e36e10fd74a48f3469ed5e0fb706b5196dacdec'
ARCHIVE='5f02932300349ee5e11b300e19b8e253b343d632b48071c0023f31b1906d3ec0'
OLD_AUDITOR=D/'audit_v209_terminal_HOST.py'
OLD_SHA='fe8a050d3dce287787017bdf9b0f319bd2577e641e40a7baef7f2df785c0db5b'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value


def cache_records(reports,need):
    values=[]
    for rank,r in enumerate(reports):
        need(r['rank']==rank and r['overlay']['ROOT224_overlay']=='cached-parent-SLO-admission1-cost-aware-N9to16' and r['overlay']['SLO_AWARE_ADMISSION']is True,
             'actual current211 overlay and rank')
        cache=r['ROOT224_cached_parent_dispatch']
        need(cache['initial_parent_source_checks']==1 and type(cache['cached_parent_returns'])is int
             and cache['cached_parent_returns']>1 and cache['canonical_mapper_or_base_module_mutated']is False,
             'once source-checked binding and actual repeated cache returns')
        need(cache['canonical_mapper_module']==str(DONOR/'mapper209.py')
             and cache['canonical_parent_module']==str(B/'tree-causal80-real16-bankfix-full-v173/mapper173.py'),
             'only exact canonical mapper/parent source objects cached')
        for name in('actual_helper_origins_before_constructor','actual_helper_origins_after_constructor'):
            need(bool(r[name]['ROOT224_explicit_overlay_sources']),'new source provenance not hidden under old module origin')
        values.append(cache)
    need(len(values)==4,'actual all4 cached mapper records');return values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-authorized',action='store_true',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)is None:raise ValueError('safe exact tag')
    if a.output.parent.resolve()!=D or a.output.exists():raise ValueError('fresh private D result')
    if hashlib.sha256(OLD_AUDITOR.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('pinned original209 independent auditor')
    old=module(OLD_AUDITOR,'_ROOT224_readonly_independent209');need=old.need;sha=old.sha;read=old.read;desc=old.desc
    # Source-pinned old audit requires SLO0; this actual arm deliberately uses SLO1.
    old.definitions(old.PARENT_AUDITOR,{'admission_and_full_statistics'},(
        ("from empty197 import audit_trace","from empty209 import audit_trace"),
        ("c['spec_rhythm_slo_aware_admission_enabled']==c['spec_rhythm_target_slo_first_enabled']==c['spec_rhythm_prefill_coalesce_enabled']==0","c['spec_rhythm_slo_aware_admission_enabled']==1 and c['spec_rhythm_target_slo_first_enabled']==c['spec_rhythm_prefill_coalesce_enabled']==0"),
        ("actual SLO0/TARGET0/original coalesce1","actual SLO1/TARGET0/original coalesce1")))
    receipt=DONOR/f'receipt-{a.tag}.json'
    need(a.terminal_authorized and sha(receipt)==a.expected_receipt_sha256,'exact ROOT terminal authorization')
    r=read(receipt)
    need(r['schema']=='ROOT.V224.cached-parent-full-supervisor.v1'and r['status']in('complete','failed')
         and r['tag']==a.tag and r['action']==old.ACTION,'actual terminal current211 receipt')
    sys.meta_path.insert(0,old.Firewall());sys.path[:0]=[str(P),str(DONOR)]
    actual=DONOR/'actual'/a.tag
    completed_compute_exit_marker_failure=(r['status']=='failed'and len(r['errors'])==1
        and 'actual complete full32 cap4 workers and closed process pool'in r['errors'][0]
        and r['child']['exit_code']==0
        and read(actual/'worker-exit.json')==dict(closed=True,all_exited=False)
        and (actual/'completed.json').exists())
    result=dict(schema='ROOT.V224.independent-terminal-HOST.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        receipt=desc(receipt),guardian_status=r['status'],script=desc(__file__),source_peer_review=False,
        errors=[],NPU_used=False,Torch_imported=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
    try:
        import guard224
        guard224.EXPECTED_FREEZE=FREEZE;guard224.EXPECTED_ARCHIVE=ARCHIVE
        freeze,sources,dependency=guard224.prepared()
        need(sha(P/'FREEZE224.json')==FREEZE and sha(P/'SOURCE224.tar.gz')==ARCHIVE and len(sources)==1905,
             'all actual frozen211 source/archive1905 verified')
        need(r['readiness']==json.loads(json.dumps(dependency)) and r['sources_unchanged']is True and r['owned_remaining']==[],
             'actual new/donor source closure and owned process cleanup')
        for label in('pre_verifier','post_verifier'):
            v=r[label];need(v['exit_code']==0 and sha(v['path'])==v['sha256']and read(v['path'])==v['actual_result'],
                            'saved verifier receipt bytes '+label)
            need(v['actual_result']['status']=='pass'and v['actual_result']['errors']==[]
                 and tuple(v['actual_result'][k]for k in('checked_frozen_files','checked_external_files','checked_packages'))==(10496,15,212),
                 'original actual10496/15/212 '+label)
        for label in('devices_before','devices_after'):
            v=r[label];need(v['all_idle']is v['four_910B2']is True and set(v['processes'])=={'2','5','6','7'}
                 and all(c['exit_code']==0 and c['idle']is True for c in v['processes'].values()),'actual saved all4 idle '+label)
        need(r['execution_lock']['path']==str(B/'.ROOT_NPU_EXECUTION.lock')and r['execution_lock']['nonblocking_exclusive']is True,
             'one ROOT lock, no outer flock')
        if r['child']is not None:need(r['child']['argv']==guard224.launch_argv(a.tag),'actual fixed new211 launcher')
        result['source_provenance']=dict(freeze=desc(P/'FREEZE224.json'),archive=desc(P/'SOURCE224.tar.gz'),
            source_members=len(sources),original209_source_only_dependency=dependency,
            source_peer_review=False,no_donor_device_qualification_inherited=True)
    except Exception as e:result['errors'].append(dict(stage='source_or_terminal_provenance',error=repr(e)))
    files=sorted(actual.rglob('*'))if actual.exists()else[]
    need(not any(p.is_symlink()for p in files),'actual private outputs have no aliases')
    manifest={str(p):desc(p)for p in files if p.is_file()};mp=a.output.with_name(a.output.stem+'_RAW_MANIFEST.json')
    need(not mp.exists(),'fresh actual raw manifest');mp.write_text(json.dumps(manifest,indent=2));result['raw_manifest']=dict(**desc(mp),files=len(manifest))
    if (r['status']=='complete'or completed_compute_exit_marker_failure)and 'source_provenance'in result:
        try:
            import audit209
            original=audit209.audit(actual,(32,));complete=read(actual/'completed.json')
            need(original==complete['audit']and original['passed']is True and r['child']['exit_code']==0,'completed original producer computation audit')
            if completed_compute_exit_marker_failure:
                result['preserved_execution_failure']=dict(reason='API worker-exit snapshot all_exited=False; later owned cleanup and saved all4idle verified',original_errors=r['errors'],execution_upgraded_to_success=False)
            else:
                need(original==r['actual_full_audit']and r['errors']==[]and r['actual_execution_complete']is True,'same completed guardian audit')
                need(read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'all workers closed')
            need(original['native_zero_subtractions']==252
                 and original['current_multiowner_eager_component']['Native_zero_scalar_subtractions']==336
                 and original['seven_usage']['Native_zero_subtractions']==1176,'all current original/new zero arithmetic')
            reports=[read(actual/f'rank{rank}-local-diagnostic.json')for rank in range(4)]
            result['actual_cached_dispatch']=cache_records(reports,need)
            policies=[]
            for rank,report in enumerate(reports):
                policy=report['ROOT224_tie_policy'];initial=report['overlay']['ROOT224_target_tie_policy']
                need(policy['policy']=='FP32-refinement-of-complete-BF16-maximum-ties-cap4-v1'
                    and policy['rank']==rank and policy['installed']is(rank!=0)
                    and policy['installed_before_constructor']is True and policy['actual_engine_role_verified']is True
                    and policy['complete_ties_per_shard_cap']==4 and policy['overflow_policy']=='original minimum token ID',
                    'actual explicitly changed target tie policy and role')
                need(policy['model_weights_changed']is policy['model_body_changed']is policy['draft_sampler_changed']is False
                    and policy['greedy_tie_policy_changed']is True
                    and policy['weight_dtype']==policy['body_embedding_dtype']=='torch.bfloat16',
                    'original BF16 model/weights with declared new FP32 tie comparison')
                need(initial['python_head_entries']==0 and policy['python_head_entries']>0 if rank else initial['python_head_entries']==policy['python_head_entries']==0,
                    'new target wrapper actually entered during original fresh qualification')
                for name,digest in policy['files'].items():need(sha(name)==digest and freeze['files'].get(name)==digest,'new helper in exact frozen flat source map')
                need(sha(policy['original_source'])==policy['original_source_sha256'],'original native greedy source retained')
                reference=read(actual/'postfull-independent-FIA-AR'/f'rank{rank}-summary.json')
                need(reference['target_greedy_tie_policy']=='ROOT224 complete BF16 top ties up to4 per shard refined in FP32; original choice on overflow',
                    'independent own-AR explicitly declares same numerical rule')
                policies.append(policy)
            result['actual_explicit_new_target_tie_policy']=dict(by_rank=policies,old_BF16_tie_outputs_are_not_new_policy_reference=True,
                new_policy_applies_to_tree_and_independent_AR=True,model_weights_and_body_unchanged=True)

            result['original_producer_audit']=original
            result['independent_original_integer_components']=old.component_integer_verdicts(actual)
            result['independent_seven_usage']=old.independent_seven_usage(actual)
            traces=[read(actual/'windows/window32/full463'/f'rank{rank}-full463-actual-trace.json.gz')for rank in range(4)]
            need(all(t['decode_counters']['spec_rhythm_slo_aware_admission_enabled']==1 and t['decode_counters']['spec_rhythm_target_slo_first_enabled']==0 for t in traces),'actual all4 full SLO-admission1 counters')
            result['independent_multiowner']=old.independent_multiowner(actual,traces,reports)
            result['retained_full463']=old.full_statistics(traces,reports)
            cost_rows=[]
            for report,trace in zip(reports,traces,strict=True):
                need(report['overlay']['cost_aware_N9to16_budget']is True,'actual explicit cost-aware budget overlay')
                state=report['ROOT224_cached_parent_dispatch']['cost_aware_N9to16_budget']
                need(state['private_shape_bindings']==1,'exact original candidate closure privately rebound once')
                from mapper173 import expected_geometry as original_geometry
                table=[dict(N=n,Q=q,M=original_geometry(n,q,'causal')[0],S=original_geometry(n,q,'causal')[1])for n in range(9,17)for q in range(4*n+1,5*n)]
                need(state['original_cost_table']==table,'all92 actual precomputed original mapper costs independently recomputed')
                row=state['by_generation'][str(trace['source_generation'])]
                saved=report['windows']['traces'][-1]
                need(saved['role']=='full463','actual annotated full463 trace')
                policy=saved['actual_split_work_counts']['delta']['policy']
                actual_shapes=sum(policy['draft_normal_N'+str(n)+'_shape_calls']for n in range(33))
                need(row['shape_calls']==actual_shapes and actual_shapes>0,'actual full-generation shapes match original retained counters')
                need(all(type(v)is int and v>=0 for r0 in state['by_generation'].values()for v in r0.values()),'all original-generation repair counters nonnegative integer')
                for r0 in state['by_generation'].values():
                    need(r0.get('eligible_mixed_N9to16',0)==r0.get('repaired_calls',0)+r0.get('blocked_by_original_limits',0)
                         and r0.get('added_selected_candidates',0)==r0.get('upgraded_owners',0)
                         and r0.get('repaired_calls',0)==sum(r0.get('repaired_N'+str(n)+'_calls',0)for n in range(9,17))
                         and r0.get('repaired_calls',0)==r0.get('repaired_old_M96_calls',0)+r0.get('repaired_old_M128_calls',0)
                         and r0.get('proposed_target_physical_queries_saved',0)==16*r0.get('repaired_old_M96_calls',0)+48*r0.get('repaired_old_M128_calls',0),
                         'independent repair/cap-blocked/selected accounting')
                cost_rows.append(row)
            result['independent_cost_aware_budget_accounting']=dict(by_rank=cost_rows,
                proposed_query_saving_is_not_subtracted_from_actual_timing=True,
                native_keys_unchanged=True,budget_grants_may_change=True)

            from audit_reference224 import audit as independent_own_FIA_AR
            result['independent_postfull_own_FIA_AR']=independent_own_FIA_AR(actual)
            packed=[]
            for rank,(report,trace)in enumerate(zip(reports,traces,strict=True)):
                need(report['overlay']['packed_target_KV_preflight']is True,'actual declared packed target preflight hook')
                state=report['ROOT224_slot_preflight']
                need(state['shadow']is False and state['qualified']is True and state['shadow_pairs']>0,
                     'actual component original/new HOST-mapping and device-view asserts complete before performance')
                need(state['shadow_pairs']==state['shadow_draft_pairs'if rank==0 else'shadow_target_pairs'],
                     'actual same role shadow counts')
                row=state['by_generation'][str(trace['source_generation'])]
                calls=len(trace['normal_target_calls']);owners=sum(c['N']for c in trace['normal_target_calls'])
                need(row['calls']>=calls and row.get('direct_packed_target_calls',0)==(0 if rank==0 else calls)
                     and row.get('target_rows',0)==(0 if rank==0 else owners)
                     and row['calls']==row.get('direct_packed_target_calls',0)+row.get('unchanged_other_calls',0),
                     'independent actual all-call and all-owner packed mapping path accounting')
                dtype=state['actual_target_dtype_counts']
                need((not dtype if rank==0 else bool(dtype)) and all(k in('torch.int32','torch.int64')and type(v)is int and v>0 for k,v in dtype.items()),
                     'actual original target mapping dtype coverage, no value narrowing')
                packed.append(state)
            result['source_backed_actual_packed_preflight_shadow']=dict(by_rank=packed,
                actual_runtime_original_new_integer_maps_and_pending_views_asserted=True,
                HOST_reexecuted_unretained_device_tensors=False,
                complete_independent_clock_arithmetic_unchanged=True)
            result['actual_compatibility_classification']=dict(old_padding_behavior_equal=original['old_padding_behavior_equal'],
                numerical_variant_only=original['numerical_variant_only'],whole_workload_AR_qualified=False)
        except Exception as e:result['errors'].append(dict(stage='actual_components_and_full463',error=repr(e)))
    elif r['status']=='failed':
        result['failure_only']=dict(errors=r['errors'],child=r['child'],Goodput=None,no_timeout_inferred_from_generic_text=True)
    need(not old.Firewall.attempts and not any(n.split('.')[0]in old.BACKENDS for n in sys.modules),'stdlib HOST only')
    result['status']=('passed_terminal_component_and_full463_audit'if r['status']=='complete'else'terminal_failed_evidence_only')if not result['errors']else'terminal_audit_errors'
    if completed_compute_exit_marker_failure and not result['errors']:
        result['status']='completed_computation_audited_execution_failed_exit_marker'
    result['limitations']=['Same frozen209 weights/body/protocol/layout and current budget, with an explicit new target FP32 maximum-tie rule applied to both tree and own-AR. This changes numerical selection among BF16 ties; original BF16 token results are not the new numerical reference. Fresh qualification is required.',
        'HOST recomputes integer verdicts, history, metadata, counters and clocks from retained raw. Unretained hidden/KV tensors are not reexecuted.',
        'Same-model output equality or a single full result does not establish whole-workload independent AR, stable1.5x or final matched-clock official baseline performance.']
    with a.output.open('x')as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(output=desc(a.output),status=result['status'],errors=result['errors'])))
    return int(bool(result['errors']))


if __name__=='__main__':raise SystemExit(main())
