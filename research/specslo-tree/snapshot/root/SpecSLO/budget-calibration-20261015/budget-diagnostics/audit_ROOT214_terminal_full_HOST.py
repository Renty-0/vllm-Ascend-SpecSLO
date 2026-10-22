"""Future exact-authorized ROOT214 HOST audit. No actual reads before terminal CLI."""
import argparse,datetime,hashlib,importlib.util,json,re,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-cost-aware-mixed16-budget-full-v214'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
FREEZE='52252f8f00d1413e3340952bb94c302ff65023f38f06f413df1ebf3125c2a714'
ARCHIVE='54a3d15ee4ebfeef713950912c8d8933ae5aed762225af2265a85cf87f50386e'
OLD_AUDITOR=D/'audit_v209_terminal_HOST.py'
OLD_SHA='fe8a050d3dce287787017bdf9b0f319bd2577e641e40a7baef7f2df785c0db5b'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value


def cache_records(reports,need):
    values=[]
    for rank,r in enumerate(reports):
        need(r['rank']==rank and r['overlay']['ROOT214_overlay']=='cached-parent-SLO-admission1-cost-aware-mixed16' and r['overlay']['SLO_AWARE_ADMISSION']is True,
             'actual current211 overlay and rank')
        cache=r['ROOT214_cached_parent_dispatch']
        need(cache['initial_parent_source_checks']==1 and type(cache['cached_parent_returns'])is int
             and cache['cached_parent_returns']>1 and cache['canonical_mapper_or_base_module_mutated']is False,
             'once source-checked binding and actual repeated cache returns')
        need(cache['canonical_mapper_module']==str(DONOR/'mapper209.py')
             and cache['canonical_parent_module']==str(B/'tree-causal80-real16-bankfix-full-v173/mapper173.py'),
             'only exact canonical mapper/parent source objects cached')
        for name in('actual_helper_origins_before_constructor','actual_helper_origins_after_constructor'):
            need(bool(r[name]['ROOT214_explicit_overlay_sources']),'new source provenance not hidden under old module origin')
        values.append(cache)
    need(len(values)==4,'actual all4 cached mapper records');return values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-authorized',action='store_true',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)is None:raise ValueError('safe exact tag')
    if a.output.parent.resolve()!=D or a.output.exists():raise ValueError('fresh private D result')
    if hashlib.sha256(OLD_AUDITOR.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('pinned original209 independent auditor')
    old=module(OLD_AUDITOR,'_ROOT214_readonly_independent209');need=old.need;sha=old.sha;read=old.read;desc=old.desc
    receipt=DONOR/f'receipt-{a.tag}.json'
    need(a.terminal_authorized and sha(receipt)==a.expected_receipt_sha256,'exact ROOT terminal authorization')
    r=read(receipt)
    need(r['schema']=='ROOT.V214.cached-parent-full-supervisor.v1'and r['status']in('complete','failed')
         and r['tag']==a.tag and r['action']==old.ACTION,'actual terminal current211 receipt')
    sys.meta_path.insert(0,old.Firewall());sys.path[:0]=[str(P),str(DONOR)]
    actual=DONOR/'actual'/a.tag
    result=dict(schema='ROOT.V214.independent-terminal-HOST.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        receipt=desc(receipt),guardian_status=r['status'],script=desc(__file__),source_peer_review=False,
        errors=[],NPU_used=False,Torch_imported=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
    try:
        import guard214
        guard214.EXPECTED_FREEZE=FREEZE;guard214.EXPECTED_ARCHIVE=ARCHIVE
        freeze,sources,dependency=guard214.prepared()
        need(sha(P/'FREEZE214.json')==FREEZE and sha(P/'SOURCE214.tar.gz')==ARCHIVE and len(sources)==1619,
             'all actual frozen211 source/archive1619 verified')
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
        if r['child']is not None:need(r['child']['argv']==guard214.launch_argv(a.tag),'actual fixed new211 launcher')
        result['source_provenance']=dict(freeze=desc(P/'FREEZE214.json'),archive=desc(P/'SOURCE214.tar.gz'),
            source_members=len(sources),original209_source_only_dependency=dependency,
            source_peer_review=False,no_donor_device_qualification_inherited=True)
    except Exception as e:result['errors'].append(dict(stage='source_or_terminal_provenance',error=repr(e)))
    files=sorted(actual.rglob('*'))if actual.exists()else[]
    need(not any(p.is_symlink()for p in files),'actual private outputs have no aliases')
    manifest={str(p):desc(p)for p in files if p.is_file()};mp=a.output.with_name(a.output.stem+'_RAW_MANIFEST.json')
    need(not mp.exists(),'fresh actual raw manifest');mp.write_text(json.dumps(manifest,indent=2));result['raw_manifest']=dict(**desc(mp),files=len(manifest))
    if r['status']=='complete'and 'source_provenance'in result:
        try:
            import audit209
            original=audit209.audit(actual,(32,));complete=read(actual/'completed.json')
            need(original==complete['audit']==r['actual_full_audit']and original['passed']is True
                 and r['errors']==[]and r['child']['exit_code']==0 and r['actual_execution_complete']is True,
                 'same original donor audit, completed and new actual guardian')
            need(read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'all workers closed')
            need(original['native_zero_subtractions']==252
                 and original['current_multiowner_eager_component']['Native_zero_scalar_subtractions']==336
                 and original['seven_usage']['Native_zero_subtractions']==1176,'all current original/new zero arithmetic')
            reports=[read(actual/f'rank{rank}-local-diagnostic.json')for rank in range(4)]
            result['actual_cached_dispatch']=cache_records(reports,need)
            result['original_producer_audit']=original
            result['independent_original_integer_components']=old.component_integer_verdicts(actual)
            result['independent_seven_usage']=old.independent_seven_usage(actual)
            traces=[read(actual/'windows/window32/full463'/f'rank{rank}-full463-actual-trace.json.gz')for rank in range(4)]
            need(all(t['decode_counters']['spec_rhythm_slo_aware_admission_enabled']==1 and t['decode_counters']['spec_rhythm_target_slo_first_enabled']==0 for t in traces),'actual all4 full SLO-admission1 counters')
            result['independent_multiowner']=old.independent_multiowner(actual,traces,reports)
            result['retained_full463']=old.full_statistics(traces,reports)
            cost_rows=[]
            for report,trace in zip(reports,traces,strict=True):
                need(report['overlay']['cost_aware_mixed16_budget']is True,'actual explicit cost-aware budget overlay')
                state=report['ROOT214_cached_parent_dispatch']['cost_aware_mixed16_budget']
                need(state['private_shape_bindings']==1,'exact original candidate closure privately rebound once')
                row=state['by_generation'][str(trace['source_generation'])]
                saved=report['windows']['traces'][-1]
                need(saved['role']=='full463','actual annotated full463 trace')
                policy=saved['actual_split_work_counts']['delta']['policy']
                actual_shapes=sum(policy['draft_normal_N'+str(n)+'_shape_calls']for n in range(33))
                need(row['shape_calls']==actual_shapes and actual_shapes>0,'actual full-generation shapes match original retained counters')
                need(all(type(v)is int and v>=0 for r0 in state['by_generation'].values()for v in r0.values()),'all original-generation repair counters nonnegative integer')
                for r0 in state['by_generation'].values():
                    need(r0.get('eligible_mixed16',0)==r0.get('repaired_calls',0)+r0.get('blocked_by_original_limits',0)
                         and r0.get('added_selected_candidates',0)==r0.get('upgraded_owners',0)
                         and r0.get('proposed_target_physical_queries_saved',0)==48*r0.get('repaired_calls',0),
                         'independent repair/cap-blocked/selected accounting')
                cost_rows.append(row)
            result['independent_cost_aware_budget_accounting']=dict(by_rank=cost_rows,
                proposed_query_saving_is_not_subtracted_from_actual_timing=True,
                native_keys_unchanged=True,budget_grants_may_change=True)

            result['actual_compatibility_classification']=dict(old_padding_behavior_equal=original['old_padding_behavior_equal'],
                numerical_variant_only=original['numerical_variant_only'],whole_workload_AR_qualified=False)
        except Exception as e:result['errors'].append(dict(stage='actual_components_and_full463',error=repr(e)))
    elif r['status']=='failed':
        result['failure_only']=dict(errors=r['errors'],child=r['child'],Goodput=None,no_timeout_inferred_from_generic_text=True)
    need(not old.Firewall.attempts and not any(n.split('.')[0]in old.BACKENDS for n in sys.modules),'stdlib HOST only')
    result['status']=('passed_terminal_component_and_full463_audit'if r['status']=='complete'else'terminal_failed_evidence_only')if not result['errors']else'terminal_audit_errors'
    result['limitations']=['Same frozen209 math/protocol/layout with explicit cache, SLO admission1 and bounded mixed16 budget repair; budget grants can change. Source reviews do not transfer numerical qualification.',
        'HOST recomputes integer verdicts, history, metadata, counters and clocks from retained raw. Unretained hidden/KV tensors are not reexecuted.',
        'Same-model output equality or a single full result does not establish whole-workload independent AR, stable1.5x or final matched-clock official baseline performance.']
    with a.output.open('x')as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(output=desc(a.output),status=result['status'],errors=result['errors'])))
    return int(bool(result['errors']))


if __name__=='__main__':raise SystemExit(main())
