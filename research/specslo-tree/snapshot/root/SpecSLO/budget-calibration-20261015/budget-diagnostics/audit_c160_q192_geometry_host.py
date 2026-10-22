"""Read-only HOST geometry/source audit; never a device/throughput proof."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-s35-combined-controls-launchfix-v116'
D=B/'budget-diagnostics'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
PROFILE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def source_rows(path,needles):
    lines=Path(path).read_text().splitlines()
    return {needle:[i+1 for i,line in enumerate(lines) if needle in line] for needle in needles}


def main():
    started=time.perf_counter()
    names=('budget_policy.py','candidate_budget_policy.py','tree_entry.py','startup_config_v75.py','runtime_tree_decode.py',
        'target_layout.py','finite_inventory_plan.py','target_inventory.py','flexible_scope.py','live_runtime_hooks.py',
        'shape_scope_v101.py','local_qualification_scope.py','production_runtime_v101.py','poisson_coverage_probe.py',
        'audit_capacity64_v96.py','combined_controls_v113.py','root_seed.py','PREPARED_SOURCE_HASHES_V116.json')
    paths=[P/x for x in names]+[N/x for x in ('native_engine.py','native_model.py','native_graph.py','native_cache.py','spec_rhythm.py')]+[PROFILE,
        B/'tree-qualified44-cost-v114/EXACT_QUALIFIED_KEY_INVENTORY.json',
        B/'tree-active-cap-controls-v103/startup-progress/MC2-preflight-rank1.json']
    before={str(p):sha(p) for p in paths}
    old=load(P/'target_layout.py','_C160_readonly_layout')
    export=json.loads((B/'tree-qualified44-cost-v114/EXACT_QUALIFIED_KEY_INVENTORY.json').read_text());qualified={tuple(x) for x in export['geometries']}
    actual_inventory=[]
    for path,record in export['actual_target_inventory_receipts'].items():
        if sha(path)!=record['sha256']:raise RuntimeError('Actual44 Native receipt changed')
        before[path]=sha(path);actual_inventory.append(dict(path=path,sha256=record['sha256'],keys=len(record['actual_keys'])))
    profile=json.loads(PROFILE.read_text());mrows=sorted({r['m'] for r in profile['measurements']})
    if len(qualified)!=44 or max(k[0] for k in qualified)!=165 or any(k[0]==192 for k in qualified):raise RuntimeError('Current44 geometry assumptions changed')
    if any(row['m']==192 for row in profile['measurements']):raise RuntimeError('Exact M192 profile scope changed')
    preflight=json.loads((B/'tree-active-cap-controls-v103/startup-progress/MC2-preflight-rank1.json').read_text())
    chained=preflight['model_chained_row_counts']
    if preflight['status']!='passed' or 197 not in chained or 192 in chained:raise RuntimeError('Actual Native MC2 preflight row intersection changed')
    rejection=None
    try:old.envelope(32,192,'full9')
    except ValueError as error:rejection=str(error)
    if rejection is None:raise RuntimeError('Old admitted domain changed')
    legal197=[]
    for s in old.SEGMENTS:
        lengths=old._lengths(32,192,197,s)
        if lengths is not None:legal197.append(dict(M=197,S=s,dummy=lengths,new_native_key=(197,s,'full9') not in qualified))
    mixed=[]
    for c in (96,160):
        q=32+c;d=35-1-32;cap=197-q-2*d
        mixed.append(dict(N=32,C=c,Q=q,physical_M=197,physical_S=35,private_scratch_requests=d,
            max_genuine_prefill_queries=min(128,cap),minimum_scratch_queries=2*d,
            example_partition=[3+1]*32+[65,2,2] if c==96 else [5+1]*32+[1,2,2]))
    # Source-only scope arithmetic. No original admitted domain or key list is
    # modified, and these new partitions do not gain qualification from keys.
    extended=[];uncovered=[]
    added={(197,38,'causal'),(197,38,'full9'),(197,34,'full9')}
    uncovered_after=[]
    for n in range(1,33):
        for q in range(2*n,min(6*n,n+160,192)+1):
            for family,ok in (('causal',q<=5*n),('full9',q>=2*n+4)):
                if not ok:continue
                extended.append((n,q,family))
                fits=lambda key:key[2]==family and old._lengths(n,q,key[0],key[1]) is not None
                if not any(fits(k) for k in qualified):uncovered.append((n,q,family))
                if not any(fits(k) for k in qualified|added):uncovered_after.append((n,q,family))
    source_locations={
        'policy_default_and_caps':dict(path=str(P/'budget_policy.py'),lines=source_rows(P/'budget_policy.py',('target_queries=128','candidate_roof =','left = self.candidate_roof','queries > self.target_queries'))),
        'wrapper_ignores_call_roof':dict(path=str(P/'tree_entry.py'),lines=source_rows(P/'tree_entry.py',('policy = TreeBudgetPolicy','def shape(instance','policy.allocate(','verification_roof=policy.candidate_roof','policy.check_target'))),
        'config_roof':dict(path=str(P/'startup_config_v75.py'),lines=source_rows(P/'startup_config_v75.py',('spec_rhythm_verification_budget=96','max_num_seqs=64','num_kvcache_blocks=544'))),
        'normal_mapper_domain':dict(path=str(P/'target_layout.py'),lines=source_rows(P/'target_layout.py',('TOKENS=','SEGMENTS=','real_queries<=min','original=legacy_envelope','candidates='))),
        'finite_plan_domain':dict(path=str(P/'finite_inventory_plan.py'),lines=source_rows(P/'finite_inventory_plan.py',('sum(result)>96','count+96,128','candidate_roof=96'))),
        'mixed_domain':dict(path=str(P/'flexible_scope.py'),lines=source_rows(P/'flexible_scope.py',('HARDWARE_CANDIDATE_ROOF =','REAL_VERIFICATION_QUERY_ROOF =','d = physical_S-1-N','cap = PHYSICAL_M-Q-2*d'))),
        'mixed_live_guard':dict(path=str(P/'live_runtime_hooks.py'),lines=source_rows(P/'live_runtime_hooks.py',('sum(budgets) <= 96','shape = envelope'))),
        'large_normal_cap4':dict(path=str(P/'candidate_budget_policy.py'),lines=source_rows(P/'candidate_budget_policy.py',('4 if len(normal) > 16',))),
        'ready_bound_not_aggregate':dict(path=str(N/'native_engine.py'),lines=source_rows(N/'native_engine.py',('def _bound_spec_rhythm_tree_ready','ticket.gamma <= roof','packed = pack_selected_tree_plan(plan, range(roof))'))),
        'controller_subset_budget':dict(path=str(N/'spec_rhythm.py'),lines=source_rows(N/'spec_rhythm.py',('selected_tokens + candidate_counts[index] > verification_budget',))),
        'cache_row_legality':dict(path=str(N/'native_model.py'),lines=source_rows(N/'native_model.py',('num_cache_blocks < max_num_seqs','torch.zeros(cache_shape',))),
        'lazy_page_lifecycle':dict(path=str(N/'native_cache.py'),lines=source_rows(N/'native_cache.py',('block_tables.extend','def activate_sequence','def ensure_capacity','def release_sequence','ref_count == 0',))),
    }
    result=dict(schema='HOST.C160-Q192-current44-and-MC2-feasibility.v1',actual_HOST_argv=sys.orig_argv,status='complete-HOST-audit-only',
        current_actual_qualified44=dict(count=44,maximum_M=165,has_FULL192_S32=False,has_normal_FULL197_S33=False,receipts=actual_inventory),
        current_admitted_scope=dict(normal_N=32,C=96,Q=128,old_N32_Q192_rejection=rejection),
        current_frozen_MC2=dict(profile_path=str(PROFILE),measurement_count=len(profile['measurements']),measured_M_rows=mrows,
            M192_measurements=0,M197_measurements=[{k:r[k] for k in ('m','k','n','dtype','changed_input_replays')} for r in profile['measurements'] if r['m']==197],
            actual_Native_chained_rows=chained,M192_route_qualification_present=False,M197_route_qualification_present=True),
        proposed_single_local_control=dict(N=32,budgets=[5]*32,C=160,Q=192,new_physical_197_options=legal197,
            preferred_M=197,preferred_S=33,scratch_queries=[5],Native_greedy_and_hidden_keys_are_new=True,
            MC2_M197_route_can_be_observed_from_existing_profile_but_new_partition_needs_actual_gate=True,
            existing_independent_all32_B5_normal_FULL192_or_FULL197_witness_bound=False,
            current_S35_N32_representatives='allB3 C96, and16B5+16B1 C96; neither all32B5',
            new_S33_normal_FULL9_is_distinct_from_existing_S35_mixed_FULL128=True),
        mixed_prefill_geometry=mixed,
        source_only_broader_domain=dict(C=160,Q=192,N_max=32,reachable_family_geometries=len(extended),
            not_coverable_by_current44=len(uncovered),not_covered_first12=uncovered[:12],
            one_S33_key_does_not_qualify_general_dynamic_roof=True,
            additional_three_geometries_covering_extended_domain_in_HOST=sorted(added),
            uncovered_after_three_Host_only=len(uncovered_after),all_are_new_Native_keys=True,actual_numerical_qualification=False),
        capacity=dict(active_limit_unchanged=64,service_rows=463,scratch_rows=64,actual_flag0_storage_rows=527,
            original_KV_pages=544,constructor_544_ge_527=True,inactive_queued_requests_do_not_each_pin_one_page=True,
            existing_fixed_KV_tensor_bytes_per_target_rank=6845104128,existing_fixed_KV_tensor_GiB_per_target_rank=6.375,
            changing_candidate_roof_does_not_reallocate_the_fixed_KV_tensor=True,
            extra_query_page_bound_at_fixed_prefixes=32,full_new_trace_page_highwater_unknown=True,
            existing_V96_actual_full463_page_peak=470,existing_V96_peak_is_not_new_C160_measurement=True,
            graph_memory_after_new_S33_capture_unknown=True),
        required_new_gates=['HOST exact source/domain/budget/accounting delta and all rejects',
            'new M197/S33 FULL9 standard greedy+hidden actual capture and natural first/changed/restored, no flag assignment',
            'fresh independent sameM/S/logical-input all197 hidden and all64queryKV, full valid-prefix/hash/lease, allreal greedy/verdict/commit/nextdraft',
            'actual MC2 all128 route manifest agrees exactM197 and raw dispatch/fallback/exception evidence',
            'two complete zero-discovery traces then unified original seal; new partition guards do not borrow key flags',
            'actual retained3.11/CANN9 memory/512keys/544pages and raw latency, then independent full463 SLO/GP if chosen'],
        source_locations=source_locations,source_hashes=before,
        no_source_changed=True,no_NPU_initialized=True,no_model_initialized=True,performance_claim=False,
        C160_is_current_B5x32_scope_not_machine_global_max_B=True)
    after={path:sha(path) for path in before}
    if after!=before or 'torch' in sys.modules:raise RuntimeError('Read-only/pure HOST audit closure changed')
    result['elapsed_seconds']=time.perf_counter()-started
    out=D/'C160_Q192_NORMAL_ROOF_FEASIBILITY_HOST_AUDIT.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),elapsed_seconds=result['elapsed_seconds'],current44_maxM=165,
        M192_profile_measurements=0,preferred_197_S33_dummy=[5],extended_domain=len(extended),uncovered_current44=len(uncovered),uncovered_after_three=len(uncovered_after))))


if __name__=='__main__':main()
