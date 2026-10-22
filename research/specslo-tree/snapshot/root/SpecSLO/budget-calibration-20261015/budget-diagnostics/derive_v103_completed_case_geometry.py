"""Light HOST attribution for one completed case, while its session may run.

Does not read the running supervisor receipt, audit final cleanup, import
Torch, recompress traces, execute a benchmark or claim a whole-session pass.
"""
import ast
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
V103=B/'tree-active-cap-controls-v103/local-output/active32-64-cap3-4-full463-v1/active32-cap3/rank1-full463-actual-trace.json.gz'
V96=B/'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1/rank1-full463-actual-trace.json.gz'
V63=B/'tree-worker-window-qualified-B5-v63/measurement-output/runs/candidate-on-worker-window-qualified-rootseed-B5-full463-r1/result.json'


def sha(blob):return hashlib.sha256(blob).hexdigest()


def load(path,compressed=False):
    blob=path.read_bytes();value=json.loads(gzip.decompress(blob) if compressed else blob)
    return value,dict(path=str(path),sha256=sha(blob),bytes=len(blob))


def geometry(calls):
    hist=Counter();N=Counter();Bhist=Counter();n16=Counter();logical=physical=rows=candidates=0
    q16=[];n16M128=[]
    for c in calls:
        n=c['N'];q=c['real_query_count'];m=c['physical_M'];s=c['physical_S'];family=c['family']
        assert len(c['budgets'])==len(c['contexts'])==n
        assert q==n+sum(c['budgets']) and len(c['physical_query_partition'])==s and sum(c['physical_query_partition'])==m
        hist[f'{family}/M{m}/S{s}']+=1;N[str(n)]+=1;Bhist.update(map(str,c['budgets']))
        logical+=q;physical+=m;rows+=n;candidates+=sum(c['budgets'])
        if n==16:
            n16[f'{family}/M{m}/S{s}']+=1;q16.append(q)
            if family=='causal' and m==128:n16M128.append(q)
    return dict(normal_target_calls=len(calls),normal_owner_rounds=rows,normal_candidates=candidates,
        logical_queries=logical,physical_queries=physical,padding_queries=physical-logical,
        padding_fraction_of_physical=(physical-logical)/physical,physical_queries_per_logical_query=physical/logical,
        physical_family_histogram=dict(sorted(hist.items())),N_histogram=dict(sorted(N.items(),key=lambda x:int(x[0]))),
        B_histogram=dict(sorted(Bhist.items())),N16_physical_family_histogram=dict(sorted(n16.items())),
        N16_real_Q_range=[min(q16),max(q16)] if q16 else None,
        N16_causal_M128_calls=len(n16M128),N16_causal_M128_padding_queries=sum(128-q for q in n16M128))


def normal_gain(trace):
    byowner=defaultdict(list);mixed={i for e in trace['logical_events'] if e['kind']=='plan' for i in e['target_ids']}
    for c in trace['normal_target_calls']:
        for i,b,prefix in zip(c['sequence_ids'][:c['N']],c['budgets'],c['contexts'],strict=True):
            byowner[i].append(dict(B=b,prefix=prefix,N=c['N']))
    excluded=Counter();groups=defaultdict(lambda:Counter());owner_count=rounds=delivered=accepted=stable_rounds=stable_delivered=0
    terminal_extra=0
    for i,result in enumerate(trace['generate_results']):
        calls=byowner[i]
        if i in mixed:excluded['mixed_owner']+=1;continue
        if len(calls)!=result['verification_rounds'] or sum(c['B'] for c in calls)!=result['verified_draft_tokens']:
            excluded['normal_owner_ledger_mismatch']+=1;continue
        prefix=len(trace['tokenized_inputs'][i]);assert calls and calls[0]['prefix']==prefix
        contexts=[c['prefix'] for c in calls]+[prefix+255]
        advances=[q-p for p,q in zip(contexts,contexts[1:])]
        assert sum(advances)==255 and all(1<=g<=min(c['B'],4)+1 for g,c in zip(advances,calls,strict=True))
        extra=result['accepted_draft_tokens']-sum(g-1 for g in advances)
        assert 0<=extra<=min(calls[-1]['B'],4)+1-advances[-1]
        owner_count+=1;rounds+=len(calls);delivered+=255;accepted+=sum(g-1 for g in advances);terminal_extra+=extra
        for g,c in zip(advances,calls,strict=True):
            if prefix+255-c['prefix']<5:excluded['terminal_horizon_lt5_round']+=1;continue
            stable_rounds+=1;stable_delivered+=g
            key=f"{result['slo_class']}/B{c['B']}";groups[key]['rounds']+=1;groups[key]['delivered']+=g
    return dict(owners=owner_count,normal_owner_rounds=rounds,actual_delivered_tokens=delivered,
        actual_g_per_normal_owner_round_including_terminal_clip=delivered/rounds,
        accepted_nodes_implied_by_actual_normal_output_after_bonus=accepted,
        accepted_nodes_per_normal_owner_round_including_terminal_clip=accepted/rounds,
        Native_accepted_nodes_beyond_final_output_clip=terminal_extra,
        nonterminal_rounds=stable_rounds,nonterminal_g=stable_delivered/stable_rounds,
        byclass_B_nonterminal={k:dict(v,mean_delivered=v['delivered']/v['rounds']) for k,v in sorted(groups.items())},
        excluded=dict(excluded),full_normal_owner_candidate_and_round_ledger_matched=True,
        whole255_prefix_advance_matched=True,selected_observational_not_paired=True,
        selection_bias_and_within_owner_correlation=True)


def trace_summary(trace,source):
    assert trace['role']=='full463' and trace['scope_count']==463 and trace['rank']==1 and len(trace['generate_results'])==463
    assert all(v==0 for v in trace['native']['delta'].values()) and len(trace['native']['delta'])==21
    counters=trace['decode_counters'];wall=trace.get('host_generate_call_seconds',trace['elapsed_seconds'])
    outputs=trace['generate_results'];rounds=sum(r['verification_rounds'] for r in outputs)
    early=counters['spec_rhythm_tree_early_target_target_cycles']
    fields=('spec_rhythm_tree_early_target_submit_host_ns','spec_rhythm_tree_early_target_CPU_scheduling_gap_ns',
        'spec_rhythm_tree_early_target_residual_wall_ns','spec_rhythm_tree_early_target_total_wall_ns')
    return dict(source=source,host_generate_call_seconds=wall,Native_inner_elapsed_seconds=trace.get('native_elapsed_seconds'),
        legacy_Native_Goodput=sum(r['paper_slo_goodput_tokens'] for r in outputs)/wall,
        raw_tokens_per_second=118528/wall,original_phase_seconds=trace['phase_seconds'],
        standalone_prefill_blocking_host_seconds=counters['spec_rhythm_online_refill_ms']/1000,
        standalone_prefill_batches=len(trace['standalone_prefill_calls']),
        standalone_prefill_requests=sum(len(c['sequence_ids']) for c in trace['standalone_prefill_calls']),
        standalone_prefill_timer_is_not_pure_target_kernel=True,
        mixed_prefill_cost_is_not_independently_subtracted=True,
        target_cycles=early,target_phase_ms_per_target_cycle=trace['phase_seconds']['target']*1000/early,
        original_early_target_wall_seconds={k:counters[k]/1e9 for k in fields},
        eager_promoted=counters['spec_rhythm_tree_eager_promoted'],
        global_decode_tokens_per_owner_verification_round_including_clip=463*255/rounds,
        all_owner_verification_rounds=rounds,geometry=geometry(trace['normal_target_calls']),
        normal_owner_gain=normal_gain(trace),Native21_zero_observed=True,
        observation_is_single_completed_case_not_session_qualification=True)


def v63_summary(value,source):
    r=value['results'][0];w=r['worker_aclgraph_metrics'][1];d=r['measured_worker_aclgraph_deltas'][1]
    hist=Counter();calls=0
    for text,count in d['aclgraph_target_replay_entry_key_histogram_delta'].items():
        key=ast.literal_eval(text)
        if not key[0].startswith('target-greedy:'):continue
        shape=ast.literal_eval(key[0].split('|buffers:',1)[1]);s=shape[0][0][0];m=key[1][0]
        family='full9' if 'tree-fia-full' in key[0] else 'causal'
        hist[f'{family}/M{m}/S{s}']+=count;calls+=count
    rounds=sum(r['request_verification_rounds']);early=w['worker_spec_rhythm_tree_early_target_target_cycles']
    assert calls==early==w['worker_spec_rhythm_tree_rounds']
    return dict(source=source,e2e_seconds=r['e2e_elapsed_seconds'],Native_inner_decode_seconds=r['decode_elapsed_seconds'],
        legacy_Native_Goodput=259.3551343096774,raw_tokens_per_second=r['e2e_throughput_tokens_per_second'],
        original_phase_seconds={name:w['worker_'+name+'_seconds'] for name in ('draft','target','exchange','verify','broadcast','state_update')},
        standalone_prefill_blocking_host_seconds=w['worker_spec_rhythm_online_refill_ms']/1000,
        standalone_prefill_batches=w['worker_spec_rhythm_prefill_batches'],standalone_prefill_requests=463,
        standalone_prefill_timer_is_not_pure_target_kernel=True,target_cycles=early,
        target_phase_ms_per_target_cycle=w['worker_target_seconds']*1000/early,
        eager_promoted=w['worker_spec_rhythm_tree_eager_promoted'],all_owner_verification_rounds=rounds,
        global_decode_tokens_per_owner_verification_round_including_clip=463*255/rounds,
        global_Native_accepted_nodes_per_owner_round_before_terminal_clip=w['worker_spec_rhythm_tree_accepted_nodes']/rounds,
        geometry=dict(normal_target_calls=calls,physical_family_histogram=dict(sorted(hist.items())),
            evidence='measured_worker_aclgraph_deltas actual target-greedy keys only; hidden prefill keys excluded',
            real_N_and_B_and_contexts_missing=True,N16_causal_M128_calls=None,logical_queries=None,padding_fraction_of_physical=None),
        normal_owner_gain=None,normal_owner_gain_missing_reason='V63 full result does not retain owner/context normal-call ledger; global round ratio is not relabeled normal-only',
        independent_trials=1)


def main():
    started=time.perf_counter();case,cs=load(V103,True);old,os=load(V96,True);best,bs=load(V63)
    assert bs['sha256']=='b9cc3898a068b0e6836da08f25e6864e252e556ac68781c0bc22e1e0d5409068'
    a=trace_summary(case,cs);b=trace_summary(old,os);c=v63_summary(best,bs)
    result=dict(schema='ROOT.V103.completed-active32-cap3-case-only-attribution.v1',UTC=datetime.now(timezone.utc).isoformat(),
        actual_HOST_argv=[sys.executable,*sys.argv],source_script_sha256=sha(Path(__file__).read_bytes()),
        analysis_elapsed_seconds=time.perf_counter()-started,actual_readonly_case='active32-cap3',
        comparisons={'V103_active32_cap3':a,'V96_active64_cap5':b,'V63_best_active32_cap5':c},
        interpretation=dict(observational_cross_run_not_paired=True,
            N16_causal128_waste_is_checked_from_actual_calls_not_assumed=True,
            padding_query_ratio_is_not_measured_time_savings=True,
            owner_round_gain_has_terminal_and_selection_caveats=True,
            target_phase_is_original_wall_interval_not_pure_CPU_or_device=True,
            HOST_warmup_profile_does_not_prove_cap3_speedup=True),
        parent_job_may_still_be_running=True,supervisor_receipt_read=False,final_cleanup_audited=False,
        final_actual_external_audit_executed=False,whole_workload_AR_qualified=False,
        genuine_branched_tree_target_qualified=False,NPU_initialized=False,formal_Goodput=False)
    path=D/'V103_ACTUAL_ACTIVE32_CAP3_COMPLETED_CASE_ONLY_ATTRIBUTION.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path.read_bytes()),elapsed_seconds=result['analysis_elapsed_seconds'],
        comparisons={key:{k:val for k,val in row.items() if k not in ('source','geometry','normal_owner_gain','original_early_target_wall_seconds')}
            |dict(geometry=row['geometry'],normal_owner_gain=row['normal_owner_gain']) for key,row in result['comparisons'].items()})))


if __name__=='__main__':main()
