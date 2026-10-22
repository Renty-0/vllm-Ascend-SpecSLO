"""Independently reopen frozen408-case target math and all1155 cost calls."""
import hashlib,json,math,statistics,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-alternate-chain-target-node-cost-v272'
TAG='alternate-chain-root-v272-B1to8-408-node-math-paired-cost-v1'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';A=DONOR/'actual'/TAG
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    receipt=read(DONOR/f'receipt-{TAG}.json');pins=read(P/'SOURCE272_RECEIPT.json')
    assert receipt['status']=='complete'and receipt['errors']==[]and receipt['owned_remaining']==[]
    assert receipt['sources_unchanged']is True and receipt['devices_after']['all_idle']is True
    for k in('pre_verifier','post_verifier'):
        v=receipt[k];assert v['exit_code']==0 and sha(v['path'])==v['sha256']
        v=v['actual_result'];assert v['status']=='pass'and v['errors']==[]
        assert(v['checked_frozen_files'],v['checked_external_files'],v['checked_packages'])==(10496,15,212)
    assert sha(P/'FREEZE272.json')==pins['freeze_sha256']and sha(P/'SOURCE272.tar.gz')==pins['archive_sha256']
    inputs=read(P/'FREEZE272.json')['files']
    for p,h in inputs.items():assert sha(p)==h,p
    archived={**inputs,str(P/'FREEZE272.json'):pins['freeze_sha256']}
    with tarfile.open(P/'SOURCE272.tar.gz','r:gz')as ar:
        assert len(ar.getmembers())==len(archived)==pins['members']
        for m in ar:assert m.isfile()and hashlib.sha256(ar.extractfile(m).read()).hexdigest()==archived['/'+m.name]
    sys.path[:0]=[str(P),str(B/'tree-static-q5-M120-node-math-v255'),str(B/'tree-static-q5-M128-projection-host-v249')]
    from audit_static_calls272 import audit_calls
    declared=read(P/'DECLARED272_CASES.json')['cases'];assert len(declared)==408
    configs=[(n,k)for n in(4,8,12,16,20,23)for k in sorted({1,min(n,24-n,(96-4*n)//4)})]
    cost_cases=[]
    def shape(n,k,b):
        budgets=[b]*k+[4]*(n-k);q=n+sum(budgets);left=120-q;padding=[]
        while left:padding.append(min(5,left));left-=min(5,left)
        assert sum(budgets)<=96 and n+(k if b>4 else 0)<=24
        return dict(N=n,budgets=budgets,query_count=q,padding=padding)
    for n,k in configs:
        cost_cases.extend(shape(n,k,b)for b in range(4,9))
        for iteration in range(20):
            order=list(range(4,9));order=order[iteration%5:]+order[:iteration%5]
            cost_cases.extend(shape(n,k,b)for b in order)
    assert len(configs)==11 and len(cost_cases)==1155
    reports=[];timings=[]
    for rank in(1,2,3):
        report=read(A/f'rank{rank}-numeric.json');math_report=report['math'];cost=report['paired_target_cost']
        assert report['model_dtype']=='torch.bfloat16'and report['adopted']is False
        assert math_report['all_independent_prefixes_unchanged']is True and report['all_prefixes_still_exact_after_costs']is True
        assert [r['case']for r in math_report['cases']]==declared
        raw_files=list((A/'independent-node-math').glob(f'rank{rank}-canonical-case[0-9]*.json'))
        assert len(raw_files)==408
        raw={}
        for f in raw_files:
            value=read(f);assert value['rank']==rank and value['neural_math_on_rank']is True
            raw[value['case']['name']]=value['observation']
        assert len(raw)==408
        for row in math_report['cases']:
            assert raw[row['case']['name']]==row
            assert all(row[k]is True for k in('exact_greedy','hidden_bitwise_equal','KV_bitwise_equal','all128_KV_original_matcher_passed','graph_eager_KV_bitwise_equal'))
            assert row['candidate_KV_copied']is False and row['candidate_tokens']==row['reference_tokens']
            assert all(v['candidate_KV_copied']is False and v['reference_eager_only']is True for v in row['independent_reference_frames'])
        bank=report['bank'];assert bank['independent_neural_prefills']==72 and len(bank['all_prefix_evidence'])==72
        assert bank['candidate_KV_copied']is False and bank['all_triplet_roots_equal']is True and bank['all_triplet_prefix_KV_equal']is True
        assert len(report['actual_static_graph_calls'])==412 and len(report['actual_benchmark_static_graph_calls'])==1155
        dispatch=audit_calls(declared+cost_cases,report['actual_static_graph_calls']+report['actual_benchmark_static_graph_calls'])
        assert dispatch['actual_graph_dispatches']==1567 and dispatch['original_changed_input_restore_extra_dispatches']==4
        assert cost['no_new_capture_or_validation_in_costs']is True
        for key in('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks'):
            assert cost['graph_counters_before'][key]==cost['graph_counters_after'][key]
        assert [(r['N'],r['branch_owners'])for r in cost['rows']]==configs
        for row in cost['rows']:
            assert row['rounds']==len(row['pairs'])==20 and row['same_original_committed_prefix']is True and row['original_greedy_output_stable']is True
            for iteration,pair in enumerate(row['pairs']):
                expected=list(range(4,9));expected=expected[iteration%5:]+expected[:iteration%5]
                assert pair['order']==expected and set(pair['milliseconds'])==set(map(str,range(4,9)))
                assert all(math.isfinite(v)and v>0 for v in pair['milliseconds'].values())
            med={str(b):statistics.median(v['milliseconds'][str(b)]for v in row['pairs'])for b in range(4,9)}
            timings.append(dict(rank=rank,N=row['N'],branch_owners=row['branch_owners'],median_ms=med,
                median_paired_B8_minus_B4_ms=statistics.median(v['milliseconds']['8']-v['milliseconds']['4']for v in row['pairs']),
                median_paired_B8_minus_B5_ms=statistics.median(v['milliseconds']['8']-v['milliseconds']['5']for v in row['pairs'])))
        assert cost['real_draft_generation']is False and cost['empirical_g_measured']is False and cost['Goodput']is None
        reports.append(dict(rank=rank,path=str(A/f'rank{rank}-numeric.json'),sha256=sha(A/f'rank{rank}-numeric.json'),raw_cases408=True,dispatch_audit=dispatch))
    result=dict(status='PASS_independent_bounded_target408_and_paired_cost_audit',
        tag=TAG,receipt=str(DONOR/f'receipt-{TAG}.json'),receipt_sha256=sha(DONOR/f'receipt-{TAG}.json'),
        frozen_source_and_archive_verified=True,original_environment_pre_post_verified=True,
        owned_cleanup_complete=True,all4_idle_at_terminal=True,reports=reports,timings=timings,
        timing_scope='synchronized whole-target pad/metadata/model/sync; excludes output D2Hcomparison and stage consensus',
        all408_hidden_all128KV_greedy_bitwise_equal_all3targets=True,
        no_new_graph_capture_validation_or_fallback_during_costs=True,
        real_draft_generation=False,empirical_g_measured=False,whole463_AR=False,
        formal_Goodput=None,adopted=False)
    out=D/'ROOT272_INDEPENDENT_TARGET408_AND_PAIRED_COST_AUDIT.json'
    with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],reports=len(reports),timing_rows=len(timings),path=str(out))))
if __name__=='__main__':main()
