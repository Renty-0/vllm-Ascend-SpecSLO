"""Read and independently reconstruct every actual bounded model case."""
import argparse,datetime,hashlib,importlib.util,json,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-static-q5-M120-node-math-v255';DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
def need(v,m):
    if not v:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    need('/'not in args.tag and not args.output.exists(),'fresh private actual tag and audit result')
    receipt=DONOR/('receipt-'+args.tag+'.json');r=json.loads(receipt.read_text())
    need(r['status']=='complete'and not r['errors']and r['sources_unchanged']and r['devices_after']['all_idle']and not r['owned_remaining'],'complete unchanged actual bounded process and all4 idle')
    for phase in ('pre_verifier','post_verifier'):
        v=r[phase];need(v['exit_code']==0 and v['actual_result']['status']=='pass'and v['actual_result']['checked_frozen_files']==10496 and v['actual_result']['checked_external_files']==15 and v['actual_result']['checked_packages']==212 and sha(v['path'])==v['sha256'],'actual original source/environment verification')
    original=D/'audit_v209_terminal_HOST.py';need(sha(original)=='fe8a050d3dce287787017bdf9b0f319bd2577e641e40a7baef7f2df785c0db5b','immutable HOST backend firewall')
    spec=importlib.util.spec_from_file_location('_ROOT255_HOST_firewall',original);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);sys.meta_path.insert(0,m.Firewall())
    sys.path[:0]=[str(P),str(B/'tree-static-q5-M120-full463-own-AR-v256'),str(B/'tree-static-q5-M128-full463-own-AR-v251'),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(DONOR),str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-contiguous-branch-tail-final-metadata-math-v234')]
    import guard255
    guard255.EXPECTED_FREEZE='accc1452ad666306679a4f33482655eb975e9556732b4d3dac8d9abdb7d9c2be';guard255.EXPECTED_ARCHIVE='e1199c704974857ccd2a644e7c062c0191a28d253751c25c0d0448c8aa06585e'
    _,sources,_=guard255.prepared()
    from oracle255 import cases
    from audit_static_calls256 import audit_calls
    expected=cases();need(len(expected)==227,'predeclared actual227 supported shapes')
    actual=DONOR/'actual'/args.tag;heads=[];rows=[]
    for rank in(1,2,3):
        path=actual/f'rank{rank}-numeric.json';v=json.loads(path.read_text());proof=v['math']
        need(v['rank']==rank and v['physical_M']==120 and v['physical_query_sizes']==[5]*24 and len(proof['cases'])==227,'actual model and complete case dimensions')
        need(proof['all_independent_prefixes_unchanged']is True and v['bank']['independent_neural_prefills']==72 and v['bank']['all_triplet_prefix_KV_equal']and v['bank']['all_triplet_roots_equal'],'72 independently prefilled protected prefixes')
        own_heads=[]
        for index,(case,obs)in enumerate(zip(expected,proof['cases'],strict=True),1):
            need(obs['case']==case and all(obs[k]is True for k in('exact_greedy','hidden_bitwise_equal','KV_bitwise_equal','graph_eager_KV_bitwise_equal','all128_KV_original_matcher_passed'))and obs['candidate_tokens']==obs['reference_tokens']and obs['candidate_KV_copied']is False,'every original real-node matcher and bitwise condition')
            checkpoint=json.loads((actual/'independent-node-math'/f'rank{rank}-canonical-case{index:03d}.json').read_text())
            need(checkpoint['case']==case and checkpoint['observation']==obs,'exact actual per-case checkpoint retained')
            own_heads.append(obs['candidate_tokens'])
        calls=audit_calls(expected,v['actual_static_graph_calls']);need(calls['actual_graph_dispatches']==231,'227 real graph calls plus4 original changed/restore cold calls')
        heads.append(own_heads);rows.append(dict(rank=rank,report_sha256=sha(path),cases227=True,independent_prefixes72=True,all_bitwise=True,actual_graph_audit=calls))
    need(heads[0]==heads[1]==heads[2],'all3 target greedy case predictions agree')
    need(not m.Firewall.attempts and not any(n.split('.')[0]in m.BACKENDS for n in sys.modules),'HOST-only read/reconstruction')
    result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='bounded_all227_actual_cases_independently_audited',receipt_sha256=sha(receipt),source_members=len(sources),all3_targets_all227_bitwise=True,all72_prefixes_unchanged=True,actual_case_node_evaluations_per_target=sum(c['query_count']for c in expected),by_rank=rows,whole_workload_AR_qualified=False,NPU_used=False,performance=False,adopted=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
