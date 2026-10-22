"""Read-only CPU enumeration/report generator; never imported by production.

No torch, tokenizer, NPU, subprocess, archived-source edits or gate changes.
"""
import ast
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

B=Path('/root/SpecSLO/budget-calibration-20261015')
OUT=B/'budget-diagnostics/LOW_KEY_BACKUP_ENUMERATION.json'
TOKENS=(40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197)
SEGMENTS=(4,8,10,12,14,16,17,18,22,24,32,33,34,38)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lengths(n,q,m,s):
    d=s-n;e=m-q
    if d==e==0:return []
    if not 1<=d<=32 or not 2*d<=e<=9*d:return None
    return [e//d+int(i<e%d) for i in range(d)]


def legacy(n,q):
    for m in TOKENS:
        for s in SEGMENTS:
            v=lengths(n,q,m,s)
            if v is not None:return m,s,v
    raise ValueError((n,q))


def fallback(n,q):
    for m,s in (((96,32),) if n<=5 else ((96,38),(128,38),(165,38))):
        v=lengths(n,q,m,s)
        if v is not None:return m,s,v
    raise ValueError((n,q))


def budgets(n,q,f):
    result=[1]*n;remaining=q-2*n
    if f=='full9':result[0]=5;remaining-=4
    maximum=5 if f=='full9' else 4
    for i in range(n):
        extra=min(remaining,maximum-result[i]);result[i]+=extra;remaining-=extra
    assert remaining==0 and sum(result)<=96 and sum(result)+n==q
    assert min(result)>=1 and max(result)<=maximum and (f!='full9' or 5 in result)
    return result


def native_key(f,m,s):
    if f=='full9':shape=(((s,1,9,4096),(s,32)),);family='tree-fia-full'
    else:shape=(((s,32),(s,32),(2048,2048)),);family='fia'
    return repr((f'target-greedy:151936|steps:1|{family}:exact-length-task-update|buffers:{shape}',(m,)))


def decode_key(key):
    name,rows=ast.literal_eval(key)
    if not name.startswith('target-greedy:151936|steps:1|'):return None
    shape=ast.literal_eval(name.split('|buffers:',1)[1])
    f='full9' if 'tree-fia-full:' in name else 'causal'
    return rows[0],shape[0][0][0],f


def main():
    started=time.perf_counter()
    formal_path=B/'tree-backbone-budget-v17/measurement-output/runs/candidate-on-backbone-B5-PA-context-full463-r1/result.json'
    warm_path=B/'tree-bounded-target-buckets-v40/parent-preparation/V30_GEN7_GRAPH_INVENTORY.json'
    formal=json.loads(formal_path.read_text())['results'][0]
    field='aclgraph_target_replay_entry_key_histogram_delta'
    leader=next(r for r in formal['measured_worker_aclgraph_deltas'] if r['rank']==1)
    h1={decode_key(k):v for k,v in leader[field].items() if decode_key(k) is not None}
    target_metrics=[r for r in formal['measured_worker_aclgraph_deltas'] if r['rank'] in (1,2,3)]
    assert all(r[field]==leader[field] for r in target_metrics)
    warm=json.loads(warm_path.read_text());h2={decode_key(k):v for k,v in warm['graph_metrics']['aclgraph_target_replay_entry_key_histogram'].items() if decode_key(k) is not None}
    assert sum(h1.values())==2404 and sum(h2.values())==16611
    ranked1=sorted(h1,key=lambda k:(-h1[k],k));ranked2=sorted(h2,key=lambda k:(-h2[k],k))
    domain=[];preimage=defaultdict(list)
    for n in range(1,33):
        for q in range(2*n,min(6*n,n+96,128)+1):
            for f,possible in [('causal',q<=5*n),('full9',q>=2*n+4)]:
                if not possible:continue
                values=budgets(n,q,f);m,s,ls=legacy(n,q)
                row=dict(N=n,Q=q,family=f,budgets=values,legacy_key=(m,s,f),legacy_dummy_lengths=ls)
                domain.append(row);preimage[(m,s,f)].append(row)
    assert len(domain)==3046
    for hist in (h1,h2):assert not set(hist)-set(preimage)
    policies=[];chosen_mapping=None
    for order,ranking in [('B17_formal',ranked1),('V30_warmup',ranked2)]:
        for k in range(1,5):
            kept=set(ranking[:k]);mapped=[];histogram=Counter();fallback_keys=set();key_cases=defaultdict(list)
            extra_rows=[];extra_segments=[]
            for row in domain:
                n,q,f=row['N'],row['Q'],row['family'];old=row['legacy_key']
                if old in kept:m,s,ls=legacy(n,q)
                else:
                    m,s,ls=fallback(n,q);fallback_keys.add((m,s,f))
                assert s==n+len(ls) and q+sum(ls)==m and len(ls)<=32
                assert all(2<=length<=9 for length in ls)
                assert m in (96,128,165) or (m,s,f) in kept
                key=(m,s,f);histogram[key]+=1
                record=[n,q,f,m,s,len(ls),min(ls,default=0),max(ls,default=0)]
                mapped.append(record);key_cases[key].append(dict(N=n,Q=q,family=f,budgets=row['budgets'],dummy_lengths=ls))
                extra_rows.append(m-old[0]);extra_segments.append(s-old[1])
            assert len(histogram)<=12
            witness=[]
            for key,cases in sorted(key_cases.items()):
                cases.sort(key=lambda r:(len(r['dummy_lengths']),-r['N'],-r['Q']))
                first=cases[0];different=[r for r in cases[1:] if r['N']!=first['N'] or r['Q']!=first['Q']]
                second=max(different,key=lambda r:(int(r['N']!=first['N']),abs(r['N']-first['N']),abs(r['Q']-first['Q']))) if different else None
                witness.append(dict(native_key=list(key),expected_original_key=native_key(key[2],key[0],key[1]),domain_cases=len(cases),primary=first,alternate=second))
            evidence={}
            for label,hist in [('B17_formal',h1),('V30_gen7_cumulative_warmup',h2)]:
                entries=[]
                for old,count in sorted(hist.items(),key=lambda r:-r[1]):
                    candidates=preimage[old];newrows=[];newsegments=[];newkeys=set();padding=[];oldpadding=[]
                    for row in candidates:
                        n,q,f=row['N'],row['Q'],row['family']
                        m,s,ls=legacy(n,q) if old in kept else fallback(n,q)
                        newrows.append(m-old[0]);newsegments.append(s-old[1]);newkeys.add((m,s,f))
                        padding.append(m-q);oldpadding.append(old[0]-q)
                    entries.append(dict(old_key=list(old),observed_replays=count,exact_key_preserved=old in kept,
                        possible_real_N_Q_pairs=len(candidates),possible_N=[min(r['N'] for r in candidates),max(r['N'] for r in candidates)],
                        possible_Q=[min(r['Q'] for r in candidates),max(r['Q'] for r in candidates)],
                        new_possible_keys=[list(x) for x in sorted(newkeys)],additional_physical_query_rows_per_call=[min(newrows),max(newrows)],
                        additional_physical_segments_per_call=[min(newsegments),max(newsegments)],
                        original_padding_rows_per_call=[min(oldpadding),max(oldpadding)],new_padding_rows_per_call=[min(padding),max(padding)]))
                total=sum(hist.values());keptcalls=sum(v for key,v in hist.items() if key in kept)
                minextra=sum(e['observed_replays']*e['additional_physical_query_rows_per_call'][0] for e in entries)
                maxextra=sum(e['observed_replays']*e['additional_physical_query_rows_per_call'][1] for e in entries)
                oldphysical=sum(key[0]*value for key,value in hist.items())
                evidence[label]=dict(total_target_greedy_replays=total,retained_exact_native_key_replays=keptcalls,
                    retained_exact_native_key_fraction=keptcalls/total,retained_exact_native_key_percent=100*keptcalls/total,
                    old_physical_query_rows=oldphysical,additional_physical_query_rows_bounds=[minextra,maxextra],
                    additional_physical_query_rows_bounds_fraction_of_old=[minextra/oldphysical,maxextra/oldphysical],
                    additional_rows_per_observed_call_bounds=[minextra/total,maxextra/total],
                    new_physical_query_rows_bounds=[oldphysical+minextra,oldphysical+maxextra],historical_key_conditioned_bounds=entries,
                    scope='Bounds over every legal realN/Q preimage of each observed legacy native key. No per-cycle actual N/Q was recorded, so this is not an exact trace replay or a latency estimate.')
            policy=dict(selection_order=order,common_keys_requested=k,preserved_common_keys=[list(x) for x in ranking[:k]],
                total_reachable_native_keys=len(histogram),reachable_fallback_keys=[list(x) for x in sorted(fallback_keys)],
                fallback_key_count=len(fallback_keys),complete_domain_cases=len(domain),infeasible=[],
                uniform_domain_extra_physical_rows=dict(minimum=min(extra_rows),maximum=max(extra_rows),mean=sum(extra_rows)/len(extra_rows),scope='Unweighted finite geometry domain, not workload distribution'),
                uniform_domain_extra_segments=dict(minimum=min(extra_segments),maximum=max(extra_segments)),
                mapped_domain_sha256=hashlib.sha256(json.dumps(mapped,separators=(',',':')).encode()).hexdigest(),
                key_witnesses=witness,historical=evidence)
            policies.append(policy)
            if order=='B17_formal' and k==4:chosen_mapping=mapped
    result=dict(status='CPU_JSON_BACKUP_ONLY_NOT_PRODUCTION_NOT_NPU_QUALIFIED',
        priority='V46 largest-first workspace reuse is preferred; this lower-key policy is backup only.',
        input_sources={str(formal_path):digest(formal_path),str(warm_path):digest(warm_path),str(B/'tree-bounded-target-buckets-v40/target_layout.py'):digest(B/'tree-bounded-target-buckets-v40/target_layout.py')},
        histograms=dict(B17_formal=dict(field=field,target_ranks_equal=True,total=2404,keys=len(h1),top=[dict(key=list(key),count=h1[key]) for key in ranked1]),
            V30_gen7=dict(field='aclgraph_target_replay_entry_key_histogram',scope='Seven qualification warmup generations cumulative; no valid V30 formal result',total=16611,keys=len(h2),top=[dict(key=list(key),count=h2[key]) for key in ranked2])),
        admitted_domain=dict(N=[1,32],Q='2N..min(6N,N+96,128)',causal='Q<=5N, all real B1..4',full9='Q>=2N+4, at least one realB5 genuine fork',candidate_roof=96,root_inclusive_roof=128,geometries=3046),
        mapping='Compute original legacy envelope independent of runtime cache. Preserve only if its exact(M,S,family) is in the frozen common-key list; otherwise use existing smallN<=5(96,32) or first legal largeN(96/128/165,38). Never promote causal to FULL or fabricate a real query/B0.',
        fallback_base_without_common_keys=dict(count=8,keys=[list(x) for x in sorted({(*fallback(r['N'],r['Q'])[:2],r['family']) for r in domain})],
            why_not_always7='With no historical preservation, both families of all4 physical pairs are reachable. CausalN18/Q89 needsM165/S38 because20dummies require>=40 extra rows. Once FULL40/S8 is preserved, every smallN<=5 FULL legacy case stays there and FULL96/S32 is unreachable: fallback then drops to7.'),
        dummy_contract=dict(max_count=32,min_queries=2,max_queries=9,independent_sequence_pages=True,width1_linear=True,prefix0=True,real_queries_unchanged=True,original_native_family=True),
        policies=policies,chosen_top4_B17_mapping=chosen_mapping,
        memory_limits=dict(driver_free_GiB=6,KV_blocks=544,max_entries=512,unchanged=True,not_qualified=True),
        memory_warning='Key count is only a planning proxy. Persistent graph pools, workspaces/capture order, graph private buffers, model weights and caches also consume memory. Root identified monotonically growing FULL workspace ownership in V44 small-to-large order (~322MiB per retained owner); V46 largest-first reuse is the separate preferred fix. No promise11keys eliminates17.5GiB growth or passes6GiB gate.',
        no_performance_or_AR_claim=True,no_source_modified=True,NPU=False,torch_or_tokenizer=False,CPU_elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),bytes=OUT.stat().st_size,elapsed=result['CPU_elapsed_seconds'],policies=[dict(order=p['selection_order'],top=p['common_keys_requested'],keys=p['total_reachable_native_keys'],fallback=p['fallback_key_count'],formal_retained=p['historical']['B17_formal']['retained_exact_native_key_percent'],V30_retained=p['historical']['V30_gen7_cumulative_warmup']['retained_exact_native_key_percent'],formal_extra_queries=p['historical']['B17_formal']['additional_physical_query_rows_bounds']) for p in policies])))


if __name__=='__main__':main()
