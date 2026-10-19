"""HOST-only all-N/mixed-eager scope and actual guard ABI negative controls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch
from probe371 import HERE, configurations
import audit371
import guard371

def main():
    output=HERE/'HOST371_RESULT.json'
    assert not output.exists()
    rows=[]
    for n,origin,mode in configurations():
        physical=5*(1<<n.bit_length())
        eager=0 if mode=='normal' else (n if mode=='all_eager' else (n+1)//2)
        frontier=5*(1<<eager.bit_length()) if eager else 0
        for phase in ('A1','B','A2'):
            rows.append(dict(N=n,origin=origin,mode=mode,phase=phase,eager_owners=eager,
                all8_tokens_exact=True,all8_FP32_probabilities_bitwise=True,mean_confidence_bitwise=True,
                original_eager_frontier_roots_exact=True,eager_promotion_map_exact=True,
                all_KV_tensors=56,original_graph_calls=4+bool(eager),candidate_graph_calls=2+bool(eager),
                original_actual_model_steps=4+bool(eager),candidate_actual_model_steps=4+bool(eager),
                physical_queries_equal=True,query_count=4*physical+frontier,
                A2_restored=True if phase=='A2' else None,
                tail=dict(graph_calls=1,model_calls=1,actual_model_steps=3,query_count=3*physical,
                    graph_padding_query_count=3*(physical-n),logical_owners=n,
                    original_first_step_preserved=True,final_leaf_KV_materialized=False)))
    costs=[dict(N=n,mode=mode,pair=pair,operation=name,seconds=.02+pair/1000000)
        for n in (4,12,24,32) for mode in ('mixed_eager','all_eager') for pair in range(20)
        for name in (('generic','first_plus_tail3') if pair%2==0 else ('first_plus_tail3','generic'))]
    summary={f'N{n}_{mode}':{name:statistics.median([r['seconds']for r in costs if r['N']==n and r['mode']==mode and r['operation']==name])
            for name in ('generic','first_plus_tail3')}for n in (4,12,24,32)for mode in ('mixed_eager','all_eager')}
    fixture=dict(status='complete',source_sha256=hashlib.sha256(b'HOST').hexdigest(),
        models_loaded=['Qwen3-0.6B'],target_model_loaded=False,TP=1,dtype='BF16',quantized=False,
        independent_prefills=96,layers=28,math_rows=rows,costs=costs,median_seconds=summary,
        all_committed_prefix_KV_protected=True,native_qualification_bodies_unchanged=True,
        all_logical_N_1_to32_exercised=True,normal_mixed_and_all_eager_exercised=True,
        graph_stats=dict(unvalidated_entries=0,disabled_entries=0),serving_installed=False,
        Raw=None,Goodput=None,adopted=False,goal_achieved=False)
    def run(value):
        with patch.object(Path,'read_text',return_value=json.dumps(value)),patch.object(Path,'read_bytes',return_value=b'HOST'):
            return audit371.audit('/HOST-only-fixture')
    actual=run(fixture)
    assert isinstance(actual,tuple)and len(actual)==3 and actual[0]['math_rows']==306
    with patch.object(audit371,'audit',return_value=actual):assert guard371.actual_audit('/HOST-only')==actual
    negatives=[]
    for label in ('missing_N','frontier_count','frontier_roots','promotion_map','probability_bits','KV',
                  'query_charge','ABA','timing_order','unvalidated','prefix','invented_Raw'):
        value=copy.deepcopy(fixture)
        if label=='missing_N':value['math_rows'].pop()
        if label=='frontier_count':value['math_rows'][1]['eager_owners']=1
        if label=='frontier_roots':value['math_rows'][0]['original_eager_frontier_roots_exact']=False
        if label=='promotion_map':value['math_rows'][0]['eager_promotion_map_exact']=False
        if label=='probability_bits':value['math_rows'][0]['all8_FP32_probabilities_bitwise']=False
        if label=='KV':value['math_rows'][0]['all_KV_tensors']=55
        if label=='query_charge':value['math_rows'][0]['query_count']-=1
        if label=='ABA':value['math_rows'][2]['A2_restored']=False
        if label=='timing_order':value['costs'][0],value['costs'][1]=value['costs'][1],value['costs'][0]
        if label=='unvalidated':value['graph_stats']['unvalidated_entries']=1
        if label=='prefix':value['all_committed_prefix_KV_protected']=False
        if label=='invented_Raw':value['Raw']=1000
        try:run(value)
        except RuntimeError:negatives.append(label)
        else:raise AssertionError('corrupt result accepted: '+label)
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST_allN_eager_inventory_and_actual_guard_ABI',math_rows=306,timing_calls=320,
        rejected_corruptions=negatives,probe_sha256=hashlib.sha256((HERE/'probe371.py').read_bytes()).hexdigest(),
        NPU_used=False,model_qualified=False,Raw=None,Goodput=None)
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
