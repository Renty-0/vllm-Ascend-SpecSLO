"""HOST-only optimized-baseline scope, auditor corruptions and actual guard ABI."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from unittest.mock import patch
from probe374 import HERE,configurations,need
import audit374
import guard374

def main():
    output=HERE/'HOST374_RESULT.json';assert not output.exists()
    assert len(configurations())==170
    try:need(False,'HOST-reject')
    except RuntimeError:pass
    else:raise AssertionError('missing active source gate')
    rows=[]
    for n,origin,budget in configurations():
        physical=audit374.physical_bucket(n);steps=min(budget,4)
        for phase in ('A1','B','A2'):
            rows.append(dict(N=n,origin=origin,budget=budget,phase=phase,chain_steps=steps,
                original_graph_submissions=2,fused_graph_submissions=1,all_tokens_and_roots_exact=True,
                all_original_confidence_bits_exact=True,canonical_ledger_equal=True,physical_queries_equal=True,
                all_KV_tensors=56,A2_restored=True if phase=='A2'else None,
                actual=dict(logical=n,physical=physical,chain_steps=steps,model_steps=steps+1,
                    graph_submissions=1,actual_physical_queries=(steps+1)*physical,
                    frontier_CPU_readback_eliminated=True,original_weight_precision_unchanged=True)))
    costs=[dict(N=n,budget=budget,pair=pair,operation=name,seconds=.02+pair/1000000)
        for n in (4,12,24,32)for budget in (4,5)for pair in range(20)
        for name in (('optimized_existing','fused_frontier_chain')if pair%2==0 else('fused_frontier_chain','optimized_existing'))]
    summary={f'N{n}_B{budget}':{name:statistics.median([r['seconds']for r in costs if r['N']==n and r['budget']==budget and r['operation']==name])
        for name in ('optimized_existing','fused_frontier_chain')}for n in (4,12,24,32)for budget in (4,5)}
    fixture=dict(status='complete',source_sha256=hashlib.sha256(b'HOST').hexdigest(),models_loaded=['Qwen3-0.6B'],
        target_model_loaded=False,TP=1,dtype='BF16',quantized=False,model_weights_or_precision_changed=False,
        comparison_baseline='actual installed RootSeed over FastV5, not bare Native',
        original_confidence_encoding_retained=True,independent_prefills=96,layers=28,math_rows=rows,costs=costs,
        median_seconds=summary,all_committed_prefix_KV_protected=True,native_qualification_bodies_unchanged=True,
        graph_stats=dict(unvalidated_entries=0,disabled_entries=0),serving_installed=False,Raw=None,Goodput=None,adopted=False)
    def run(value):
        with patch.object(Path,'read_text',return_value=json.dumps(value)),patch.object(Path,'read_bytes',return_value=b'HOST'):
            return audit374.audit('/HOST-only')
    actual=run(fixture);assert isinstance(actual,tuple)and len(actual)==3 and actual[0]['math_rows']==510
    with patch.object(audit374,'audit',return_value=actual):assert guard374.actual_audit('/HOST-only')==actual
    rejected=[]
    for label in ('missing_case','bare_native_baseline','confidence','ledger','KV','wrong_work_charge','timing_order','unvalidated','invented_Raw'):
        v=copy.deepcopy(fixture)
        if label=='missing_case':v['math_rows'].pop()
        if label=='bare_native_baseline':v['comparison_baseline']='bare Native'
        if label=='confidence':v['math_rows'][0]['all_original_confidence_bits_exact']=False
        if label=='ledger':v['math_rows'][0]['canonical_ledger_equal']=False
        if label=='KV':v['math_rows'][0]['all_KV_tensors']=55
        if label=='wrong_work_charge':v['math_rows'][0]['actual']['actual_physical_queries']-=1
        if label=='timing_order':v['costs'][0],v['costs'][1]=v['costs'][1],v['costs'][0]
        if label=='unvalidated':v['graph_stats']['unvalidated_entries']=1
        if label=='invented_Raw':v['Raw']=1000
        try:run(v)
        except RuntimeError:rejected.append(label)
        else:raise AssertionError('invalid evidence accepted: '+label)
    source=(HERE/'probe374.py').read_text();tree=ast.parse(source)
    functions={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert {'need','configurations','main'}<=functions
    assert 'root_seed.install(native)'in source and 'parallel_steps.install(native_graph)'in source
    assert 'tree_target_replay.install(native_graph)'in source
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    subprocess.run(['bash','-n',str(HERE/'startup374.sh')],check=True)
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST_actual_optimized_stack_source_inventory_auditor_and_guard_ABI',
        math_rows=510,timing_calls=320,rejected_corruptions=rejected,
        actual_fast_root_and_replay_installers_present=True,
        probe_sha256=hashlib.sha256((HERE/'probe374.py').read_bytes()).hexdigest(),
        NPU_used=False,model_qualified=False,Raw=None,Goodput=None)
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
