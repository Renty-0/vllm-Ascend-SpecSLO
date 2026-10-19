"""Synthetic protocol cases only; no NN output or performance is synthesized."""
import copy
import json
from pathlib import Path
import sys

P=Path(__file__).parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-static-q5-M120-node-math-v255'),
    str(B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'),
    str(B/'tree-canonical-contiguous-full463-own-AR-v235')]


def main():
    from oracle377 import cases
    from physical377 import plan
    from bucket377 import extent
    from dispatch377 import audit_calls
    expected=cases();prefixes=[[0]*n for n in [1505,2555]+[127]*22]
    calls=[];seen=set()
    for case in expected:
        sizes=[]
        for b in case['budgets']:sizes.extend([5,1]if b==5 else[b+1])
        sizes.extend(case['padding']);ends=[];cursor=0
        for q in sizes:cursor+=q;ends.append(cursor)
        m=case['M'];s=m//5;branch=5 in case['budgets'];capacity=11 if branch else 10
        physical=plan(ends,case['query_count'],m)
        visible=[len(prefixes[i])+min(b,4)+1 for i,b in enumerate(case['budgets'])]
        selected=extent(tuple(visible));family=(m,capacity)
        row=dict(N=case['N'],real_queries=case['query_count'],M=m,attention_segments=s,
            actual_seq_lengths_q=list(range(5,m+1,5)),sequence_lens=[selected]*s,
            mask_shape=[s,1,capacity,4096],copy_enabled=branch,used_aclgraph=True,fallback_reason=None,
            physical_to_original=physical['physical_to_original'],original_to_physical=physical['original_to_physical'],
            real_moves=physical['real_moves'],primary_side_groups=physical['supported_primary_side_groups'],
            adaptive377_enabled=True,uniform_KV_bucket377=dict(enabled=True,extent=selected,original_sequence_lens=visible))
        modes=['capture_replay','replay','replay']if family not in seen else['replay']
        calls.extend(dict(copy.deepcopy(row),execution_mode=mode)for mode in modes)
        seen.add(family)
    result=audit_calls(expected,calls,prefixes)
    assert result['positive_node_cases']==231 and len(calls)==239
    def rejects(value):
        try:audit_calls(expected,value,prefixes)
        except(RuntimeError,ValueError,KeyError):return
        raise RuntimeError('ROOT377 HOST invalid dispatch accepted')
    bad=copy.deepcopy(calls);bad[0]['M']=120;rejects(bad)
    bad=copy.deepcopy(calls);bad.pop();rejects(bad)
    bad=copy.deepcopy(calls);bad[0]['physical_to_original'][0]=80;rejects(bad)
    bad=copy.deepcopy(calls);bad[0]['sequence_lens']=[512]*16;bad[0]['uniform_KV_bucket377'].update(extent=512,original_sequence_lens=[1]);rejects(bad)
    bad=copy.deepcopy(calls);bad[0]['used_aclgraph']=False;rejects(bad)
    warm=[];cursor=0;seen=set()
    for case in expected:
        family=(case['M'],11 if 5 in case['budgets']else 10);count=3 if family not in seen else 1
        warm.append(calls[cursor+count-1]);cursor+=count;seen.add(family)
    assert audit_calls(expected,warm,prefixes)['actual_graph_dispatches']==231
    assert 'torch'not in sys.modules
    value=dict(status='PASS_HOST377_ALL231_DISPATCH_COLD_REUSE_AND_CORRUPTIONS',NPU_used=False,
        synthetic_protocol_only=True,neural_qualified=False,cases=231,first_cold_calls=239,warm_reuse_calls=231,
        corruptions=5,Raw=None,Goodput=None)
    (P/'HOST377_DISPATCH_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
