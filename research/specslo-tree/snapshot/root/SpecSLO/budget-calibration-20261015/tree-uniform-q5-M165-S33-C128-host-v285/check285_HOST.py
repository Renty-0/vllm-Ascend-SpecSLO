"""Original CPU Native protocol at prospective N32/C128/F12/M165/S33."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-alternate-chain-target-host-v269'),str(B/'tree-fused-branch-KV-row-copy-v262'),str(B/'tree-static-q5-M120-node-math-v255'),str(B/'tree-static-q5-M128-projection-host-v249'),str(B/'tree-static-q5-immutable-index-cache-host-v247'),str(B/'tree-static-q5-long-prefix-origins-node-math-v246'),str(B/'tree-static-q5-projection-host-v243'),str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-contiguous-branch-tail-final-metadata-math-v234'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-high-normal-cap4-native-families-v148')]

def main():
    from fixture285_HOST import build
    env=build();e=env['engine'];native=env['native'];torch=env['torch']
    from padding285 import pad
    from plan285 import make
    from contract262 import checked
    from layout285 import layout
    results=[];seen=set()
    for n in range(1,33):
        for raw in([1]*n,[3]*n,[4]*n,[5]*n,[6]*n,[7]*n,[8]*n,[(i%8)+1 for i in range(n)]):
            budgets=list(raw)
            while sum(b>4 for b in budgets)>12 or n+sum(b>4 for b in budgets)>33:
                i=next(i for i in reversed(range(n))if budgets[i]>4);budgets[i]=4
            while sum(budgets)>128:
                i=next(i for i in reversed(range(n))if budgets[i]>1);budgets[i]-=1
            for mode in('zero','boundary127','boundary128','two-long'):
                key=(tuple(budgets),mode)
                if key in seen:continue
                seen.add(key)
                ps=([0]*n if mode=='zero'else[127]*n if mode=='boundary127'else[128]*n if mode=='boundary128'else([1505,2555]+[127]*n)[:n])
                plans=[make(b,p,0)for b,p in zip(budgets,ps,strict=True)]
                rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)];roots=list(range(41,41+n));ids=list(range(n))
                pp,rr,dd,ss,real=pad(e,plans,roots,rows,ids,[],[])
                # Match an actual prefilled/activated lease: synchronize the
                # allocated device table before the private HOST metadata view.
                # A synthetic long-prefix fixture must allocate the WHOLE
                # prefix, not merely its sparse current query positions.
                for owner,plan in zip(ids,plans,strict=True):
                    length=plan.prefix_len+plan.candidate_budget+1
                    e._prepare_attention_metadata([owner]*length,list(range(length)),True)
                    assert all(v>=0 for v in e.cache_allocation.block_tables[owner][:(length+127)//128])
                all_owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
                all_positions=[v for plan in pp for v in plan.cache_positions.tolist()]
                e._prepare_attention_metadata(all_owners,all_positions,True)
                out=native.NativePearlEngine.target_tree_forward(e,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
                q=n+sum(budgets);state=e._branch_tail_state;meta=state.static_current['physical'][2]
                assert out['query_count']==165 and out['logical_query_count']==q and out['num_draft_tokens']==budgets
                assert out['target_query_token_ids'].tolist()==[v for r,row in zip(roots,rows,strict=True)for v in [r]+row]
                assert tuple(meta.actual_seq_lengths_q)==tuple(range(5,166,5))and meta.sequence_lens==(2560,)*33
                assert tuple(meta.tree_attention_mask.shape)==(33,1,11 if any(b>4 for b in budgets)else 10,4096)
                assert out['cache_slot_mapping'].numel()==out['cache_slot_mapping'].unique().numel()==q
                moved={}
                if any(b>4 for b in budgets):
                    desc=layout(pp,ss,e.cache_allocation.block_tables,e._tree_graph_scratch_sequence_ids)
                    checked(desc['source'],desc['dest'],rows=544*128)
                    moved=dict(zip(desc['dest'],desc['source'],strict=True))
                # Independently resolve every actual visible key through the
                # physical output mask/table and compare the original tree.
                physical=state.static_current['physical'];mapping=physical[4];tables=meta.request_block_tables.tolist();offset=0
                for owner,plan in zip(ids,plans,strict=True):
                    canonical=e.cache_allocation.block_tables[owner]
                    for j in range(plan.candidate_budget+1):
                        row=mapping['original_to_physical'][offset+j];group,query=divmod(row,5)
                        old_visible=(~plan.attention_mask[j]).nonzero().flatten().tolist()
                        new_visible=(~meta.tree_attention_mask[group,0,query]).nonzero().flatten().tolist()
                        expected=[canonical[v//128]*128+v%128 for v in old_visible]
                        actual=[tables[group][v//128]*128+v%128 for v in new_visible]
                        reconstructed=[moved.get(v,v)for v in actual]
                        if reconstructed!=expected:
                            raise AssertionError(dict(N=n,budgets=budgets,mode=mode,owner=owner,node=j,prefix=plan.prefix_len,group=group,query=query,old_visible_head=old_visible[:8],old_visible_tail=old_visible[-8:],new_visible_head=new_visible[:8],new_visible_tail=new_visible[-8:],expected_head=expected[:8],expected_tail=expected[-8:],actual_head=reconstructed[:8],actual_tail=reconstructed[-8:],old_count=len(expected),new_count=len(reconstructed),first_differences=[(k,a,b)for k,(a,b)in enumerate(zip(reconstructed,expected))if a!=b][:8]))
                    offset+=plan.candidate_budget+1
                results.append(dict(N=n,budgets=budgets,mode=mode,real_queries=q,groups=n+sum(b>4 for b in budgets)))
    assert any(r['N']==32 and sum(r['budgets'])==128 for r in results)
    assert 'torch_npu'not in sys.modules
    value=dict(status='HOST_actual_Native_prospective165_geometry_passed',cases=len(results),rows=results,all_real_token_order_and_KV_slots=True,fixed165_S33_uniform_q5_KV2560=True,raw_copy1792_contract=True,explicit_F12=True,all_real_visible_KV_ancestry_independently_resolved=True,model='CPU arithmetic protocol double only; no actual attention or NPU graph',NPU_used=False,neural_qualified=False,runtime_installed=False,Goodput=None,adopted=False)
    with(P/'HOST285_ANCESTRY_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items()if k!='rows'}))
if __name__=='__main__':main()
