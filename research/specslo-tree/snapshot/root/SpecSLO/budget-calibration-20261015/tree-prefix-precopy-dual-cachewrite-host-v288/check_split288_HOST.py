"""Compare full write-then-copy with prefix-precopy plus dual current writes."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-dynamic-root-alternate-eager-compat-full-v287'),str(B/'tree-deferred-FULL-mask-host-projection-v259')]
import check287

def main():
    from fixture259 import build
    env=build();e=env['engine'];native=env['native']
    from padding269 import pad
    from plan269 import make
    from layout269 import layout
    from static_state287 import bind_checked_prepare
    import branch_tail255
    bind_checked_prepare(e._branch_tail_state,branch_tail255)
    from split288 import split
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    cases=[]
    for prefix in (0,1,123,124,125,126,127,128,1505,2555):
        for budgets in ([5],[6],[7],[8],[8]*12,[8]*8+[4]*8,[5]*4+[4]*16):
            n=len(budgets);prefixes=([prefix]+[127]*(n-1))
            plans=[make(b,p,0)for b,p in zip(budgets,prefixes,strict=True)]
            ids=list(range(n));roots=list(range(31,31+n));draft=[[101+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)]
            pp,rr,dd,ss,real=pad(e,plans,roots,draft,ids,[],[])
            for owner,plan in zip(ids,plans,strict=True):
                length=plan.prefix_len+plan.candidate_budget+1
                e._prepare_attention_metadata([owner]*length,list(range(length)),True)
            owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
            positions=[v for plan in pp for v in plan.cache_positions.tolist()]
            e._prepare_attention_metadata(owners,positions,True)
            native.NativePearlEngine.target_tree_forward(e,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
            state=e._branch_tail_state;meta=state.static_current['physical'][2]
            desc=layout(pp,ss,e.cache_allocation.block_tables,e._tree_graph_scratch_sequence_ids)
            slots=meta.slot_mapping.tolist();new=split(desc['source'],desc['dest'],slots)
            assert new['prefix_copies']+new['current_aliases']==new['original_copies']
            assert new['current_aliases']==sum(1+b-4 for b in budgets if b>4)
            touched=set(desc['source']+desc['dest']+slots)
            for layer in (0,1,31,63):
                for kv in (0,1):
                    # All payloads are distinct 16-bit words; no BF16 arithmetic.
                    before={slot:(slot*127+layer*193+kv*8191)&65535 for slot in touched}
                    words=[((i*4099+layer*521+kv*179+prefix)&65535)for i in range(120)]
                    old=dict(before)
                    for slot,word in zip(slots,words):old[slot]=word
                    written=dict(old)
                    for src,dst in zip(desc['source'],desc['dest']):old[dst]=written[src]
                    actual=dict(before)
                    for src,dst in zip(new['prefix_source'],new['prefix_destination']):actual[dst]=before[src]
                    for row,(slot,word)in enumerate(zip(slots,words)):
                        actual[slot]=word
                        alias=new['query_alias_destination'][row]
                        if alias>=0:actual[alias]=word
                    assert actual==old,(prefix,budgets,layer,kv)
            cases.append(dict(prefix=prefix,budgets=budgets,prefix_copies=new['prefix_copies'],
                current_aliases=new['current_aliases'],original_copies=new['original_copies']))
    value=dict(status='HOST_actual_Native_physical120_split_copy_and_cache_write_equivalence_passed',
        cases=cases,case_count=len(cases),independent_layer_KV_payload_comparisons=len(cases)*8,
        page_crossings_checked=True,all_touched_cache_words_equal=True,old_query_order_and_slots_preserved=True,
        hypothesis='one all-layer committed-prefix copy before model; per-layer current cache dual write replaces original cache writer',
        Triton_operator_implemented=False,NPU_used=False,model_math_qualified=False,runtime_installed=False,Goodput=None)
    with(P/'HOST_SPLIT288_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items()if k!='cases'}))

if __name__=='__main__':main()
