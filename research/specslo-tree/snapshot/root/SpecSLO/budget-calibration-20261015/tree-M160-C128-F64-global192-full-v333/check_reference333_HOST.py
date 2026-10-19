"""Own-history frame/metadata and independent retained-mask auditor, CPU only."""
import hashlib
import json
from pathlib import Path
import sys
import types

P=Path(__file__).resolve().parent
import runtime333
import check_base326_HOST


def main():
    from fixture326_HOST import build
    env=build();engine=env['engine'];torch=env['torch']
    # The historical CPU fixture copied methods from a bootstrap owner.
    # Rebind the identical Native code to the active owner before new leases.
    for name in ('_allocate_cache','_activate_cache_sequence','_ensure_cache_capacity',
                 '_release_cache','_cache_slot_mapping','_prepare_attention_metadata'):
        method=getattr(engine,name)
        setattr(engine,name,types.MethodType(method.__func__,engine))
    from reference333 import frame
    from metadata333 import pack_q5
    from audit_reference333 import expected_static_mask_sha256,expected_allocated_table
    cases=[]
    for count in(1,8,16):
        for prompt_length in(127,128,1505,2301):
            for output_length in(1,128,255):
                prompts=[[17+i]*prompt_length for i in range(count)]
                outputs=[[41+i]*output_length for i in range(count)]
                # Each independent history boundary starts with its own lease;
                # reusing a later/longer history would retain future pages.
                engine._release_cache()
                engine._allocate_cache(prompts,enable_prefix_caching=False,reserve_sequence_capacity=True)
                for owner in range(463,527):
                    engine._activate_cache_sequence(owner,[0],enable_prefix_caching=False)
                variants=[]
                for future in(0,73):
                    data=frame(prompts,outputs,list(range(count)),future=future)
                    assert len(data['tokens'])==len(data['ids'])==len(data['positions'])==160
                    assert data['actual_seq_lengths_q']==list(range(5,161,5))
                    for owner in range(count):
                        length=prompt_length+output_length+4
                        engine._prepare_attention_metadata([owner]*length,list(range(length)),True)
                    engine._prepare_attention_metadata(data['ids'],data['positions'],True)
                    try:
                        x,pos,meta=pack_q5(engine,dict(data,selected=data['ids'][:160:5]),full=True)
                    except RuntimeError:
                        from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
                        bad=[]
                        for owner,frontier in zip(data['ids'][::5],data['positions'][::5],strict=True):
                            plan=cached_cpu_tree_speculation_plan(1,4,frontier,4096,candidate_budget=4)
                            for pos in plan.cache_positions.tolist():
                                if engine.cache_allocation.block_tables[owner][pos//128]<0:
                                    bad.append(dict(owner=owner,frontier=frontier,physical=pos,table=engine.cache_allocation.block_tables[owner][:22]))
                        print(json.dumps(dict(case=[count,prompt_length,output_length,future],unallocated=bad)))
                        raise
                    assert pos.tolist()==data['positions'] and x.tolist()==data['tokens']
                    assert meta.actual_seq_lengths_q==tuple(range(5,161,5))and meta.sequence_lens==(2560,)*32
                    assert tuple(meta.tree_attention_mask.shape)==(32,1,10,4096)
                    digest=hashlib.sha256(meta.tree_attention_mask.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    assert digest==expected_static_mask_sha256(data['positions'])
                    tables=meta.request_block_tables.tolist();original=engine.cache_allocation.block_tables
                    zero=engine._branch_tail_state.static_zero_page
                    for group,owner in enumerate(data['ids'][::5]):
                        assert tables[group]==expected_allocated_table(original[owner],data['positions'][5*group],zero,5)
                    variants.append(digest)
                assert variants[0]==variants[1]
                cases.append(dict(real_owners=count,prompt_length=prompt_length,own_output_length=output_length,
                                  max_root_position=prompt_length+output_length-1,future_inputs_do_not_change_visibility=True))
    assert len(cases)==36 and 'torch_npu'not in sys.modules
    report=dict(status='HOST_own_history_M160_S32_frame_and_independent_mask_audit_passed',
        history_cases=36,metadata_cases=72,all160_input_positions_and_tokens_exact=True,
        all32_full_tables_independently_recomputed=True,all_complete_masks_independently_recomputed=True,
        full463_own_AR_qualified=False,NPU_used=False,model_constructed=False,Goodput=None,cases=cases)
    with(P/'HOST_REFERENCE333_METADATA_RESULT.json').open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items()if k!='cases'}))

if __name__=='__main__':main()
