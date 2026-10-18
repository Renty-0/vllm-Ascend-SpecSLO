"""All admitted native chain, grouped hidden PA and frontier PA families."""
import json
from pathlib import Path
from finite_inventory_plan import CHAIN_BUCKETS, FRONTIER_BUCKETS, PA_BUCKETS, draft_keys


def grouped_lengths(rows):
    if rows not in PA_BUCKETS:
        raise ValueError('Unplanned PA bucket')
    return tuple([5]*(rows//5)+([rows%5] if rows%5 else []))


def qualify(engine):
    import torch
    import torch.distributed as dist
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from inventory_memory import checkpoint
    from root_seed import prepare
    if engine.graph_runner.graph_cache_sealed:
        raise RuntimeError('Explicit draft inventory must precede native sealing')
    group=engine.groups.verification_coordination_group
    if group is None or dist.get_backend(group)!='gloo':
        raise RuntimeError('Draft inventory requires full CPU coordination')
    runner=engine.graph_runner
    fixtures=[('chain',rows,steps) for rows in CHAIN_BUCKETS for steps in range(1,5)]
    fixtures += [('hidden_PA',rows,1) for rows in PA_BUCKETS]
    fixtures += [('frontier_PA',rows,1) for rows in FRONTIER_BUCKETS]
    if engine.is_draft:
        if engine.config.max_num_seqs!=64 or int(engine.config.spec_rhythm_tree_depth)!=4:
            raise RuntimeError('Draft explicit inventory is qualified for max64/depth4 only')
        for seq in range(64):
            prompt=[int((7+seq*13+j*19)%engine.draft_vocab_size) for j in range(127)]
            engine._activate_cache_sequence(seq,prompt,enable_prefix_caching=False)
            engine._run_packed_hidden(prompt,[seq]*127,list(range(127)),False,None,False)
    checkpoint(engine,'draft',0,len(fixtures))
    observed={kind:[] for kind in draft_keys()}
    real_prefix_restored = False
    for number,(kind,rows,steps) in enumerate(fixtures,1):
        if engine.is_draft:
            if kind != 'chain' and not real_prefix_restored:
                # The zero-prefix chain fixtures deliberately overwrote low
                # KV positions. Restore the whole real known prefix before
                # grouped suffix/root fixtures depend on positions <123.
                for seq in range(64):
                    prompt=[int((7+seq*13+j*19)%engine.draft_vocab_size) for j in range(127)]
                    engine._run_packed_hidden(prompt,[seq]*127,list(range(127)),False,None,False)
                real_prefix_restored = True
            if kind=='chain':
                # Every physical row is real. This legal heterogeneous B vector
                # attains maxGamma without all-B4 max64 violating candidate96.
                values=[steps]+[1]*(rows-1)
                if sum(values)>96:
                    raise RuntimeError('Draft witness violates real candidate roof')
                plans=[cached_cpu_tree_speculation_plan(2,4,0,4096,candidate_budget=b)
                       for b in values]
                for token in (0,1,0):
                    engine.draft_tree_forward(plans,[token]*rows,list(range(rows)))
                key=(f'draft-greedy:{engine.draft_vocab_size}|steps:{steps}|paged|confidence',rows)
                entry=runner.draft_entries.get(key)
                actual=runner.last_draft_entry_key
                if entry is None or entry.validated_real_row_count<rows:
                    raise RuntimeError('Draft logical rows failed original native qualification')
            else:
                if kind=='hidden_PA':
                    entries=[]
                    for seq,length in enumerate(grouped_lengths(rows)):
                        prefix=127; start=prefix-length+1
                        suffix=[int((7+seq*13+j*19)%engine.draft_vocab_size)
                                for j in range(start,prefix)]
                        entries.append((seq,prefix,start,suffix,int(41+seq)))
                    _,inputs,positions,metadata,physical=prepare(engine,entries,native)
                    if physical!=rows:
                        raise RuntimeError('Grouped real-query witness changed PA bucket')
                    for delta in (0,1,0):
                        runner.run_tree_hidden((inputs+delta)%engine.draft_vocab_size,positions,metadata)
                    key=('target-hidden:0|steps:1|paged',(rows,))
                else:
                    inputs=torch.zeros((rows,),dtype=torch.long,device=engine.device)
                    positions,metadata=engine._prepare_attention_metadata(list(range(rows)),[127]*rows,False)
                    for token in (0,1,0):
                        runner.run_target_greedy([inputs+token],[positions],[metadata],engine.draft_vocab_size)
                    key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|paged',(rows,))
                entry=runner.target_entries.get(key)
                actual=runner._dynamic_verification_last_key
            if (actual!=key or entry is None or not entry.runtime_validated
                    or key in runner.disabled_entry_keys):
                raise RuntimeError(f'Draft native family failed original qualification: {key!r}')
            observed[kind].append(repr(key))
        checkpoint(engine,'draft',number,len(fixtures),detail=dict(kind=kind,rows=rows,steps=steps))
    if engine.is_draft:
        for kind,keys in draft_keys().items():
            if set(observed[kind])!=set(map(repr,keys)):
                raise RuntimeError('Explicit draft native family inventory is incomplete: '+kind)
        engine._finite_draft_inventory_complete=True
        path=Path(__file__).resolve().parent/'startup-progress'/f'finite-draft-rank{engine.rank}.json'
        path.write_text(json.dumps(dict(status='complete',native_keys=observed,
            explicit_keys=86,existing_native_generic_keys_retained=28,
            original_native_flags_unchanged=True),indent=2)+'\n')
