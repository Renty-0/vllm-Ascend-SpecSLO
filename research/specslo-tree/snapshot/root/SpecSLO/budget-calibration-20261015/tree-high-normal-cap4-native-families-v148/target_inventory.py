"""Explicit legal service fixtures for every native FULL9/causal decode key.

Native capture, original changed-input validation and exact original eager
predictions remain authoritative. This helper never sets qualification flags.
"""
import json
from pathlib import Path
from finite_inventory_plan import targets, target_key, MC2_ROWS


def make_fixture(engine, fixture, *, anchor=0, prefix=127):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan, pack_selected_tree_plan
    from target_layout import pad_target
    plans=[]; rows=[]
    for index,budget in enumerate(fixture['budgets']):
        base=cached_cpu_tree_speculation_plan(2,4,prefix,4096,candidate_budget=8)
        selected=list(range(min(budget,4)))+([4+anchor] if budget==5 else [])
        plans.append(pack_selected_tree_plan(base,selected))
        # Valid, distinct teacher-forced tokens for this known tree. Target
        # verification accepts arbitrary draft candidates, not only argmax.
        rows.append([int((103+index*17+node*29)%engine.draft_vocab_size) for node in selected])
    roots=[int(41+i)%engine.draft_vocab_size for i in range(len(plans))]
    ids=list(range(len(plans)))
    padded=pad_target(engine,plans,roots,rows,ids,[],[])
    pp,rr,dd,ss,real=padded
    if (len(pp)!=fixture['physical_segments']
            or sum(p.candidate_budget+1 for p in pp)!=fixture['physical_queries']
            or len(set(ss))!=len(ss) or real!=fixture['real_count']):
        raise RuntimeError('Legal witness did not reproduce the planned target envelope')
    return padded


def _original_eager(engine, padded):
    import torch
    from host_target_metadata import make_host_tree_attention_metadata
    from fixed_mask_extent import padded as pad_mask
    plans,roots,rows,ids,_=padded
    inputs,positions,metadata=make_host_tree_attention_metadata(
        engine,plans,roots,rows,sequence_ids=ids)
    metadata=pad_mask(metadata)
    result=engine.graph_runner._execute_target([inputs],[positions],[metadata],engine.draft_vocab_size)
    torch.npu.synchronize()
    return result


def qualify(engine):
    import torch
    import torch.distributed as dist
    from inventory_memory import checkpoint
    from target_capture_order import ordered_targets
    fixtures=ordered_targets()
    runner=engine.graph_runner
    if runner.graph_cache_sealed:
        raise RuntimeError('Explicit target inventory must precede native sealing')
    group=engine.groups.verification_coordination_group
    if group is None or dist.get_backend(group)!='gloo':
        raise RuntimeError('Explicit target inventory requires full CPU coordination')
    from mc2_inventory_contract import audit as require_MC2_preflight
    require_MC2_preflight(Path(__file__).resolve().parent/'startup-progress')
    if not engine.is_draft:
        # Real request IDs are pre-admission, separately reserved from padding
        # scratch IDs. Give each actual root a real, independently written
        # causal prefix; no scratch-only/invalid shapes enter qualification.
        for seq in range(32):
            prompt=[int((7+seq*13+j*19)%engine.draft_vocab_size) for j in range(127)]
            engine._activate_cache_sequence(seq,prompt,enable_prefix_caching=False)
            engine._run_packed_hidden(prompt,[seq]*127,list(range(127)),False,None,True)
    checkpoint(engine,'target_decode',0,len(fixtures))
    observed=[]
    for number,fixture in enumerate(fixtures,1):
        details=dict(fixture)
        details['capture_position']=number
        if not engine.is_draft:
            from vllm_ascend.spec_decode.pearl.mc2 import snapshot_mc2_dispatch_counters
            before_MC2=snapshot_mc2_dispatch_counters()
            if fixture['physical_queries'] not in MC2_ROWS:
                raise RuntimeError('Target witness is outside full-chain MC2 row intersection')
            expected=target_key(fixture['family'],fixture['physical_queries'],fixture['physical_segments'])
            anchors=range(4) if fixture['family']=='full9' else (0,)
            variants=[('primary',fixture,anchor) for anchor in anchors]
            if fixture['alternate_witness'] is not None:
                variants.append(('alternate',fixture['alternate_witness'],0))
            checked_partitions=[]
            for label,witness,anchor in variants:
                packed=make_fixture(engine,witness,anchor=anchor)
                pp,rr,dd,ss,real=packed
                result=engine.target_tree_forward(pp,rr,dd,sequence_ids=ss,
                                                 return_logits=False,real_tree_count=real)
                actual=getattr(runner,'_dynamic_verification_last_key',None)
                if actual!=expected:
                    raise RuntimeError(f'Actual native target key differs: {actual!r} != {expected!r}')
                entry=runner.target_entries.get(actual)
                if (entry is None or not entry.runtime_validated or not result['used_aclgraph']
                        or runner.last_target_execution.mode not in ('capture_replay','replay')):
                    raise RuntimeError('Original target native qualification did not complete')
                ends=[]
                for length in witness['physical_query_lengths']:
                    ends.append((ends[-1] if ends else 0)+length)
                if (entry.actual_seq_lengths_q!=(tuple(ends),)
                        or entry.sequence_lens!=(tuple(witness['physical_KV_lengths']),)):
                    raise RuntimeError('Native exact-length task update lost the actual query/KV partition')
                actual_outputs=tuple(value.clone() for value in entry.outputs)
                reference=_original_eager(engine,packed)
                if not runner._target_outputs_match(actual_outputs,reference):
                    raise RuntimeError('Explicit tree anchor/length fixture changed original eager predictions')
                checked_partitions.append(dict(witness=label,anchor=anchor,
                    actual_seq_lengths_q=list(ends),sequence_lens=witness['physical_KV_lengths']))
            after_MC2=snapshot_mc2_dispatch_counters()
            delta_MC2={name:int(after_MC2[name]-before_MC2[name]) for name in after_MC2}
            if (delta_MC2['native_epilogue_success']<=0 or delta_MC2['fallback']
                    or delta_MC2['exception']):
                raise RuntimeError('Actual fixture did not retain complete MC2 native dispatch')
            observed.append(repr(expected))
            details['semantic_anchors_checked']=len(anchors)
            details['alternate_witness_checked']=fixture['alternate_witness'] is not None
            details['native_partitions_checked']=checked_partitions
            details['actual_host_MC2_dispatch_delta']=delta_MC2
        checkpoint(engine,'target_decode',number,len(fixtures),detail=details)
    if not engine.is_draft:
        expected={f['expected_key'] for f in fixtures}
        if set(observed)!=expected or not expected.issubset(set(map(repr,runner.target_entries))):
            raise RuntimeError('Target native decode family inventory is incomplete')
        engine._finite_target_inventory_complete=True
        path=Path(__file__).resolve().parent/'startup-progress'/f'finite-target-rank{engine.rank}.json'
        path.write_text(json.dumps(dict(status='complete',native_keys=observed,
            fixtures=len(fixtures),original_native_flags_unchanged=True),indent=2)+'\n')
