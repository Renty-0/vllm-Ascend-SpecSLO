"""Generate a real three-token continuation of the draft's root alternative.

Call after the unchanged original primary-four generator has materialized the
root. The alternate uses private PA tables and cannot advance the real ledger.
This module does not grant serving or performance qualification.
"""
def inputs(engine,owners,prefixes,primary,alternate,*,steps=3):
    from layout271 import prepare
    from metadata271 import build
    from vllm_ascend.spec_decode.pearl.native_engine import _next_linear_draft_graph_bucket
    desc=prepare(engine.cache_allocation.block_tables,owners,prefixes)
    bucket=_next_linear_draft_graph_bucket(len(owners),engine.config.max_num_seqs)
    args=build(engine,desc,alternate,primary,steps=steps,bucket=bucket,
               pad=engine.graph_runner._pad_inputs)
    return desc,args


def generate(engine,owners,prefixes,primary,original_output,*,copy_state,steps=3):
    import torch
    if not engine.is_draft:raise RuntimeError('ROOT277 genuine TP1 draft only')
    count=len(owners)
    rows=original_output['draft_token_ids'];scores=original_output['node_confidences']
    if len(rows)!=count or len(scores)!=count or any(len(r)!=8 for r in rows):
        raise ValueError('ROOT277 actual original full8 top2 pool required')
    if original_output['root_token_ids']!=list(primary):
        raise ValueError('ROOT277 actual current roots must agree')
    alternate=[r[4]for r in rows]
    desc,args=inputs(engine,owners,prefixes,primary,alternate,steps=steps)
    copy_state.run(desc)
    output=engine.graph_runner.run_draft_greedy(*args,engine.draft_vocab_size,
        valid_row_count=count,return_confidence=True)
    # Copy immediately: later heads/reference graphs can share graph-pool memory.
    encoded=output.detach().clone()
    execution=engine.graph_runner.last_draft_execution
    if execution.mode not in('capture_replay','replay')or not execution.used_aclgraph:
        raise RuntimeError('ROOT277 original Native draft ACLGraph gate failed')
    if tuple(encoded.shape)!=(args[0].numel(),steps,2):
        raise RuntimeError('ROOT277 genuine top2 encoded shape differs')
    pairs=encoded[:count].cpu().tolist();new_rows=[];new_scores=[]
    for row,score,values in zip(rows,scores,pairs,strict=True):
        tokens=[int(v[0]//1_000_001)for v in values]
        confidence=[float(v[0]%1_000_001)/1_000_000 for v in values]
        new_rows.append(list(row[:5])+tokens+list(row[5+steps:]))
        new_scores.append(torch.cat((score[:5],torch.tensor(confidence,dtype=score.dtype,
            device=score.device),score[5+steps:])))
    result=dict(original_output,draft_token_ids=new_rows,node_confidences=new_scores,
        num_draft_tokens=[5+steps]*count,
        draft_confidence=torch.tensor([sum(v.cpu().tolist())/8 for v in new_scores],
            dtype=original_output['draft_confidence'].dtype,device=original_output['draft_confidence'].device),
        execution_mode='primary4-plus-private-root-alternate-chain-v277')
    return dict(output=result,encoded=encoded,description=desc,metadata=args,
        actual_draft_entry_key=engine.graph_runner.last_draft_entry_key,
        execution_mode=execution.mode,steps=steps,
        canonical_tables_or_ledger_updated=False)
