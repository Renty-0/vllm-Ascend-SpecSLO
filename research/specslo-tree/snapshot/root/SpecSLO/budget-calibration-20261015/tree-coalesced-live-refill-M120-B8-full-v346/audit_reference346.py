"""Recompute retained own-AR history and metadata using stdlib only."""
import gzip,hashlib,json
from pathlib import Path


def need(value,message):
    if value is not True:raise RuntimeError('ROOT218 independent AR audit: '+message)


def read(path):
    path=Path(path)
    with (gzip.open(path,'rt')if path.suffix=='.gz'else path.open())as f:return json.load(f)


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def expected_static_mask_sha256(positions):
    need(len(positions)==120,'complete reference position list')
    mask=bytearray([1])*(24*10*4096)
    for group in range(24):
        count=positions[5*group]+1
        need(0<count<=2560,'every real or dummy first query fits static extent')
        start=group*10*4096
        mask[start:start+count]=bytearray(count)
        for query in range(1,5):mask[start+query*4096]=0
    return hashlib.sha256(mask).hexdigest()


def audit(directory):
    directory=Path(directory);folder=directory/'postfull-independent-FIA-AR'
    summaries=[read(folder/f'rank{r}-summary.json')for r in range(4)]
    leader_path=directory/'windows/window48/full463/rank1-full463-actual-trace.json.gz'
    leader_bytes=leader_path.read_bytes();leader=read(leader_path)
    leader_info=dict(path=str(leader_path),bytes=len(leader_bytes),sha256=hashlib.sha256(leader_bytes).hexdigest())
    need(leader['rank']==1 and type(leader['generate_results'])is list and len(leader['generate_results'])==463,'actual original output-leader-only results')
    candidate={r['request_id']:r['completion_token_ids']for r in leader['generate_results']}
    need(len(candidate)==463 and set(candidate)=={r['request_id']for r in leader['requests']}and all(len(v)==256 for v in candidate.values()),'complete unique original output-leader results')
    comparisons=[];all_reference=[];metadata_cases=0;canary_cases=0
    for rank,summary in enumerate(summaries):
        need(summary['rank']==rank and summary['completed']is True
             and summary['reference_only_prompt_plus_own_outputs']is True
             and summary['candidate_KV_or_tokens_seeded']is False
             and summary['performance_clock_already_finished']is True
             and summary['cache_released']is True
             and summary['cross_backend_PA_or_official_AR_equivalence_claimed']is False,
             'actual explicit independent reference scope')
        raw=directory/'windows/window48/full463'/f'rank{rank}-full463-actual-trace.json.gz'
        need(str(raw)==summary['source_artifact']
             and hashlib.sha256(raw.read_bytes()).hexdigest()==summary['source_artifact_sha256'],
             'exact own completed candidate source bytes')
        trace=read(raw)
        need(summary['candidate_result_artifact']==leader_info and summary['candidate_result_owner_rank']==1,'exact original output-leader comparison artifact')
        for key in('scope_count','manifest_sha256','prefix_sha256','requests','tokenized_inputs','source_generation','standalone_prefill_calls'):
            need(trace[key]==leader[key],'independent all-rank input/cohort identity '+key)
        need((trace['generate_results']is None)if rank!=1 else(trace['generate_results']==leader['generate_results']),'preserve original only-output-leader result materialization')
        groups=[r['sequence_ids']for r in trace['standalone_prefill_calls']]
        need(sorted(i for group in groups for i in group)==list(range(463)), 'all463 prefill owners once')
        reference={};total_steps=0
        for ordinal,owners in enumerate(groups):
            row=read(folder/f'rank{rank}-cohort{ordinal}.json.gz')
            need(row['ordinal']==ordinal and row['original_owner_indices']==owners
                 and row['request_ids']==[trace['requests'][i]['request_id']for i in owners]
                 and row['candidate_tokens_ever_used_as_inputs']is False
                 and row['own_prompt_only_allocation']is row['own_prefix_KV_only']is row['original_all_rank_finiteness_vote_passed']is True,
                 'actual reference owners and input independence')
            tables=row['block_tables'];pages=[p for t in tables for p in t if p>=0]
            need(len(tables)==527 and len(pages)==len(set(pages))==row['actual_unique_pages']<=544,
                 'actual disjoint independent allocation')
            packet=row['rank_agreement']
            need({v['rank']for v in packet}==set(range(4))
                 and len({v['reference_sha256']for v in packet if v['rank']!=0})==1,
                 'retained all3 independent target trajectories agree')
            if rank==0:
                need(row['reference_generated_on_this_rank']is False and row['reference_completions']is None
                     and row['steps']==row['canaries']==[], 'draft rank never reported own target AR')
                continue
            need(row['reference_generated_on_this_rank']is True and len(row['first_roots'])==len(owners),
                 'target own original prefill roots')
            actual=[[token]for token in row['first_roots']];cursor=0;canary_cursor=0
            for begin in range(0,len(owners),16):
                selected=list(range(begin,min(begin+16,len(owners))))
                for step in range(255):
                    saved=row['steps'][cursor];cursor+=1
                    heads=saved['heads80']
                    need(saved['segment_begin']==begin and saved['step']==step and len(heads)==120
                         and all(type(v)is int and 0<=v<151936 for v in heads), 'actual complete80 head and own step order')
                    lengths=[len(trace['tokenized_inputs'][owners[i]])+len(actual[i])+4 for i in selected]
                    positions=[];ids=[]
                    for i in selected:
                        p=len(trace['tokenized_inputs'][owners[i]])+len(actual[i])-1
                        positions.extend(range(p,p+5));ids.extend([i]*5)
                    for i in range(24-len(selected)):
                        positions.extend(range(5));ids.extend([463+i]*5);lengths.append(5)
                    if step in (0,127,254):
                        meta=saved['metadata'];slots=[tables[i][p//128]*128+p%128 for i,p in zip(ids,positions,strict=True)]
                        need(meta['positions']==positions and meta['slots']==slots
                             and meta['contexts']==[p+1 for p in positions]
                             and meta['actual_seq_lengths_q']==list(range(5,121,5))
                             and meta['sequence_lens']==[2560]*24,'independent own-history static lengths, original positions and physical slots')
                        physical_owners=selected+list(range(463,463+24-len(selected)))
                        for i,(owner,length)in enumerate(zip(physical_owners,lengths,strict=True)):
                            used=(length+127)//128
                            need(meta['block_tables'][i][:used]==tables[owner][:used], 'own correct request page-table prefix')
                        zero=meta['zero_page']
                        need(type(zero)is int and 0<=zero<544 and tables[526][0]==zero,
                             'actual separately allocated reference zero page')
                        need(meta['backend']=='static_FULL_q5_M120_S24_KV2560'
                             and meta['tree_mask_shape']==[24,1,10,4096]
                             and meta['future_query_visibility']=='nonreal q1..q4 attend prefix key0 only'
                             and meta['exact_mask_sha256']==expected_static_mask_sha256(positions),
                             'actual complete attention visibility mask matches own prefix and nonreal canaries')
                        for index,owner in enumerate(physical_owners):
                            need(meta['block_tables'][index]==expected_allocated_table(tables[owner],positions[5*index],zero,5),
                                 'exact full page table, with only unallocated tail mapped to private zero page')
                        canary=row['canaries'][canary_cursor];canary_cursor+=1
                        tokens=[heads[5*i]for i in range(len(selected))]
                        need(canary['segment_begin']==begin and canary['step']==step
                             and canary['real_owners']==selected and canary['zeros']==canary['alternate']==tokens
                             and canary['equal']is True,'actual alternative future canaries leave first query unchanged')
                        metadata_cases+=1;canary_cases+=1
                    else:need(saved['metadata']is None,'metadata sample locations explicit')
                    for i,token in zip(selected,heads[::5],strict=False):actual[i].append(token)
            need(cursor==len(row['steps']) and canary_cursor==len(row['canaries'])
                 and actual==row['reference_completions'] and all(len(v)==256 for v in actual),
                 'all own AR tokens reconstructed from heads only')
            need(all(v['reference_sha256']==digest(actual)for v in packet if v['rank']!=0), 'actual complete cohort target hashes')
            for i,tokens in zip(owners,actual,strict=True):reference[trace['requests'][i]['request_id']]=tokens
            total_steps+=cursor
        d=summary['Native']
        need(set(d['before'])==set(d['after'])==set(d['delta']) and len(d['delta'])==21
             and all(d['after'][k]-d['before'][k]==d['delta'][k]==0 for k in d['delta']), 'all actual21 original graph discovery/failure counters zero')
        if rank:
            need(reference.keys()==candidate.keys()and len(reference)==463,'all independent463 trajectories')
            rows=[]
            for rid,tokens in reference.items():
                wanted=candidate[rid]
                first=next((i for i,(a,b)in enumerate(zip(tokens,wanted,strict=True))if a!=b),None)
                rows.append(dict(request_id=rid,equal=first is None,first_mismatch_zero_based=first,
                    candidate_token=None if first is None else wanted[first],reference_token=None if first is None else tokens[first]))
            result=dict(requests=len(rows),equal_requests=sum(v['equal']for v in rows),mismatching_requests=sum(not v['equal']for v in rows),
                rows=rows,exact_all_tokens_equal=all(v['equal']for v in rows))
            need(result==summary['comparison']and summary['reference_trajectories_generated']==463,
                 'exact comparison including every mismatch independently recomputed')
            comparisons.append(dict(rank=rank,comparison=result,target_steps=total_steps));all_reference.append(reference)
        else:need(summary['reference_trajectories_generated']==0 and summary['comparison']is None,'truthful draft rank summary')
    need(all_reference[0]==all_reference[1]==all_reference[2], 'all3 independently regenerated full reference outputs equal')
    return dict(schema='ROOT346.independent-retained-static-own-AR-audit.v1',passed_retained_history_metadata_audit=True,
        references_per_target=463,output_tokens_per_target=118528,by_target_rank=comparisons,
        metadata_cases=metadata_cases,causal_future_canary_cases=canary_cases,
        same_backend_exact_output_qualified=all(v['comparison']['exact_all_tokens_equal']for v in comparisons),
        cross_backend_PA_or_official_AR_equivalence_claimed=False,
        retained_model_heads_recomputed_on_HOST=False,formal_Goodput=False,adopted=False)


def expected_allocated_table(final_table,position,zero,queries):
    need(len(final_table)==32 and type(position)is int and 0<=position<2560 and queries==5,'complete original own-reference dimensions')
    used=(position+queries+127)//128
    need(all(type(v)is int and 0<=v<544 for v in final_table[:used]),'all currently needed physical pages exist')
    return final_table[:used]+[zero]*(32-used)
