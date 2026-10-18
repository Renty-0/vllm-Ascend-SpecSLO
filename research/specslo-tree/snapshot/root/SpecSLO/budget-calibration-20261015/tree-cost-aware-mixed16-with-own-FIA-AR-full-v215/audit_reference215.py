"""Recompute retained own-AR history and metadata using stdlib only."""
import gzip,hashlib,json
from pathlib import Path


def need(value,message):
    if value is not True:raise RuntimeError('ROOT215 independent AR audit: '+message)


def read(path):
    path=Path(path)
    with (gzip.open(path,'rt')if path.suffix=='.gz'else path.open())as f:return json.load(f)


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def audit(directory):
    directory=Path(directory);folder=directory/'postfull-independent-FIA-AR'
    summaries=[read(folder/f'rank{r}-summary.json')for r in range(4)]
    comparisons=[];all_reference=[];metadata_cases=0;canary_cases=0
    for rank,summary in enumerate(summaries):
        need(summary['rank']==rank and summary['completed']is True
             and summary['reference_only_prompt_plus_own_outputs']is True
             and summary['candidate_KV_or_tokens_seeded']is False
             and summary['performance_clock_already_finished']is True
             and summary['cache_released']is True
             and summary['cross_backend_PA_or_official_AR_equivalence_claimed']is False,
             'actual explicit independent reference scope')
        raw=directory/'windows/window32/full463'/f'rank{rank}-full463-actual-trace.json.gz'
        need(str(raw)==summary['source_artifact']
             and hashlib.sha256(raw.read_bytes()).hexdigest()==summary['source_artifact_sha256'],
             'exact own completed candidate source bytes')
        trace=read(raw);groups=[r['sequence_ids']for r in trace['standalone_prefill_calls']]
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
                    need(saved['segment_begin']==begin and saved['step']==step and len(heads)==80
                         and all(type(v)is int and 0<=v<151936 for v in heads), 'actual complete80 head and own step order')
                    lengths=[len(trace['tokenized_inputs'][owners[i]])+len(actual[i])+4 for i in selected]
                    positions=[];ids=[]
                    for i in selected:
                        p=len(trace['tokenized_inputs'][owners[i]])+len(actual[i])-1
                        positions.extend(range(p,p+5));ids.extend([i]*5)
                    for i in range(16-len(selected)):
                        positions.extend(range(5));ids.extend([463+i]*5);lengths.append(5)
                    if step in (0,127,254):
                        meta=saved['metadata'];slots=[tables[i][p//128]*128+p%128 for i,p in zip(ids,positions,strict=True)]
                        need(meta['positions']==positions and meta['slots']==slots
                             and meta['contexts']==[p+1 for p in positions]
                             and meta['actual_seq_lengths_q']==list(range(5,81,5))
                             and meta['sequence_lens']==lengths,'independent own-history FIA lengths, positions and physical slots')
                        physical_owners=selected+list(range(463,463+16-len(selected)))
                        for i,(owner,length)in enumerate(zip(physical_owners,lengths,strict=True)):
                            used=(length+127)//128
                            need(meta['block_tables'][i][:used]==tables[owner][:used], 'own correct request page-table prefix')
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
            candidate={r['request_id']:r['completion_token_ids']for r in trace['generate_results']}
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
    return dict(schema='ROOT215.independent-retained-own-AR-audit.v1',passed_retained_history_metadata_audit=True,
        references_per_target=463,output_tokens_per_target=118528,by_target_rank=comparisons,
        metadata_cases=metadata_cases,causal_future_canary_cases=canary_cases,
        same_backend_exact_output_qualified=all(v['comparison']['exact_all_tokens_equal']for v in comparisons),
        cross_backend_PA_or_official_AR_equivalence_claimed=False,
        retained_model_heads_recomputed_on_HOST=False,formal_Goodput=False,adopted=False)
