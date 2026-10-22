"""Read original retained traces; arithmetic pause exposure, never causal ablation."""
from pathlib import Path
from statistics import median
import argparse,datetime,gzip,hashlib,json,math,tarfile
B=Path('/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209')
def p90(values):
    v=sorted(values);x=(len(v)-1)*0.9;i=int(x)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def run(tag,expected):
    root=B/'actual'/tag;relative='windows/window64/full463/rank1-full463-actual-trace.json.gz'
    source=root/relative
    if source.exists():data=source.read_bytes();location=str(source)
    else:
        marker=json.loads((root/'ARCHIVED.json').read_text())
        with tarfile.open(marker['archive'],'r:gz')as archive:
            found=[m for m in archive.getmembers()if m.name.endswith(relative)]
            assert len(found)==1
            data=archive.extractfile(found[0]).read();location=marker['archive']+'::'+found[0].name
    digest=hashlib.sha256(data).hexdigest();assert digest==expected
    trace=json.loads(gzip.decompress(data));requests=trace['requests'];assert len(requests)==463
    first=[None]*463;last=[None]*463;counts=[0]*463
    for event in trace['stream_events']:
        i=event['request_index'];assert requests[i]['request_id']==event['request_id']
        if event['token_ids']:
            t=event['elapsed_seconds']
            if first[i]is None:first[i]=t
            assert last[i]is None or t>=last[i]
            last[i]=t;counts[i]+=len(event['token_ids'])
    assert counts==[256]*463
    pause=[0.0]*463;panels=[]
    witnesses=trace['ROOT432_refill_witnesses'];prefills=trace['standalone_prefill_calls']
    assert len(witnesses)==len(prefills)
    for ordinal,(w,p)in enumerate(zip(witnesses,prefills,strict=True)):
        owners=w['incumbents'];elapsed=w['prefill_ms']
        assert elapsed>=0 and len(owners)==len(set(owners)) and not set(owners)&set(w['admitted'])
        assert list(w['admitted'])==p['sequence_ids']
        assert w['original_incumbent_tokens_epochs_homes_and_ready_tickets_unchanged']is True
        assert len(owners)==len(w['incumbent_decode_before_ms'])==len(w['incumbent_decode_after_ms'])
        for i,a,b in zip(owners,w['incumbent_decode_before_ms'],w['incumbent_decode_after_ms'],strict=True):
            assert math.isclose(b-a,elapsed,rel_tol=0,abs_tol=1e-5)
            pause[i]+=elapsed
        panels.append(dict(ordinal=ordinal,incumbents=len(owners),admitted=len(w['admitted']),
            prompt_queries=p['uncached_queries'],prefill_ms=elapsed,
            total_incumbent_pause_ms=len(owners)*elapsed))
    rows=[]
    for i,request in enumerate(requests):
        duration=(last[i]-first[i])*1000
        assert pause[i]<=duration+1e-5
        rows.append(dict(request_id=request['request_id'],slo_class=request['slo_class'],
            actual_paper_TPOT_ms=duration/256,charged_prefill_pause_ms=pause[i],
            arithmetic_TPOT_pause_component_ms=pause[i]/256))
    classes={}
    for label in ('tight','normal','loose'):
        selected=[r for r in rows if r['slo_class']==label]
        actual=[r['actual_paper_TPOT_ms']for r in selected]
        component=[r['arithmetic_TPOT_pause_component_ms']for r in selected]
        classes[label]=dict(requests=len(selected),actual_TPOT_median_ms=median(actual),
            actual_TPOT_P90_ms=p90(actual),prefill_pause_component_median_ms=median(component),
            prefill_pause_component_P90_ms=p90(component))
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='HOST_ACTUAL_TRACE_PREFILL_PAUSE_EXPOSURE_ONLY',tag=tag,source=location,
        source_SHA256=digest,prefill_calls=len(panels),total_prefill_wall_ms=sum(p['prefill_ms']for p in panels),
        max_single_prefill_ms=max(p['prefill_ms']for p in panels),by_class=classes,prefills=panels,
        pause_charged_to_original_incumbent_decode_clocks_verified=True,
        interpretation='Pause/256 is a factual additive component of the observed mean TPOT. Removing it arithmetically is not a prediction of an alternative scheduler: histories, batches, service time and arrivals can interact.',
        current_run_changed=False,NPU_used=False,formal_Goodput=False,rows=rows)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-trace-sha256',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.tag,args.expected_trace_sha256)
    result['reader_SHA256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Path(args.output).open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('rows','prefills')},ensure_ascii=False))
