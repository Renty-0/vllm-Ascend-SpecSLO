"""Prospective validation of the fixed pre-full462 cost proposal; no NPU."""
from pathlib import Path
from statistics import median
import argparse,datetime,hashlib,json
D=Path(__file__).resolve().parent
MODEL=D/'ROOT462_PROSPECTIVE_PREFILL_COST_MODEL_HOST.json'
EXPECTED='f0d54b1e692f02d5b7ead5aec9ca2973cc98827254f5830698124771519ad8fc'
def validate(analysis_path):
    raw=MODEL.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
    model=json.loads(raw);source=Path(analysis_path);analysis=json.loads(source.read_text())
    assert analysis['tag']==model['prospective_validation_tag']
    assert analysis['status']=='HOST_ACTUAL_TRACE_PREFILL_PAUSE_EXPOSURE_ONLY'
    panels=[]
    for row in analysis['prefills']:
        predicted=model['safety_multiplier']*(max(0,model['intercept_ms'])+
            model['slope_ms_per_prompt_query']*row['prompt_queries'])
        actual=row['prefill_ms'];assert predicted>0 and actual>=0
        panels.append(dict(ordinal=row['ordinal'],incumbents=row['incumbents'],
            admitted=row['admitted'],prompt_queries=row['prompt_queries'],actual_ms=actual,
            predicted_with_margin_ms=predicted,underestimated=actual>predicted,
            underestimate_ms=max(0,actual-predicted),actual_over_prediction=actual/predicted))
    summaries={}
    for label,subset in [('all',panels),('with_incumbents',[p for p in panels if p['incumbents']>0])]:
        assert subset
        summaries[label]=dict(calls=len(subset),underestimated=sum(p['underestimated']for p in subset),
            maximum_underestimate_ms=max(p['underestimate_ms']for p in subset),
            median_actual_over_prediction=median(p['actual_over_prediction']for p in subset),
            maximum_actual_over_prediction=max(p['actual_over_prediction']for p in subset))
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='HOST462_PREFILL_COST_PROSPECTIVE_VALIDATION_COMPLETE',model=str(MODEL),
        original_model_SHA256=EXPECTED,analysis=str(source),analysis_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        actual_trace_SHA256=analysis['source_SHA256'],panels=panels,summaries=summaries,
        model_refitted=False,NPU_used=False,serving_policy_installed=False,
        interpretation='Report every underprediction. This evaluates a cost estimate on a subsequent run and does not prove a hard runtime bound or any alternative-policy Goodput.')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=validate(args.analysis);result['reader_SHA256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with Path(args.output).open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='panels'}))
