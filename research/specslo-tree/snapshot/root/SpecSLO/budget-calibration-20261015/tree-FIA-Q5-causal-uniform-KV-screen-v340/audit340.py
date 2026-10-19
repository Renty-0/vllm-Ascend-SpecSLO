"""Recount finite operator observations; differences are results, not passes."""
import json
import math
from pathlib import Path
import statistics

from probe340 import PREFIXES, HEADS, variants, REPEATS, ROUNDS


def need(condition, message):
    if not condition:
        raise RuntimeError('ROOT340: '+message)


def audit(path, *, expected_source):
    data = json.loads(Path(path).read_text())
    need(data['schema']=='ROOT340.synthetic-Q5-causal-uniform-KV.v1' and data['seed']==340
         and data['source_sha256']==expected_source,'actual source and declared seed')
    need(data['models_loaded'] is data['model_qualified'] is data['full_model_or_service_performance'] is False
         and data['Goodput'] is None,'no model or service claim')
    expected=[(h,k,p,name,e,i,phase) for h,k in HEADS for p in PREFIXES
              for name,e,i in variants(p) for phase in ('A1','B','A2')]
    rows=data['rows']
    actual=[tuple(r[k] for k in ('heads','KVheads','prefix','name','extent','inner_precise','phase')) for r in rows]
    need(actual==expected and len(rows)==294,'complete planned order and all294 observations')
    totals={}
    for r in rows:
        need(r['physical_M']==120 and r['groups']==24 and r['width']==5 and r['finite'] is True,
             'finite complete fixed query geometry')
        flags=r['independent_root_raw16_equal_by_depth'];errors=r['independent_root_max_abs_by_depth']
        groups=r['group_position_raw16_equal_by_depth']
        need(len(flags)==len(errors)==len(groups)==5 and all(type(f)is bool for f in flags+groups),
             'complete five-depth comparisons')
        need(all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in errors), 'finite own-root errors')
        need(flags==[e==0 for e in errors], 'own-root raw16 difference arithmetic')
        x=r['max_abs_uniform_FULL2560']
        need(type(x)in(int,float) and math.isfinite(x) and x>=0
             and r['all_outputs_equal_uniform_FULL2560'] is (x==0), 'baseline difference arithmetic')
        need(r['complete_output_restored'] is (True if r['phase']=='A2' else None), 'whole ABA output restored')
        key=str(r['heads'])+'-'+r['name']
        t=totals.setdefault(key,dict(cases=0,independent_root_noninvariant=0,
            group_position_noninvariant=0,different_from_FULL2560=0,max_abs_root=0.))
        t['cases']+=1;t['independent_root_noninvariant']+=int(not all(flags))
        t['group_position_noninvariant']+=int(not all(groups))
        t['different_from_FULL2560']+=int(not r['all_outputs_equal_uniform_FULL2560'])
        t['max_abs_root']=max(t['max_abs_root'],max(errors))
    timing=data['timings'];need(len(timing)==16,'all16 current operator cost panels')
    summaries=[]
    for (h,k,p),t in zip(((h,k,p) for h,k in HEADS for p in PREFIXES),timing,strict=True):
        need((t['heads'],t['KVheads'],t['prefix'])==(h,k,p),'exact timing panel')
        names=['FULL2560']+([] if variants(p)[0][0]=='FULL2560' else [variants(p)[0][0]])+['causal-Native','causal-inner1']
        need(t['names']==names and t['repeats']==REPEATS and t['rotations']==ROUNDS
             and len(t['pairs'])==ROUNDS and t['all_graph_outputs_raw16_equal_eager'] is True,
             'actual graph arithmetic and complete rotated measurements')
        for i,pair in enumerate(t['pairs']):
            need(pair['order']==names[i%len(names):]+names[:i%len(names)]
                 and set(pair['milliseconds_per_FIA'])==set(names),'balanced actual arm rotation')
            need(all(type(x)in(int,float) and math.isfinite(x) and x>0 for x in pair['milliseconds_per_FIA'].values()),
                 'finite positive actual costs')
        med={name:statistics.median(pair['milliseconds_per_FIA'][name] for pair in t['pairs']) for name in names}
        need(t['median_ms']==med,'all operator medians recounted')
        summaries.append(dict(heads=h,prefix=p,median_per_FIA_ms=med))
    return dict(status='complete_synthetic_screen_only',rows=len(rows),arithmetic=totals,
        costs=summaries,models_loaded=False,model_qualified=False,Goodput=None,adopted=False)
