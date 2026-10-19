"""Audit retained observations; never promote a numerical discrepancy to a pass."""
import json
import math
from pathlib import Path


def need(x,message):
    if x is not True:raise RuntimeError('ROOT379 observation audit: '+message)


def read(path):return json.loads(Path(path).read_text())


def audit(directory):
    directory=Path(directory)
    done=read(directory/'completed.json')
    need(done['replies']==[['calibrated',r,dict(numeric_cases=1,performance=False)]for r in range(4)],
         'all four actual diagnostic workers completed')
    need(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'original owned shutdown')
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    heads=[];observations=[]
    for rank,report in enumerate(reports):
        need(report['rank']==rank and report['diagnostic_pairs']==1
             and report['independent_prefills']==(0 if rank==0 else 144)
             and report['models_qualified'] is False and report['full463'] is False
             and report['performance'] is False and report['adopted'] is False
             and report['Raw'] is report['Goodput'] is None
             and report['raw_tensors_saved_to_disk'] is False,'observations only; no qualification or timing')
        saved=read(directory/f'rank{rank}-observation379.json')
        need(all(saved[k]==report[k]for k in saved),'same actual retained observation')
        if rank==0:
            need(report['modes']==[None,None]and report['observation'] is None,'draft makes no target claim')
            continue
        need([r['mode']for r in report['modes']]==['fixed120','adaptive80']
             and [r['M']for r in report['modes']]==[120,80]
             and [r['S']for r in report['modes']]==[24,16], 'actual paired geometries')
        a,b=report['modes']
        input_equal=(a['physical_tokens_first80']==b['physical_tokens_first80']
                     and a['physical_positions_first80']==b['physical_positions_first80'])
        for mode in report['modes']:
            need(len(mode['graph_heads'])==len(mode['eager_heads'])==80
                 and mode['graph_eager_greedy_equal']==(mode['graph_heads']==mode['eager_heads'])
                 and mode['all72_prefixes_after_diagnostic'] is True
                 and mode['bank']['independent_neural_prefills']==72
                 and mode['bank']['all_triplet_roots_equal'] is True
                 and mode['bank']['all_triplet_prefix_KV_equal'] is True,'actual graph/eager and independent prefix observations')
        observation=report['observation']
        need(observation['diagnostic_synchronizes_before_capture'] is True
             and observation['hooks_change_values'] is False
             and observation['raw_tensors_saved_to_disk'] is False
             and observation['model_qualified'] is False,'synchronized observers are not performance evidence')
        for row in observation['rows']:
            if 'bitwise_equal' in row:
                need(type(row['bitwise_equal'])is bool and type(row['different_elements'])is int
                     and 0<=row['different_elements']<=row['elements']
                     and row['bitwise_equal']==(row['different_elements']==0)
                     and math.isfinite(row['max_abs'])and row['max_abs']>=0,'retained real tensor discrepancy arithmetic')
        first=next((r for r in observation['rows']if r.get('bitwise_equal')is False),None)
        need(first==observation['first_common_tensor_difference'],'first actual differing tensor, not a chosen later stage')
        heads.append([a['graph_heads'],b['graph_heads'],a['eager_heads'],b['eager_heads']])
        observations.append(dict(rank=rank,inputs_equal=input_equal,
            graph_difference_rows=[i for i,(x,y)in enumerate(zip(a['graph_heads'],b['graph_heads']))if x!=y],
            eager_difference_rows=[i for i,(x,y)in enumerate(zip(a['eager_heads'],b['eager_heads']))if x!=y],
            first_difference=first,transient_trace_bytes=observation['transient_trace_bytes']))
    need(heads[0]==heads[1]==heads[2],'all TP3 ranks agree on graph and observed eager heads')
    return dict(status='retained_real_stage_observations379_NOT_model_qualification',observations=observations,
        model_qualified=False,performance=False,Raw=None,Goodput=None,adopted=False),\
        'observations_retained_no_equivalence_or_performance_claim',dict(hidden_KV_recomputed_by_HOST=False)
