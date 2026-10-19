"""Audit observation provenance; successful controls still grant no live scope."""
import json
from pathlib import Path


def need(value,message):
    if value is not True:raise RuntimeError('ROOT380 observation audit: '+message)


def read(path):return json.loads(Path(path).read_text())


def audit(directory):
    directory=Path(directory)
    need(read(directory/'completed.json')['replies']==[['calibrated',r,dict(numeric_cases=3,performance=False)]for r in range(4)],'actual four-role completion')
    need(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'original owned shutdown')
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    heads=[];observations=[]
    for rank,report in enumerate(reports):
        need(report['rank']==rank and report['diagnostic_controls']==3
            and report['independent_prefills']==(0 if rank==0 else 288)
            and report['model_qualified'] is False and report['performance'] is False
            and report['full463'] is False and report['adopted'] is False
            and report['Raw'] is report['Goodput'] is None,'no qualification/performance scope')
        saved=read(directory/f'rank{rank}-observation380.json')
        need(all(report[k]==v for k,v in saved.items()),'same retained actual observation')
        if rank==0:
            need(report['modes']==[None]*4,'draft makes no target claim');continue
        need(report['model_dtype']=='torch.bfloat16','original BF16 target')
        modes=report['modes'];base=modes[0]
        need([m['mode']for m in modes]==['fixed120','qkv120','columns120','columns_and_mc2_120']
            and [m['M']for m in modes]==[120,80,80,80]
            and [m['S']for m in modes]==[24,16,16,16],'actual original/reference and three eager controls')
        need(base['reference_greedy_equal']and base['hidden_original_matcher_passed']
            and base['KV_original_matcher_passed']and base['KV_all128_bitwise_equal'],'actual fixed120 reference')
        for mode in modes:
            need(mode==read(directory/f'rank{rank}-{mode["mode"]}-control380.json'),'each independent control retained')
            need(mode['physical_tokens_first80']==base['physical_tokens_first80']
                and mode['physical_positions_first80']==base['physical_positions_first80']
                and len(mode['controlled_eager_heads'])==80
                and mode['greedy_difference_rows']==[i for i,(a,b)in enumerate(zip(base['controlled_eager_heads'],mode['controlled_eager_heads']))if a!=b]
                and mode['reference_greedy_equal']==(mode['greedy_difference_rows']==[]),'equal actual real inputs and honest token differences')
            c=mode['control']
            need(c['original_column_padding_restored']and c['original_mc2_dispatch_restored']
                and c['qualification_flags_assigned'] is False
                and c['qualification_bodies_changed'] is False
                and c['weights_or_precision_changed'] is False,'temporary eager controls restored without qualification override')
            need(c['MC2_projection_calls']==(128 if mode['mode']=='columns_and_mc2_120'else 0)
                and c['padded_column_modules']==dict(fixed120=0,qkv120=64,columns120=129,columns_and_mc2_120=129)[mode['mode']], 'actual model calls and original modules controlled')
            need(mode['all72_prefixes_protected']and mode['bank']['independent_neural_prefills']==72
                and mode['bank']['all_triplet_roots_equal']and mode['bank']['all_triplet_prefix_KV_equal'],'four independently prefilled protected banks')
            if mode['mode']!='fixed120':
                o=mode['observation'];need(o['diagnostic_synchronizes_before_capture']and o['model_qualified'] is False,'diagnostic-only trace')
        heads.append([m['controlled_eager_heads']for m in modes])
        observations.append(dict(rank=rank,controls=[dict(mode=m['mode'],greedy_difference_rows=m['greedy_difference_rows'],hidden_original_matcher_passed=m['hidden_original_matcher_passed'],KV_original_matcher_passed=m['KV_original_matcher_passed'],first_difference=None if m['observation'] is None else m['observation']['first_common_tensor_difference'])for m in modes]))
    need(heads[0]==heads[1]==heads[2],'all target ranks agree on all controlled heads')
    return dict(status='retained_three_GEMM_controls380_NOT_model_qualification',observations=observations,
        model_qualified=False,performance=False,adopted=False,Raw=None,Goodput=None),\
        'observations_retained_no_equivalence_or_performance_claim',dict(hidden_KV_recomputed_by_HOST=False)
