"""HOST-only common-clock collection and predeclared nine-cell aggregation.

This module never launches models and never infers tree AR qualification.
"""
import hashlib,json,math,statistics
from pathlib import Path
from read_common_publication_metrics import tree,fixed,official,read,descriptor
from publication_request_identity import fingerprint

EXPECTED_WORKLOAD='7463ebc2e457179b786cb716b0ce1801ff2bc623ee1f2bb980bdf6fc8958e869'
METHODS=('fixed_gamma4','qualified_dynamic_tree','official_MC2_ON')
KINDS={'fixed_gamma4':'fixed','qualified_dynamic_tree':'tree','official_MC2_ON':'official'}
ORDERS=(('fixed_gamma4','qualified_dynamic_tree','official_MC2_ON'),
        ('official_MC2_ON','qualified_dynamic_tree','fixed_gamma4'),
        ('fixed_gamma4','qualified_dynamic_tree','official_MC2_ON'))
FIXED_FLOOR=501.63560407964184
OFFICIAL_FLOOR=429.8442987424407

def need(ok,message):
    if ok is not True:raise ValueError(message)

def collect(method,source,*,witness_directory=None):
    need(method in METHODS,'one predeclared measured method')
    value=read(source)
    if method=='qualified_dynamic_tree':
        metrics,paths,contract=tree(source);requests=value['requests']
    elif method=='fixed_gamma4':
        need(witness_directory is not None,'fixed actual worker stream witness required')
        metrics,paths,contract=fixed(source,witness_directory)
        requests=[json.loads(line)for line in Path(value['request_manifest']).read_text().splitlines()if line.strip()]
    else:
        metrics,paths,contract=official(source);requests=value['source_rows']
    identity=fingerprint(requests)
    need(identity['sha256']==EXPECTED_WORKLOAD,'identical ordered prompts, Poisson arrivals, IDs and SLOs')
    need(metrics['all']['requests']==463 and metrics['all']['output_tokens']==118528
         and metrics['all_final_outputs_reconstructed']is True,'complete463x256 real outputs')
    need({k:v['requests']for k,v in metrics['by_class'].items()}==dict(tight=271,normal=96,loose=96),
         'complete original three request classes')
    need(metrics['client_delivery_clock_claimed']is False and metrics['original_elapsed_not_adjusted']is True,
         'actual worker/engine clock and unadjusted original full wall')
    metrics.update(source_descriptors=[descriptor(path)for path in paths],input_contract=contract,
                   request_identity=identity,comparison_method=method,HOST_arithmetic_only=True,
                   AR_qualification_inferred=False)
    return metrics

def aggregate(cells):
    need(type(cells)is list and len(cells)==9,'all nine cells required; no partial or fastest subset')
    tags=set();sources=set();candidate_sources=set();ordered=[]
    for block,order in enumerate(ORDERS,1):
        for position,method in enumerate(order):
            cell=cells[len(ordered)]
            need(cell['block']==block and cell['position']==position and cell['method']==method,
                 'predeclared block order is preserved')
            need(type(cell['tag'])is str and cell['tag']not in tags,'fresh process tag for every measured cell')
            tags.add(cell['tag'])
            metrics=cell['metrics'];all_rows=metrics['all']
            need(metrics['comparison_method']==method and metrics['request_identity']['sha256']==EXPECTED_WORKLOAD,
                 'normalized same-input method and full workload identity')
            need(all_rows['requests']==463 and all_rows['output_tokens']==118528,
                 'complete original cell, no dropped requests')
            need({k:v['requests']for k,v in metrics['by_class'].items()}==dict(tight=271,normal=96,loose=96),
                 'all classes retained')
            for key in ('Goodput','Raw'):
                need(type(all_rows[key])in(int,float)and math.isfinite(all_rows[key])and all_rows[key]>0,
                     'actual finite positive throughput')
            need(all_rows['Goodput']<=all_rows['Raw']and 0<=all_rows['attained']<=463,
                 'attainment and throughput conservation')
            primary=metrics['source_descriptors'][0]['path']
            need(primary not in sources,'different actual result artifact in every cell')
            sources.add(primary)
            validation=cell['validation']
            need(validation['source_and_environment_verified']is True
                 and validation['physical_cards']==[2,5,6,7]
                 and validation['all_owned_workers_exited']is True
                 and validation['all4_idle_after']is True,'source, actual cards, owned cleanup and actual idle')
            if method=='qualified_dynamic_tree':
                need(validation['all_original_and_new_gates_passed']is True
                     and validation['whole_workload_AR_qualified']is True
                     and validation['AR_equal_requests']==463 and validation['AR_different_requests']==0,
                     'fresh full tree qualification and own-history equality in every final tree cell')
                candidate_sources.add(validation['candidate_freeze_sha256'])
            elif method=='official_MC2_ON':
                need(validation['ordinary_MC2_actually_enabled']is True
                     and validation['MC2_projection_modules_by_rank']==[128]*4,
                     'actual four-rank ordinary MC2 baseline')
            ordered.append(cell)
    need(len(candidate_sources)==1,'one fixed selected tree source across all final repeats')
    summaries={}
    for method in METHODS:
        rows=[c for c in ordered if c['method']==method]
        summaries[method]=dict(Goodput_median=statistics.median(c['metrics']['all']['Goodput']for c in rows),
                               Goodput_all=[c['metrics']['all']['Goodput']for c in rows],
                               Raw_median=statistics.median(c['metrics']['all']['Raw']for c in rows),
                               Raw_all=[c['metrics']['all']['Raw']for c in rows],
                               attained_all=[c['metrics']['all']['attained']for c in rows])
    tree_gp=summaries['qualified_dynamic_tree']['Goodput_median']
    fixed_gp=summaries['fixed_gamma4']['Goodput_median']
    official_gp=summaries['official_MC2_ON']['Goodput_median']
    fixed_reference=max(FIXED_FLOOR,fixed_gp);official_reference=max(OFFICIAL_FLOOR,official_gp)
    return dict(schema='SpecSLO.final-common-clock-nine-cell-comparison.v1',
                all_cells=ordered,method_summaries=summaries,
                historical_fixed_floor=FIXED_FLOOR,historical_strong_MC2_floor=OFFICIAL_FLOOR,
                effective_fixed_reference=fixed_reference,effective_official_reference=official_reference,
                tree_to_current_fixed_ratio=tree_gp/fixed_gp,tree_to_current_official_ratio=tree_gp/official_gp,
                tree_to_strong_official_ratio=tree_gp/official_reference,
                tree_exceeds_current_and_historical_fixed=tree_gp>fixed_reference,
                tree_exceeds_1p5_current_and_historical_MC2=tree_gp>1.5*official_reference,
                stated_performance_and_full_AR_goals_met=tree_gp>fixed_reference and tree_gp>1.5*official_reference,
                statistics='all cells and medians across three predeclared blocks; no fastest run selected',
                clock='actual worker/engine publication; original full wall retained',
                client_delivery_clock_claimed=False,HOST_aggregation_only=True)
