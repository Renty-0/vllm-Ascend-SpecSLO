"""Retain original measurements, including negative numerical or timing outcomes."""
import importlib.util
import json
import math
from pathlib import Path

EXAMPLES=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/examples')


def need(value,message):
    if value is not True:raise RuntimeError('ROOT385 actual audit: '+message)


def read(path):return json.loads(Path(path).read_text())


def audit(directory):
    directory=Path(directory)
    need(read(directory/'completed.json')['replies']==[['calibrated',r,dict(numeric_cases=1,performance=False)]for r in range(4)],'all4 actual workers completed')
    need(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'original owned shutdown')
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    for rank,r in enumerate(reports):
        need(r['rank']==rank and r['independent_prefills']==(0 if rank==0 else 72)
            and r['operator_qualification_only'] is True and r['new_profile_installed'] is False
            and r['current_profile_unchanged'] is True and r['full463'] is False
            and r['adopted'] is False and r['Raw'] is r['Goodput'] is None,'operator sequence only, no installed profile or generation claim')
        if rank==0:
            need(r['measurement'] is None and r['all72_prefixes_protected'] is None,'draft no target claim');continue
        m=r['measurement']
        need(r['all72_prefixes_protected'] is True and r['model_dtype']=='torch.bfloat16'
            and m['TP_world_size']==3 and m['TP_rank']==rank-1
            and m['actual_init_checks']==m['actual_destroy_noops']==1
            and m['original_main_unchanged'] is True and m['Native_flags_assigned'] is False
            and m['actual_epilogues_per_chain']==128,'genuine borrowed TP3 group and real payload experiment')
    path=directory/'actual-original-M120-deferred-chain385.json'
    value=read(path)
    need(value['row_count']==120 and value['layer_count']==64 and value['operations_per_chain']==128
        and value['execution_mode']=='graph'and value['tensor_parallel_size']==3
        and value['tp_rank_device_mapping']==['1','2','3'],'actual original graph experiment and device mapping')
    for key in ['baseline_latency_ms','fully_flushed_latency_ms','deferred_chain_latency_ms']:
        need(len(value['samples'][key])==9 and all(type(x)in (int,float)and math.isfinite(x)and x>0 for x in value['samples'][key]),'nine genuine positive samples per mode')
    old=read('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json')
    need(value['source_sha256']==old['metadata']['source_sha256'],'same original kernel source')
    spec=importlib.util.spec_from_file_location('_ROOT385_original_evidence_loader',EXAMPLES/'specslo_mc2_deferred_chain_evidence.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        evidence=module.load_deferred_chain_evidence([path],expected_source_sha256=value['source_sha256'],
            expected_mapping=value['tp_rank_device_mapping'],minimum_samples=7,minimum_speedup=1.0,
            allowed_row_counts=[120],allowed_input_sources=[value['attention_input_source'],value['down_input_source']])
        qualified=True;rejection=None
    except ValueError as exc:
        qualified=False;evidence=None;rejection=str(exc)
    return dict(status='retained_original_real120_chain_measurement385',qualified_by_original_loader=qualified,
        original_loader_rejection=rejection,evidence=evidence,summary=value['summary'],
        no_profile_installed=True,full463=False,Raw=None,Goodput=None,adopted=False),\
        'operator_measurement_retained_not_serving_qualification',dict(hidden_KV_recomputed_by_HOST=False)
