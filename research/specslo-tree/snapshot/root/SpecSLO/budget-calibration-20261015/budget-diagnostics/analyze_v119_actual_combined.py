"""Terminal-only independent HOST audit of V119; never imports Torch/Native.

Run only after ROOT reports terminal cleanup. Outputs are exclusive and live
outside the frozen source. Numerical tensors were compared by the actual
actors, not retained for an offline tensor re-comparison. Static import/source
resolution is distinguished from unavailable runtime import-event history.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime,timezone
import gzip
import hashlib
import importlib
import importlib.abc
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile
import traceback

BASE=Path('/root/SpecSLO/budget-calibration-20261015')
P=BASE/'tree-s35-qualified-counts-v119';D=BASE/'budget-diagnostics'
TAG='s35-qualified-counts-paired-full463-v119-1'
MANIFEST=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl')
CPU_SHA='26fc57a57960554e5c8ba01efd8212197632937ff3603571f9e28cde5723069c'
FREEZE_SHA='08af41c4812e1a38f93424d948e4dd2b33c965fe70d29a88af70d4cb2ff4f5d9'
ARCHIVE_SHA='8708b03fe3c27d9a918f8f9e8f61e8cf09fd0b53bb4f675e065383acb72d9c14'
PRE_SHA='467d4ac6f5ae1a49d2f0056e5adf036dec4dfd1c831705c11d5489d1fce8ce6d'
FACTORY_SHA='348f9e5c52dcb156e85d08159908048906bcfe527af78a4e29f9a46e29905d4a'
STARTUP_HELPER_SHA='b9e420f70d84966f0c1b5edef2721b6213756506c27e3fd9ec27be3e774ade6b'
ROLES=('warmup65','zero65-1','zero65-2','full463')
SLOS={'tight':40.,'normal':50.,'loose':150.}


def need(value,message):
    if not value:raise ValueError('V119 external: '+message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):result.update(chunk)
    return result.hexdigest()


def desc(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def finite(value):
    if type(value) is dict:
        for v in value.values():finite(v)
    elif type(value) is list:
        for v in value:finite(v)
    elif type(value) is float:need(math.isfinite(value),'nonfinite JSON')
    else:need(value is None or type(value) in (bool,int,str),'unexpected JSON type')


def read(path):
    value=json.loads(Path(path).read_bytes());finite(value);return value


def same(left,right):return json.loads(json.dumps(left))==json.loads(json.dumps(right))


def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode()).hexdigest()


def uncompressed(artifact,directory):
    path=Path(artifact['path']).resolve();need(path.is_relative_to(directory.resolve()),'artifact escaped actual directory')
    need(0<path.stat().st_size==artifact['bytes']<=64*1024**2 and sha(path)==artifact['sha256'],'artifact compressed bytes/SHA')
    with gzip.open(path,'rb') as stream:raw=stream.read(256*1024**2+1)
    need(len(raw)==artifact['uncompressed_bytes']<=256*1024**2,'artifact expansion bytes/bound')
    value=json.loads(raw);finite(value);return value


def archive(path,expected):
    wanted={str(Path(name)).lstrip('/'):value for name,value in expected.items()}
    need(len(wanted)==len(expected),'archive path collision')
    with tarfile.open(path,'r:gz') as source:
        members=source.getmembers()
        need(len(members)==len(wanted) and len({m.name for m in members})==len(members)
            and {m.name for m in members}==set(wanted) and all(m.isfile() and not Path(m.name).is_absolute()
            and '..' not in Path(m.name).parts for m in members),'exact regular archive membership')
        for member in members:
            need(hashlib.sha256(source.extractfile(member).read()).hexdigest()==wanted[member.name],
                'archive member bytes: '+member.name)
    return dict(**desc(path),members=len(wanted),all_member_bytes_checked=True)


class HOSTOnly(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('External HOST auditor forbids device/model import: '+fullname)
        return None


@contextmanager
def modules():
    previous=list(sys.path);guard=HOSTOnly();sys.path.insert(0,str(P));sys.meta_path.insert(0,guard)
    try:
        names=('audit_combined_v113','audit_production_v101','bounded_report_audit_v78',
            'startup_receipts_v75','finite_inventory_plan','local_qualification_scope','shape_scope_v101',
            'source_snapshot_v113')
        result={}
        for name in names:
            result[name]=importlib.import_module(name)
            need(Path(result[name].__file__).resolve().parent==P,'validator resolved outside V119: '+name)
        yield result
    finally:
        sys.path[:]=previous;sys.meta_path.remove(guard)


def local_source_resolution(freeze,legacy):
    """Conservative static closure; never relabel it as an import-event trace."""
    pending=['supervise_combined_v113.py','combined_runtime_v113.py'];seen={};dynamic=[]
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=P/name;need(str(path) in freeze['files'],'reachable local source not archived: '+str(path))
        tree=ast.parse(path.read_text());seen[name]=sha(path)
        for node in ast.walk(tree):
            targets=[]
            if isinstance(node,ast.Import):targets=[x.name.split('.')[0]+'.py' for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:targets=[node.module.split('.')[0]+'.py']
            for target in targets:
                if (P/target).is_file() and target not in seen:pending.append(target)
            if isinstance(node,ast.Call):
                call=ast.unparse(node.func)
                if call in ('exec','eval','compile') or any(part in call for part in ('spec_from_file_location','run_path','run_module','import_module','SourceFileLoader')):
                    dynamic.append(dict(source=str(path),line=node.lineno,call=call,expression=ast.unparse(node)))
    # Every dynamic source site in this conservative local closure has been
    # read against these exact frozen files. These are source/dataflow facts,
    # not a fabricated runtime import-event trace.
    origins={
        'combined_controls_v113.py':(2,'current P/candidate_budget_policy.py FunctionDef AST'),
        'poisson_coverage_probe.py':(1,'current P source bytes in exact-code validators'),
        'original_chunk_binding.py':(2,'inspect source of archived public Native chunk method'),
        'check_v101.py':(2,'current startup_config literal PEARLConfig keyword AST; empty builtins'),
        'host_target_metadata.py':(2,'inspect archived public Native target_tree_forward source'),
        'live_poisson_probe.py':(1,'current exact local live-source builder'),
        'live_decode_patch.py':(2,'current archived atomic/live/v63 source builders'),
        'atomic_decode_patch.py':(2,'current archived v63_runtime_reference.py'),
        'tree_entry.py':(2,'archived public Native decode plus current local source transforms'),
        'host_tree_transport.py':(2,'archived public Native original exchange source'),
        'parallel_steps.py':(2,'archived public Native graph updater inspect source'),
        'tree_target_replay.py':(2,'archived public Native graph target-run inspect source'),
        'canonical_eager.py':(2,'archived public Native preflight inspect source'),
        'original_native_protocol_v75.py':(6,'archived public Native decode inspect source plus current transforms'),
        'lifecycle_actor_v75.py':(2,'current archived v63_runtime_reference.py controlled admission AST'),
        'cpu_default_device_cleanup_host.py':(1,'retained Torch torch.utils._device from frozen external/factory closure'),
        'supervise_combined_v113.py':(1,'literal archived resume run_baseline_calibration-v1.py process helper'),
    }
    counts=Counter(Path(row['source']).name for row in dynamic)
    need(counts==Counter({name:value[0] for name,value in origins.items()}),'dynamic local source sites differ from reviewed34-site closure')
    for row in dynamic:row['reviewed_input_origin']=origins[Path(row['source']).name][1]
    old_py=[path for path in legacy if path.endswith('.py')]
    literal_old_exec=[]
    for call in dynamic:
        literal_old_exec.extend(dict(path=path,**call) for path in old_py if repr(path) in call['expression'] or '"'+path+'"' in call['expression'])
    counterparts=[]
    for path in old_py:
        local=P/Path(path).name
        counterparts.append(dict(historical_path=path,local_counterpart=str(local) if local.is_file() else None,
            local_resolved_in_conservative_import_closure=local.name in seen,
            historical_path_in_current_factory_source_closure=False))
    return dict(static_local_modules=len(seen),all_statically_resolved_local_modules_archived=True,
        local_import_search_root=str(P),source_closure_sha256=digest(seen),
        dynamic_source_sites=dynamic,dynamic_local_source_sites_reviewed=34,
        dynamic_source_inputs_resolve_to_current_archived_or_public_archived_sources_by_static_analysis=True,
        historical_Python_inputs=counterparts,
        direct_literal_historical_exec_sites=literal_old_exec,
        runtime_import_event_history_retained=False,actual_nonexecution_of_all_historical_inputs_independently_observed=False,
        boundary='Static source/import resolution and exact factory caller closure are checked. The run has no import-event log; SHA-only historical references are not asserted to be dynamically unexecuted merely because they have local counterparts.')


def audit_sources(receipt,mods):
    cpu_path=P/'CPU_RESULT_V119.json';freeze_path=P/'PREPARED_SOURCE_HASHES_V119.json'
    need(sha(cpu_path)==CPU_SHA and sha(freeze_path)==FREEZE_SHA,'CPU/freeze pinned epoch')
    cpu,freeze=read(cpu_path),read(freeze_path)
    need(cpu['status']=='passed' and cpu['tests_run']==187 and cpu['failures']==cpu['errors']==0
        and cpu['initialization_attempts']==[] and cpu['NPU_initialized'] is cpu['model_initialized'] is False
        and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files']
        and freeze['CPU_sha256']==CPU_SHA and len(freeze['files'])==273 and len(freeze['external_files'])==3508,'actual187 HOST source proof')
    expected={**freeze['files'],**freeze['external_files'],str(cpu_path):CPU_SHA,str(freeze_path):FREEZE_SHA}
    for path,wanted in expected.items():need(sha(path)==wanted,'actual frozen input changed: '+path)
    need(mods['source_snapshot_v113'].collect(P)==freeze['files'],'current static/dynamic source collector differs')
    binding=read(P/'ORIGINAL_PRIOR_BINDING_V78.json')['observed']
    external={**binding,**freeze['external_files']}
    need(receipt['source_hashes']==expected and receipt['external_input_hashes']==external,'actual supervisor source/legacy maps')
    for path,wanted in external.items():need(sha(path)==wanted,'post-run historical/external binding changed: '+path)
    prepared_path=P/'host-archive/V119_FINAL_CPU_SOURCE.tar.gz'
    need(sha(prepared_path)==ARCHIVE_SHA,'prepared full archive pinned bytes')
    prepared=archive(prepared_path,expected);actual=archive(receipt['source_snapshot'],expected)
    need(actual['sha256']==receipt['source_snapshot_sha256'],'actual source tar SHA')
    prepath=D/'ROOT_V119_PRE_NPU_SOURCE_AND_HOST_AUDIT.json';need(sha(prepath)==PRE_SHA,'ROOT prelaunch audit SHA')
    pre=read(prepath)
    need(pre['status']=='passed' and pre['CPU_sha256']==CPU_SHA and pre['freeze_sha256']==FREEZE_SHA
        and pre['archive_sha256']==ARCHIVE_SHA and pre['readiness_exit_code']==0
        and pre['NPU_started'] is pre['model_initialized'] is False,'real ROOT prelaunch/source/readiness proof')
    runtime=BASE/'serial-v1/runtime.sh';sourced=re.findall(r'^source\s+(/[^\s]+)',runtime.read_text(),re.M)
    shells={str(runtime):sha(runtime),**{path:sha(path) for path in sourced}}
    need(len(sourced)==2 and shells==pre['runtime_shells'],'actual retained CANN/ATB source shell identities')
    for side in ('pre','post'):
        verifier=receipt[side+'_verifier'];log=P/f'{TAG}-{side}-verifier.log'
        need(verifier['exit_code']==0 and verifier['argv'][0:3]==['bash',str(runtime),'python']
            and log.is_file() and log.stat().st_size>0,'retained actual verifier command/log missing')
    config=read(P/'COMBINATION_V113.json');factory=config['full_pipeline_factory_source_proof']
    need(factory['sha256']==FACTORY_SHA and sha(factory['path'])==FACTORY_SHA,'actual full target factory proof identity')
    proof=read(factory['path'])
    need(proof['source_dataflow_closed_for_V119_actual_target_pipeline'] is True
        and proof['global_factory_equivalence_proved'] is False
        and all(path in expected and expected[path]==wanted for path,wanted in proof['source_sha256'].items()),'full caller scope or archive coverage overstated')
    for path,wanted in proof['source_sha256'].items():need(sha(path)==wanted,'factory source changed: '+path)
    component=config['component_actual_evidence']['V110'];audit=read(component['external_audit']);old=read(component['receipt'])
    need(sha(component['receipt'])==component['receipt_sha256'] and sha(component['external_audit'])==component['external_audit_sha256']
        and old['status']=='complete' and old['errors']==[] and audit['status']=='passed' and audit['errors']==[],
        'required actual target-only CPU component proof')
    historical={path:w for path,w in binding.items() if path not in expected}
    resolution=local_source_resolution(freeze,historical)
    need(not resolution['direct_literal_historical_exec_sites'],'literal historical Python execution absent from source tar')
    helper_path='/root/SpecSLO/resume-20261012/run_baseline_calibration-v1.py'
    need(helper_path in expected,'actual dynamically loaded process-helper source not archived')
    return dict(CPU=desc(cpu_path),freeze=desc(freeze_path),local_sources=273,external_sources=3508,
        prepared_archive=prepared,actual_archive=actual,ROOT_prelaunch=desc(prepath),runtime_shells=shells,
        factory_proof=desc(factory['path']),V110_actual_receipt=desc(component['receipt']),V110_actual_audit=desc(component['external_audit']),
        current_static_and_factory_execution_source_bytes_archived=True,
        historical_additional_SHA_only_inputs=len(historical),historical_additional_Python_inputs=sum(p.endswith('.py') for p in historical),
        all_historical_prior_input_bytes_in_current_archive=not bool(historical),
        historical_inputs_source_map_sha256=digest(historical),static_execution_source_resolution=resolution,
        runtime_import_event_history_available=False,all_possible_dynamic_execution_sources_independently_observed=False)


def cleanup(receipt,directory):
    need(receipt['status'] in ('complete','failed') and receipt['child'].get('exit_code') is not None,'receipt must be terminal')
    need(receipt['physical_cards']==[2,5,6,7] and receipt['owned_remaining']==[]
        and receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True,'physical mapping/source/owned cleanup')
    for side in ('before','after'):
        state=receipt['devices_'+side]
        parsed=[(int(c),name) for c,name in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',state['hardware']['output'],re.M) if not name.isdecimal()]
        need(sorted(parsed)==[(c,'910B2') for c in (2,5,6,7)] and state['four_910B2'] is state['all_idle'] is True
            and state['hardware']['exit_code']==0 and not state['hardware']['timed_out']
            and set(state['processes'])=={'2','5','6','7'} and all(v['exit_code']==0 and v['idle'] is True
            and re.search(r'No process in device\.|No running processes found',v['output'],re.I) for v in state['processes'].values()),'retained hardware/idle evidence')
    exitpath=directory/'worker-exit.json';exited=read(exitpath)
    need(exited['closed'] is exited['all_exited'] is True,'original workers not all closed')
    return dict(passed=True,worker_exit=desc(exitpath),historical_pre_post_four_cards_idle=True,owned_remaining=[],actual_child_exit_code=receipt['child']['exit_code'])


def output_metrics(trace,rows):
    results=trace['generate_results'];need(type(results) is list and len(results)==len(rows),'leader result scope')
    delivered=[[] for _ in rows];first=[None]*len(rows);last=[None]*len(rows);finished=[False]*len(rows);previous=-1.
    for event in trace['stream_events']:
        i=event['request_index'];t=event['elapsed_seconds'];tokens=event['token_ids']
        need(type(i) is int and 0<=i<len(rows) and event['request_id']==rows[i]['request_id']
            and type(tokens) is list and tokens and all(type(x) is int and x>=0 for x in tokens)
            and type(t) in (int,float) and math.isfinite(t) and t>=previous and not finished[i]
            and type(event['finished']) is bool,'actual stream owner/token/clock/terminal sequence')
        if first[i] is None:first[i]=t
        last[i]=t;previous=t;finished[i]=event['finished'];delivered[i].extend(tokens)
    wall=trace['host_generate_call_seconds'];need(wall==trace['elapsed_seconds'] and wall>=previous>0,'host timing scope')
    metrics=[]
    for i,(request,result) in enumerate(zip(rows,results,strict=True)):
        need(len(delivered[i])==256 and finished[i] and delivered[i]==result['completion_token_ids']
            and result['request_id']==request['request_id'],'stream/result complete256 identity')
        paper,observed=result['paper_tpot_ms'],result['observed_tpot_ms']
        need(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in (paper,observed))
            and math.isclose(paper,observed*255/256,rel_tol=1e-10,abs_tol=1e-8),'Native original denominator relationship')
        slo=request['slo_tpot_ms'];need(slo==SLOS[request['slo_class']],'manifest SLO')
        need(result['paper_slo_attained'] is (paper<=slo)
            and result['paper_slo_goodput_tokens']==(256 if paper<=slo else 0),'Native attainment arithmetic')
        elapsed=(last[i]-first[i])*1000
        metrics.append(dict(index=i,request_id=request['request_id'],slo_class=request['slo_class'],slo_ms=slo,
            native_paper_ms=paper,native_observed_ms=observed,event_wall_paper_ms=elapsed/256,event_wall_observed_ms=elapsed/255,
            first_event_seconds=first[i],last_event_seconds=last[i],event_wall_minus_native_ms=elapsed/256-paper))
    def aggregate(group):
        def clock(name):
            good=sum(v[name]<=v['slo_ms'] for v in group)
            return dict(attained=good,good_tokens=256*good,goodput_tokens_per_second=256*good/wall,
                median_tpot_ms=statistics.median(v[name] for v in group),max_tpot_ms=max(v[name] for v in group))
        return dict(requests=len(group),native_paper=clock('native_paper_ms'),event_wall_paper=clock('event_wall_paper_ms'),
            event_wall_observed=clock('event_wall_observed_ms'))
    return dict(**aggregate(metrics),elapsed_seconds=wall,native_inner_elapsed_seconds=trace['native_elapsed_seconds'],
        raw_tokens_per_second=256*len(rows)/wall,by_class={c:aggregate([v for v in metrics if v['slo_class']==c]) for c in SLOS if any(v['slo_class']==c for v in metrics)},
        median_event_wall_minus_native_ms=statistics.median(v['event_wall_minus_native_ms'] for v in metrics),
        request_metrics=metrics,output_sha256=digest(delivered),raw_Native_finish_start_clocks_retained=False,
        Native_reported_TPOT_arithmetic_independently_recomputed=True,Native_TPOT_independently_reconstructed_from_raw_finish_start=False,
        stream_event_wall_independently_recomputed=True,formal_Goodput=False)


def proposal_metrics(records,envelopes):
    ordinary=records[1]['normal_target_calls'];mixed=[e for e in records[1]['logical_events'] if e['kind']=='plan']
    all_b=[b for call in ordinary for b in call['budgets']]+[b for event in mixed for b in event['budgets']]
    need(all(type(b) is int and 1<=b<=5 for b in all_b),'actual B outside1..5')
    histogram={str(b):all_b.count(b) for b in range(1,6)}
    actual=dict(target_branching_proposals=histogram['5'],target_linear_proposals=sum(histogram[str(b)] for b in range(1,5)),
        **{f'actual_target_B{b}':histogram[str(b)] for b in range(1,6)})
    for row in envelopes:
        raw=row['actual_proposal_counts'];need(raw['delta']==actual and {k:raw['after'][k]-raw['before'][k] for k in actual}==actual,'original per-trace raw proposal Counter delta')
    full=sum(max(c['budgets'])==5 for c in ordinary)
    for trace in records[1:]:
        counters=trace['decode_counters']
        need(counters['spec_rhythm_target_full_tree_rounds']==full+len(mixed)
            and counters['spec_rhythm_target_causal_tree_rounds']==len(ordinary)-full,'actual target FULL/causal round counters disagree with calls')
    eager_keys=('spec_rhythm_tree_eager_promoted','spec_rhythm_tree_eager_invalidated','spec_rhythm_tree_eager_dependency_rejections')
    eager=[{k:trace['decode_counters'][k] for k in eager_keys} for trace in records]
    need(all(row==eager[0] for row in eager) and all(type(n) is int and n>=0 for n in eager[0].values()),'actual eager aggregate role counters')
    return dict(selected_proposals=len(all_b),B1_to_B5=histogram,branching=actual['target_branching_proposals'],linear=actual['target_linear_proposals'],
        branching_proposal_fraction=histogram['5']/len(all_b) if all_b else None,
        ordinary_calls=len(ordinary),ordinary_FULL_calls=full,ordinary_causal_calls=len(ordinary)-full,
        mixed_FULL_calls=len(mixed),mixed_N_counts=dict(Counter(str(len(e['target_ids'])) for e in mixed)),
        all_target_call_FULL_fraction=(full+len(mixed))/(len(ordinary)+len(mixed)) if ordinary or mixed else None,
        ordinary_C96_calls=sum(sum(c['budgets'])==96 for c in ordinary),ordinary_N_gt16_calls=sum(c['N']>16 for c in ordinary),
        all_branching_from_B5_is_source_bound_nested_backbone_rule=True,
        actual_eager_original_counters=eager[0],eager_eligibility_need_window_observed=False)


def same_m(chains,reports,mods,directory):
    scopes=mods['local_qualification_scope'].SCOPES;signatures=[];all_rows=[];nonces=[]
    validator=mods['bounded_report_audit_v78']
    for rank,chain in enumerate(chains):
        signatures.append(validator.chain_record(chain,'draft' if rank==0 else 'target',rank=rank))
        need(len(chain['scopes'])==21 and sum(len(row['phases']) for row in chain['scopes'])==26,'actual21scope26phase')
        leases=[];phases=[]
        for row,scope in zip(chain['scopes'],scopes,strict=True):
            identity=row['reservation_identity'];leases.append(identity['nonce'])
            joins=validator.normal_reference_common_boundaries(row,scope)
            for phase in row['phases']:
                geometry=phase['geometry'];need(geometry['N']==scope.target_requests,'sameM scope N')
                if rank==0:validator.normal_reference_count_record(phase['draft_numeric'],scope)
                phase_path=directory/f"rank{rank}-qualification-{scope.name}-phase{phase['phase_index']}.json.gz"
                item=desc(phase_path)
                with gzip.open(phase_path,'rb') as stream:raw=stream.read(256*1024**2+1)
                need(len(raw)<=256*1024**2,'phase expansion bound')
                checkpoint=json.loads(raw);finite(checkpoint)
                need(checkpoint['rank']==rank and checkpoint['scope']==scope.name and checkpoint['phase_index']==phase['phase_index']
                    and checkpoint['complete_structure_qualification'] is False and checkpoint['completed_actual_phase']==phase,'retained original per-phase checkpoint differs')
                phases.append(dict(scope=scope.name,index=phase['phase_index'],N=geometry['N'],S=geometry['physical_S'],M=geometry['physical_M'],
                    prompt_start=geometry['prompt_start'],prompt_end=geometry['prompt_end'],final=geometry['final'],
                    artifact=item,normal_reference_common_joins=joins))
        need(len(set(leases))==21,'scope lease nonce reused')
        nonces.append(leases);outer=reports[rank]['mixed_same_M_qualification']['native']
        need(outer['delta']=={k:outer['after'][k]-outer['before'][k] for k in outer['before']},'sameM raw Native delta')
        mods['shape_scope_v101'].validate_inventory(chain['actual_inventory']['before'],chain['actual_inventory']['after'],is_draft=rank==0)
        all_rows.append(dict(rank=rank,scopes=21,phases=phases,lifecycle_cases=chain['lifecycle']['actual_case_count'],lease_nonces_unique=True,
            actual_layer_KV_gate_count=28 if rank==0 else 64))
    need(all(v==signatures[0] for v in signatures[1:]) and all(v==nonces[0] for v in nonces[1:]),'four-rank actual chain/lease identities differ')
    return dict(passed=True,ranks=all_rows,shared_once_in_original_control=True,
        lease_nonce_and_declared_allocation_epoch_observed=True,independent_engine_epoch_object_ID_separately_serialized=False,
        independent_initial_KV_source_gate_executed_by_source_bound_actor=True,offline_hidden_KV_tensor_recomparison_available=False,
        arbitrary_input_or_own_AR_qualification=False)


def run(tag):
    global TAG
    TAG=tag;receipt_path=P/f'local-receipt-{tag}.json';receipt=read(receipt_path)
    need(receipt['status'] in ('complete','failed') and 'devices_after' in receipt,'ROOT terminal notification required; running attempt is not audited')
    directory=P/'local-output'/tag
    with modules() as mods:
        provenance=audit_sources(receipt,mods);clean=cleanup(receipt,directory)
        base=dict(receipt=desc(receipt_path),original_run_status=receipt['status'],original_run_errors=receipt['errors'],
            source_provenance=provenance,cleanup=clean,torch_imported='torch' in sys.modules,NPU_initialized=False,
            raw_tensor_numerical_recomparison=False,formal_Goodput=False,independent_own_AR_qualified=False)
        if receipt['status']!='complete':
            errors=[desc(p) for p in sorted(directory.glob('rank*-last-error.json'))]
            phases=[desc(p) for p in sorted(directory.glob('rank*-qualification-*-phase*.json.gz'))]
            return dict(base,status='failed_attempt_evidence_audited',all_workload_and_qualification_complete=False,
                error_artifacts=errors,retained_completed_phase_artifacts=phases,numerical_qualification_granted=False,performance_result=None)
        need(receipt['errors']==[] and receipt['child']['exit_code']==0 and receipt['local_execution_complete'] is True,'original complete status gates')
        paths=[directory/f'rank{rank}-local-diagnostic.json' for rank in range(4)];reports=[read(p) for p in paths]
        completed=read(directory/'completed.json');need(completed['replies']==receipt['completed_replies']==[['calibrated',r,dict(rows=0,timings=0)] for r in range(4)],'four original completion replies')
        helper=D/'analyze_v96_actual_capacity64.py';need(sha(helper)==STARTUP_HELPER_SHA,'source-bound startup checker changed')
        previous=list(sys.path);sys.path.insert(0,str(D))
        try:startup_checker=importlib.import_module('analyze_v96_actual_capacity64')
        finally:sys.path[:]=previous
        startup=startup_checker.audit_startup(dict(enumerate(reports)),mods,P)
        formula=startup_checker.production_tpot_contract(P)
        # Frozen source validators re-run the complete actual chain/transactions;
        # independent calculations below separately rebuild clocks and proposals.
        frozen_audit=mods['audit_combined_v113'].audit_worker_reports(reports,MANIFEST,directory)
        need(same(frozen_audit,completed['actual_capacity64_audit']),'producer completed audit differs from frozen HOST re-evaluation')
        chains=[uncompressed(r['mixed_same_M_qualification']['artifact'],directory) for r in reports]
        local=same_m(chains,reports,mods,directory)
        schedule=[(arm,role) for arm in ('original','combined') for role in ROLES[:-1]]+[(arm,'full463') for arm in ('original','combined')]
        envelopes={(arm,role):[r['arms'][arm][ROLES.index(role)] for r in reports] for arm,role in schedule}
        traces={key:[uncompressed(e['artifact'],directory) for e in value] for key,value in envelopes.items()}
        metrics={arm:[] for arm in ('original','combined')};generation_rows=[];previous_native=[report['mixed_same_M_qualification']['native']['after'] for report in reports];full={}
        for arm,role in schedule:
            records=traces[arm,role];count=463 if role=='full463' else 65
            rows,prefix=mods['audit_production_v101'].manifest_rows(MANIFEST,count)
            for rank,(trace,envelope) in enumerate(zip(records,envelopes[arm,role],strict=True)):
                need(trace['rank']==rank and trace['role']==role and trace['native']==envelope['native'],'raw trace role/rank/counter envelope')
                delta=mods['audit_production_v101'].native_delta(trace,zero=role!='warmup65')
                need(len(delta)==21,'all21 original Native counters required')
                if previous_native[rank] is not None:need(trace['native']['before']==previous_native[rank],'Native changes hidden between persistent traces')
                previous_native[rank]=trace['native']['after']
                allocator=trace['allocator']
                need(trace['fresh_allocation'] is True and trace['cache_released'] is True and trace['class_delegate_restored'] is trace['wrappers_restored'] is True
                    and allocator['allocation_storage_rows']==[527] and allocator['pool_blocks']==544
                    and 0<allocator['referenced_page_highwater']<=544 and allocator['final_referenced_pages']==0
                    and allocator['acquired_pages']==allocator['released_pages'],'fresh generation allocator/highwater/release')
                need(trace['sealed_before'] is trace['sealed_after'] is (role=='full463'),'one unified seal before two full runs')
            metric=output_metrics(records[1],rows);branches=proposal_metrics(records,envelopes[arm,role])
            frozen_output=frozen_audit['arms'][arm][ROLES.index(role)]['output']
            need(math.isclose(metric['native_paper']['goodput_tokens_per_second'],frozen_output['goodput_tokens_per_second'],rel_tol=1e-12)
                and math.isclose(metric['event_wall_paper']['goodput_tokens_per_second'],frozen_output['existing_stream_event_wall']['goodput_tokens_per_second'],rel_tol=1e-12),'independent SLO/clock arithmetic disagrees')
            metrics[arm].append(dict(role=role,metrics=metric,branches=branches,
                Native_zero_delta_by_rank=[r['native']['delta'] for r in records],artifacts=[e['artifact'] for e in envelopes[arm,role]]))
            generation_rows.append(dict(arm=arm,role=role,by_rank=[r['source_generation'] for r in records]))
            if role=='full463':full[arm]=records
        for rank,report in enumerate(reports):
            need([r['by_rank'][rank] for r in generation_rows]==list(range(report['source_generation_before_phases']+1,report['source_generation_before_phases']+9)),'eight fresh persistent generations')
            checkpoint_path=directory/f'rank{rank}-paired-checkpoint.json.gz'
            with gzip.open(checkpoint_path,'rb') as stream:checkpoint=json.loads(stream.read(256*1024**2+1))
            finite(checkpoint)
            need(checkpoint['rank']==rank and checkpoint['arms']==report['arms']
                and checkpoint['phase_control_evidence']==report['phase_control_evidence']
                and checkpoint['mixed_same_M_qualification']==report['mixed_same_M_qualification'],'last paired checkpoint/final report agreement')
        need(full['original'][0]['tokenized_inputs']==full['combined'][0]['tokenized_inputs']
            and full['original'][0]['sampling']==full['combined'][0]['sampling'],'paired full exact prompt/sampling scope')
        differences=[]
        for i,(a,b) in enumerate(zip(full['original'][1]['generate_results'],full['combined'][1]['generate_results'],strict=True)):
            first=next((j for j,(x,y) in enumerate(zip(a['completion_token_ids'],b['completion_token_ids'],strict=True)) if x!=y),None)
            if first is not None:differences.append(dict(request_index=i,first_difference=first,original=a['completion_token_ids'][first],combined=b['completion_token_ids'][first]))
        return dict(base,status='passed',all_workload_and_qualification_complete=True,producer_frozen_audit_recomputed=True,
            startup=startup,Native_TPOT_source_formula=formula,same_M=local,eight_generation_order=generation_rows,
            arms=metrics,full463={arm:metrics[arm][-1] for arm in metrics},
            paired_prompt_tokens_and_sampling_identical=True,paired_output_difference_requests=len(differences),paired_output_differences=differences,
            paired_output_equality_was_not_required_or_qualified=True,one_shared_S35_qualification_not_two_independent_cold_namespaces=True,
            original_completion=desc(directory/'completed.json'),final_reports=[desc(p) for p in paths])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default=TAG);parser.add_argument('--output',required=True)
    args=parser.parse_args();start=datetime.now(timezone.utc).isoformat();result=None
    try:
        result=run(args.tag);result['audit_errors']=[]
    except BaseException:
        result=dict(status='audit_failed',audit_errors=[traceback.format_exc()],formal_Goodput=False,NPU_initialized=False)
    result.update(schema='V119.external-independent-terminal-HOST-audit.v1',actual_audit_argv=list(sys.argv),
        actual_audit_python=sys.executable,actual_python_version=sys.version,runtime_wrapper_used=False,
        audit_started_UTC=start,audit_finished_UTC=datetime.now(timezone.utc).isoformat(),auditor_source=desc(__file__))
    with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],output=args.output,formal_Goodput=False)))
    return 0 if result['status'] in ('passed','failed_attempt_evidence_audited') else 1


if __name__=='__main__':raise SystemExit(main())
