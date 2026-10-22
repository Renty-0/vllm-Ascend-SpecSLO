"""External read-only HOST audit of an actual terminal V98 S35 attempt.

The frozen actor's numerical/transaction observations are checked separately
from local timing arithmetic and source/CPU/archive/cleanup provenance. This
does not qualify all N17..32 partitions, a workload, overlap or Goodput.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
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
SOURCE=BASE/'tree-mixed-s35-v98'
PRE=BASE/'budget-diagnostics/ROOT_V98_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
FREEZE_SHA='dfc2dd9c97b12545f3d2d16bbec0931ecc40e85cbb5a5fa4b678c187eef93b6c'
CPU_SHA='0bbafa86d8ec21296e62aedee607edffe3a4139c4f53f31fe147c1590f01ad6a'
ARCHIVE_SHA='ed9693765463f4d5b0e587820a5a835386fff1cdce9b3f7b2b1dca8b6ed2517a'
MAX_FILE=64*1024**2
MAX_EXPANSION=256*1024**2
TRACES=('warmup','zero1','zero2')
FORBIDDEN_SCOPE=('all_N17_to32_partitions_qualified','online_qualified','formal_Goodput',
                 'independent_own_AR_qualified')


class Rejected(ValueError):pass


def need(value,message):
    if not value:raise Rejected(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):value.update(chunk)
    return value.hexdigest()


def descriptor(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def finite_json(value):
    if type(value) is dict:
        for child in value.values():finite_json(child)
    elif type(value) is list:
        for child in value:finite_json(child)
    elif type(value) is float:need(math.isfinite(value),'Nonfinite actual JSON value')
    else:need(value is None or type(value) in (str,bool,int),'Non-JSON actual value')


def read_json(path):
    value=json.loads(Path(path).read_bytes());finite_json(value);return value


def read_gzip(pointer,directory):
    path=Path(pointer['path']).resolve();root=Path(directory).resolve()
    need(path.is_relative_to(root) and path.is_file(),'Actual artifact escaped run directory')
    need(0<path.stat().st_size<=MAX_FILE and path.stat().st_size==pointer['bytes']
         and sha(path)==pointer['sha256'],'Actual gzip SHA/bytes mismatch')
    with gzip.open(path,'rb') as stream:raw=stream.read(MAX_EXPANSION+1)
    need(len(raw)<=MAX_EXPANSION and len(raw)==pointer['uncompressed_bytes'],'Actual gzip expansion/length mismatch')
    value=json.loads(raw);finite_json(value);return value


def recover_gzip(path,directory):
    item=descriptor(path)
    with gzip.open(path,'rb') as stream:raw=stream.read(MAX_EXPANSION+1)
    need(len(raw)<=MAX_EXPANSION,'Raw recovery gzip exceeds expansion bound')
    item['uncompressed_bytes']=len(raw)
    return item,read_gzip(item,directory)


class HostOnly(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
            raise Rejected('External HOST auditor forbids model/device import: '+fullname)
        return None


@contextmanager
def validators(source=SOURCE):
    previous=list(sys.path);guard=HostOnly()
    sys.path.insert(0,str(source));sys.meta_path.insert(0,guard)
    names=('audit_s35_v98','bounded_report_audit_v78','local_qualification_scope',
           'finite_inventory_plan','startup_receipts_v75')
    try:
        modules={}
        for name in names:
            module=importlib.import_module(name)
            need(Path(module.__file__).resolve().parent==Path(source).resolve(),'Wrong frozen validator path: '+name)
            modules[name]=module
        yield modules
    finally:
        sys.path[:]=previous;sys.meta_path.remove(guard)


def archive(path,expected):
    names={str(Path(name)).lstrip('/'):wanted for name,wanted in expected.items()}
    need(len(names)==len(expected),'Archive normalized input collision')
    with tarfile.open(path,'r:gz') as handle:
        members=handle.getmembers()
        need(len(members)==len(names) and len({m.name for m in members})==len(names)
             and {m.name for m in members}==set(names) and all(m.isfile() for m in members),
             'Archive exact regular-member coverage differs')
        for member in members:
            need(hashlib.sha256(handle.extractfile(member).read()).hexdigest()==names[member.name],
                 'Archive member actual bytes differ: '+member.name)
    return descriptor(path)|dict(regular_members=len(names),all_member_bytes_match=True)


def fusion(path):
    need(0<Path(path).stat().st_size<=MAX_FILE,'Fusion artifact size invalid')
    pairs=json.loads(Path(path).read_text(),object_pairs_hook=lambda rows:rows)
    need(type(pairs) is list and pairs,'Fusion top-level structure invalid')
    allowed={'TransdataCastFusionPass','TransdataFz2FzgFusionPass','TransdataFzg2FzFusionPass'}
    for key,value in pairs:
        need(re.fullmatch(r'session_and_graph_id_\d+_\d+',key) and type(value) is list
             and len(value)==1 and value[0][0]=='graph_fusion','Unexpected fusion graph structure')
        passes=value[0][1]
        need(type(passes) is list and passes and len({k for k,_ in passes})==len(passes),
             'Unexpected fusion pass inventory')
        for name,counters in passes:
            need(name in allowed and type(counters) is list and len(counters)==2
                 and {k for k,_ in counters}=={'effect_times','match_times'}
                 and all(type(v) is int and v>=0 or type(v) is str and re.fullmatch(r'\d+',v)
                         for _,v in counters),'Unexpected fusion pass payload')
    return descriptor(path)|dict(kind='actual-CANN-graph-fusion-counts',graph_records=len(pairs))


def source_inventory(source,expected):
    names=read_json(source/'STATIC_INPUTS_V98.json')['files']
    need(names==sorted(set(names)) and len(names)==214 and 'STATIC_INPUTS_V98.json' in names
         and {str((source/name).resolve()) for name in names}==set(expected),'Explicit static214 inventory differs')
    allowed=set(names)|{'CPU_RESULT_V98.json','PREPARED_SOURCE_HASHES_V98.json'}
    runtime=[]
    for path in source.iterdir():
        if not path.is_file() or path.suffix not in ('.py','.sh','.json') or path.name in allowed:continue
        if path.name=='fusion_result.json':runtime.append(fusion(path))
        elif path.name.startswith('CPU_ATTEMPT_') and path.suffix=='.json':
            need(read_json(path).get('schema')=='V98.actual-HOST-source-and-contract-tests.v1','Unknown HOST attempt schema')
        elif re.fullmatch(r'local-receipt-[A-Za-z0-9][A-Za-z0-9._-]*\.json',path.name):
            value=read_json(path)
            need(value.get('schema')=='V98.mixed-S35-supervisor.v1'
                 and value.get('action')=='mixed-s35-qualify-cost','Unknown runtime receipt schema')
        else:raise Rejected('Unknown unbound static input: '+str(path))
    return runtime


def source_closure(source,receipt):
    fp,cp=source/'PREPARED_SOURCE_HASHES_V98.json',source/'CPU_RESULT_V98.json'
    need(sha(fp)==FREEZE_SHA and sha(cp)==CPU_SHA,'Pinned V98 freeze/CPU bytes changed')
    freeze,cpu,pre=read_json(fp),read_json(cp),read_json(PRE)
    need(pre['all_checks_passed'] is True and pre['freeze_sha256']==FREEZE_SHA
         and pre['CPU_sha256']==CPU_SHA and pre['archive_sha256']==ARCHIVE_SHA,'Actual ROOT prelaunch identity differs')
    need(len(freeze['files'])==214 and len(freeze['external_files'])==1830 and freeze['CPU_sha256']==CPU_SHA,
         'Actual214/1830 source closure differs')
    need(cpu['status']=='passed' and cpu['tests_run']==131 and cpu['failures']==cpu['errors']==0
         and cpu['NPU_initialized'] is False and cpu['initialization_attempts']==[]
         and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
         and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],
         'Actual131 CPU source-before-after result differs')
    for path,wanted in {**freeze['files'],**freeze['external_files']}.items():
        need(sha(path)==wanted,'Actual source/external SHA mismatch: '+path)
    runtime=source_inventory(source,freeze['files'])
    prepared=dict(freeze['files'])|dict(freeze['external_files'])|{str(fp):FREEZE_SHA,str(cp):CPU_SHA}
    attempts=[]
    for path,pointer in pre['retained_attempts'].items():
        need(sha(path)==pointer['sha256'],'Original retained CPU attempt changed')
        actual=read_json(path)
        need(all(actual.get(key)==pointer[key] for key in ('status','tests_run','errors','failures')),
             'Original CPU attempt status was rewritten')
        prepared[path]=pointer['sha256'];attempts.append(descriptor(path)|dict(status=actual['status']))
    need(len(attempts)==2 and len(prepared)==2048,'Prepared archive must retain both actual CPU attempts')
    path=Path(pre['archive']);need(sha(path)==ARCHIVE_SHA,'Pinned complete HOST archive changed')
    archives=[archive(path,prepared)]
    actual_snapshot=dict(freeze['files'])|dict(freeze['external_files'])|{str(fp):FREEZE_SHA,str(cp):CPU_SHA}
    if 'source_hashes' in receipt:
        need(receipt['source_hashes']==actual_snapshot,'Actual supervisor snapshot input map differs')
        archives.append(archive(Path(receipt['source_snapshot']),actual_snapshot))
        need(archives[-1]['sha256']==receipt['source_snapshot_sha256'],'Actual supervisor archive SHA mismatch')
    if 'external_input_hashes' in receipt:
        for path,wanted in receipt['external_input_hashes'].items():need(sha(path)==wanted,'Actual external after-SHA mismatch: '+path)
    runtime_shell=BASE/'serial-v1/runtime.sh'
    paths=re.findall(r'^source\s+([^\s]+)',runtime_shell.read_text(),re.M)
    before=pre['actual_sourced_runtime_shells_before']
    need(len(paths)==2 and set(paths)==set(before),'Actual runtime source paths differ from real prelaunch record')
    shells=[]
    for path in paths:
        actual=sha(path);need(actual==before[path],'Actually sourced shell changed: '+path)
        shells.append(dict(path=path,resolved_after=str(Path(path).resolve()),before_sha256=before[path],after_sha256=actual))
    return dict(passed=True,CPU=descriptor(cp),freeze=descriptor(fp),local_files=214,external_files=1830,
        archives=archives,retained_actual_attempts=attempts,actual_prelaunch_audit=descriptor(PRE),
        actual_runtime_shells=shells,actual_runtime_outputs=runtime,unknown_static_inputs_rejected=True)


def genuine_inputs(source):
    fixture=read_json(source/'V98_GENUINE_INPUTS.json')
    need(fixture['schema']=='v98-genuine32-normal-prompts-and-original2301-prompt-v1'
         and len(fixture['trees'])==32 and len(fixture['prompt']['token_ids'])==2301,
         'Genuine32/original2301 scope differs')
    checkpoint=fixture['original_V88_checkpoint'];prior=fixture['original_V45_prompt']
    need(sha(checkpoint['path'])==checkpoint['sha256'] and sha(prior['path'])==prior['sha256'],
         'Original actual token-binding sources changed')
    with gzip.open(checkpoint['path'],'rb') as stream:raw=stream.read(MAX_EXPANSION+1)
    need(len(raw)<=MAX_EXPANSION,'Old actual checkpoint exceeds expansion bound')
    old=json.loads(raw);original=read_json(prior['path'])
    manifest=Path(fixture['manifest_path'])
    need(sha(manifest)==fixture['manifest_sha256'],'Original manifest changed')
    requests=[json.loads(line) for line in manifest.read_bytes().splitlines()]
    for position,tree in enumerate(fixture['trees']):
        index=fixture['manifest_indices'][position]
        actual=old['tokenized_inputs'][index]
        token_sha=hashlib.sha256(json.dumps(tree['token_ids'],separators=(',',':')).encode()).hexdigest()
        need(type(index) is int and tree['request_id']==requests[index]['request_id']
             and actual==dict(request_index=index,token_count=len(tree['token_ids']),token_sha256=token_sha)
             and tree['prompt_token_sha256']==token_sha
             and len(tree['token_ids'])==tree['context_length'] and tree['output_offset']==0
             and all(tree[key]==requests[index][key] for key in ('source_dataset','source_index','slo_class','slo_tpot_ms')),
             'Genuine normal prompt does not match original actual tokenized input')
    need(fixture['prompt']==original['prompt'] and fixture['prompt']['request_id']==requests[265]['request_id'],
         'Original genuine2301 prompt was changed or extended')
    for path,wanted in fixture['tokenizer']['resource_sha256'].items():need(sha(path)==wanted,'Pinned tokenizer resource changed')
    return dict(passed=True,normal_prompts=32,normal_token_SHA_equal_original_V88_actual=True,
        original_V88_checkpoint_retains_token_arrays=False,
        token_binding_method='Fixture complete token arrays hash and length exactly match original actual V88 per-owner tokenized-input SHA records; pinned manifest/tokenizer resources also unchanged',
        full_prompt_tokens=2301,full_prompt_equals_original_V45=True,fixture=descriptor(source/'V98_GENUINE_INPUTS.json'),
        checkpoint=descriptor(checkpoint['path']),prior=descriptor(prior['path']),
        source_token_binding_is_own_AR_equality=False)


def startup_inventory(reports,modules,source):
    plan=modules['finite_inventory_plan'];startup=modules['startup_receipts_v75']
    explicit={kind:set(map(repr,keys)) for kind,keys in plan.draft_keys().items()}
    generic=set(map(repr,startup.original_draft_family_keys()))
    target={row['expected_key'] for row in plan.targets()}
    query=[r['query_count'] for r in read_json(source/'prompt-shape-examples.json')['examples']]
    prefill=set(map(repr,startup.expected_prefill_keys(query)))
    summaries=[]
    for rank,report in sorted(reports.items()):
        value=report['startup'];keys=value['actual_native_keys']
        expected=dict(entries=set(),draft_entries=explicit['chain'],target_entries=explicit['hidden_PA']|explicit['frontier_PA']|generic) if rank==0 else dict(entries=set(),draft_entries=set(),target_entries=target|prefill)
        need({k:set(v) for k,v in keys.items()}==expected and all(len(v)==len(set(v)) for v in keys.values())
             and value['actual_native_entry_counts']=={k:len(v) for k,v in expected.items()},'Actual Native startup mapping inventory differs')
        family=value['original_draft_family'];count=28 if rank==0 else 0
        need(family['actual_added_qualified_entries']==count
             and family['original_native_method_executed'] is (rank==0)
             and family['native_qualification_flags_modified'] is False,'Original generic28 actual execution missing')
        delta={key:family['native_after'][key]-v for key,v in family['native_before'].items()}
        need(delta==family['native_delta'] and all(type(v) is int and v>=0 for v in delta.values()),'Original generic28 raw counters differ')
        for key in ('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays'):
            need(delta[key]==count,'Original generic28 qualification count differs')
        need(all(value['actual_native_counters'][key]==0 for key in startup.FAILURE_COUNTERS),'Actual startup failure counter nonzero')
        mc2=value['actual_MC2_preflight']
        need(mc2['rank']==rank and mc2['role']==('draft' if rank==0 else 'target')
             and mc2['all4rank_contract_passed'] is True and mc2['failed_checks']==[]
             and mc2['native_qualification_flags_modified'] is False
             and sorted(mc2['planned_full_chain_rows'])==sorted(plan.MC2_ROWS),'Actual startup MC2 preflight incomplete')
        if rank:
            need(mc2['manifest_route_count']==128 and mc2['disabled_routes']==[]
                 and mc2['routes_with_missing_or_unqualified_rows']==[],'Actual startup target128 MC2 routes incomplete')
        details=value['original_startup_receipts']
        if rank==0:
            row=details['draft']
            need(row['status']=='complete' and row['explicit_keys']==86 and row['original_native_flags_unchanged'] is True
                 and {k:set(v) for k,v in row['native_keys'].items()}==explicit,'Actual explicit86 receipt differs')
        else:
            p,t=details['prefill'],details['target']
            need(p['status']=='complete' and p['total']==p['completed']==201 and p['transition_checks_passed']==8
                 and t['status']=='complete' and t['fixtures']==44 and set(t['native_keys'])==target,
                 'Actual201/44 receipt incomplete')
        modules['bounded_report_audit_v78'].memory_record(value['actual_original_memory_finite_vote'],rank,'V80_original_startup')
        summaries.append(dict(rank=rank,startup_graph_entries=sum(len(v) for v in expected.values()),
            explicit_draft=86 if rank==0 else 0,generic_draft=28 if rank==0 else 0,
            original_prefill=201 if rank else 0,original_target_decode=44 if rank else 0))
    return dict(passed=len(reports)==4,ranks=summaries)


def phase_cost(cost,geometry,*,rank,trace_index,scope_index):
    need(cost['controlled_same_phase_repeated_state'] is True and cost['sequential_repeat_count']==5
         and cost['statistically_independent_samples_claimed'] is False
         and cost['production_overlap_qualified'] is False and cost['formal_Goodput'] is False,
         'Matched timing scope widened or independent-sample claim invented')
    rows=cost['records'];need(len(rows)==5,'Exactly five sequential timing repeats required')
    if rank==0:
        need(cost['role']=='draft' and rows==[None]*5,'Draft cannot supply target timing');return []
    need(cost['role']=='target','Target timing role differs')
    durations=[]
    for index,row in enumerate(rows):
        value=row['elapsed_ms']
        need(row['repeat']==index and type(value) in (float,int) and math.isfinite(value) and value>0,
             'Actual local timing must be positive and finite')
        need(row['physical_M']==197 and row['physical_S']==35
             and row['tree_queries']==geometry['tree_queries']
             and row['prompt_queries']==geometry['prompt_queries']
             and row['scratch_queries']==geometry['padding_queries']
             and row['head_rows']==geometry['target_head_rows'],'Timed geometry differs from actual phase')
        entry=row['entry']
        need(entry['key']==repr(('target-hidden:0|steps:1|tree-fia-full:exact-length-task-update|buffers:'
            '(((35, 1, 128, 4096), (35, 32)),)',(197,))) and entry['runtime_validated'] is True
            and entry['segments']==35 and entry['mask_shape']==[35,1,128,4096],
            'Timed call did not use the qualified S35 Native entry')
        need(row['sealed'] is (trace_index==2 and scope_index==2),'Unexpected timing seal placement')
        need(all(row[k] is True for k in ('hidden_and_all64_query_KV_original_matcher','exact_heads','committed_prefix_unchanged')),
             'Actual timed replay lacks complete numerical/prefix preservation')
        delta=row['native_delta']
        required={'total_calls','capture_replay_calls','replay_calls','eager_fallback_calls',
            'disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures',
            'changed_input_validation_calls','logical_row_expansion_validation_calls'}
        need(set(delta)==required and all(type(v) is int and v==int(k in ('total_calls','replay_calls'))
                                        for k,v in delta.items()),'Timed observation was not one validated resident replay')
        durations.append(value)
    return durations


def percentile(values,p=.9):
    values=sorted(values);need(values and all(type(v) in (float,int) and math.isfinite(v) for v in values),'Invalid percentile data')
    point=(len(values)-1)*p;lo,hi=math.floor(point),math.ceil(point)
    return values[lo]+(values[hi]-values[lo])*(point-lo)


def summarize_costs(bounded,modules):
    result=[];groups=defaultdict(list);sealed=total=0
    for rank,value in sorted(bounded.items()):
        chains=[value['qualification']]+[row['actor_chain'] for row in value['fixed_point']]
        need(len(chains)==3,'Warmup and two complete zero chains required')
        for trace_index,chain in enumerate(chains):
            need(len(chain['scopes'])==3 and sum(len(s['phases']) for s in chain['scopes'])==5,
                 'Actual three-scope/five-phase chain absent')
            for scope_index,scope in enumerate(chain['scopes']):
                for phase in scope['phases']:
                    geometry=phase['geometry'];cost=phase['V98_matched_target_cost']
                    need(cost['scope']==scope['scope'] and cost['phase_index']==phase['phase_index'],'Timing phase identity differs')
                    values=phase_cost(cost,geometry,rank=rank,trace_index=trace_index,scope_index=scope_index)
                    if not values:continue
                    for row in cost['records']:
                        modules['bounded_report_audit_v78'].mc2_call(row['actual_MC2'],reference=False,rank=rank,
                            native=dict(actual_entry=row['entry'],restored_entry=row['entry']))
                    key=(geometry['N'],geometry['tree_B'],geometry['prompt_queries'],geometry['final'])
                    groups[key].extend(values);total+=len(values)
                    sealed+=sum(row['sealed'] for row in cost['records'])
                    result.append(dict(rank=rank,trace=TRACES[trace_index],scope=scope['scope'],phase=phase['phase_index'],
                        N=geometry['N'],B=geometry['tree_B'],prompt_queries=geometry['prompt_queries'],
                        final=geometry['final'],M=197,S=35,head_rows=geometry['target_head_rows'],
                        milliseconds=values,median_ms=statistics.median(values),P90_ms=percentile(values),
                        sealed=cost['records'][0]['sealed']))
    aggregated=[dict(N=n,B=b,prompt_queries=p,final=f,observations=len(values),median_ms=statistics.median(values),
                    P90_ms=percentile(values),minimum_ms=min(values),maximum_ms=max(values))
                for (n,b,p,f),values in sorted(groups.items())]
    return dict(passed=total==225 and sealed==15,target_observations=total,sealed_target_observations=sealed,
        samples=result,by_geometry=aggregated,
        timer_scope='Actual metadata preparation + Native FULL197/S35 hidden replay + original common-vocabulary head + local device fence',
        excludes='Prefix/source prefill, independent reference, capture, evidence copy/compare, four-rank join, original commit, next draft',
        percentile_method='Linear interpolation at sorted zero-based (n-1)*0.9; median is sample median',
        sequential_same_state_samples_statistically_independent=False,production_overlap_qualified=False,formal_Goodput=False)


def compatible_numerical_audit(reports,bounded,modules):
    """Keep original numerical gates, replace only complete-boundary schema."""
    from actual_boundary_schema_host import isolated_chain_validator,audit_raw_boundaries
    original=modules['bounded_report_audit_v78']
    chain,provenance=isolated_chain_validator(original)
    tree=ast.parse(Path(original.__file__).read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_audit_reports')
    namespace=dict(vars(original),chain_record=chain)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
        original.__file__+':external-original-report-checks','exec'),namespace)
    combined=[dict(reports[r],bounded_qualification=bounded[r]) for r in range(4)]
    before=hashlib.sha256(json.dumps(combined,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    checked=namespace['_audit_reports'](combined)
    scopes=modules['local_qualification_scope'].scopes();rows=[];common={}
    for rank,value in sorted(bounded.items()):
        chains=[value['qualification']]+[row['actor_chain'] for row in value['fixed_point']]
        for trace_index,raw in enumerate(chains):
            for scope_index,(record,scope) in enumerate(zip(raw['scopes'],scopes,strict=True)):
                result=audit_raw_boundaries(record,scope,'draft' if rank==0 else 'target',rank)
                common.setdefault((trace_index,scope_index),[]).append(result['common_signature_sha256'])
                rows.append(dict(trace=TRACES[trace_index],**result))
    need(all(len(values)==4 and len(set(values))==1 for values in common.values()),
         'Actual complete common boundary sequences disagree across ranks')
    after=hashlib.sha256(json.dumps(combined,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(before==after,'External HOST numerical audit changed actual evidence')
    return dict(passed=True,original_substantive_audit=checked,raw_boundary_observations=rows,
        narrow_schema_adaptation=provenance,original_report_check_function_AST_unchanged=True,
        raw_chain_records_unchanged=True,all_N17_to32_partitions_qualified=False,
        whole_workload_AR_qualified=False,online_qualified=False,formal_Goodput=False)


def inventory_transition(report,new_key):
    rank=report['rank'];before,after=report['inventory_before'],report['inventory_after']
    need(set(before)==set(after)=={'entries','draft_entries','target_entries'},'Actual three mapping families required')
    for family in before:
        need(len(before[family])==len(set(before[family])) and len(after[family])==len(set(after[family]))
             and set(before[family])<=set(after[family]) and set(after[family])-set(before[family])==
             ({new_key} if rank!=0 and family=='target_entries' else set()),'More than the one allowed S35 hidden key was added/removed/aliased')
    return dict(rank=rank,before={k:len(v) for k,v in before.items()},after={k:len(v) for k,v in after.items()},
        new_hidden_keys=0 if rank==0 else 1,key=None if rank==0 else new_key)


def hbm_observations(source,reports,bounded,modules):
    results=[]
    for rank in range(4):
        path=source/'startup-progress'/f'finite-inventory-rank{rank}.jsonl'
        rows=[json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
        for row in rows:
            finite_json(row);modules['bounded_report_audit_v78'].memory_record(row,rank,row['phase'])
        if rank in bounded:
            value=bounded[rank]
            modules['bounded_report_audit_v78'].memory_record(value['after_qualification_memory'],rank,'V98-after-qualification')
            modules['bounded_report_audit_v78'].memory_record(value['after_seal_memory'],rank,'V98-after-seal')
        results.append(dict(rank=rank,progress=descriptor(path) if path.exists() else None,records=len(rows),
            phase_counts=dict(Counter(row['phase'] for row in rows)),
            npu_reserved_bytes_max=max((r['memory']['npu_reserved_bytes'] for r in rows),default=None),
            npu_peak_allocated_bytes_max=max((r['memory']['npu_peak_allocated_bytes'] for r in rows),default=None),
            npu_free_bytes_min=min((r['memory']['npu_free_bytes'] for r in rows),default=None),
            graph_entries_max=max((r['memory']['graph_entries'] for r in rows),default=None),
            after_qualification_memory=bounded.get(rank,{}).get('after_qualification_memory'),
            after_seal_memory=bounded.get(rank,{}).get('after_seal_memory')))
    return dict(passed=all(r['records']>0 for r in results),ranks=results,
        original_gates=dict(max_torch_reserved_GiB=54,max_torch_peak_GiB=54,min_device_free_GiB=6,max_graph_entries=512),
        qualification_memory_delta_is_not_attributed_exclusively_to_one_graph=True)


def raw_hardware_idle(value):
    if not isinstance(value,dict):return False
    info=value.get('hardware',{});raw=info.get('output','')
    parsed=[(int(i),name) for i,name in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',raw,re.M) if not name.isdecimal()]
    processes=value.get('processes',{})
    return bool(info.get('exit_code')==0 and not info.get('timed_out')
        and sorted(parsed)==[(i,'910B2') for i in (2,5,6,7)] and value.get('all_idle') is True
        and set(processes)=={'2','5','6','7'}
        and all(r.get('exit_code')==0 and not r.get('timed_out') and r.get('idle') is True
            and re.search(r'No process in device\.|No running processes found',r.get('output',''),re.I)
            for r in processes.values()))


def cleanup(receipt,directory):
    ep,cp=directory/'worker-exit.json',directory/'completed.json'
    exited=read_json(ep) if ep.exists() else None;completed=read_json(cp) if cp.exists() else None
    checks=dict(original_child_exit_zero=receipt.get('child',{}).get('exit_code')==0,
        original_workers_exited=bool(exited and exited.get('closed') is True and exited.get('all_exited') is True),
        original_completion=bool(completed and completed.get('replies')==[['calibrated',r,dict(rows=0,timings=0)] for r in range(4)]
            and completed.get('action')=='mixed-s35-qualify-cost'),
        pre_verifier_zero=receipt.get('pre_verifier',{}).get('exit_code')==0,
        post_verifier_zero=receipt.get('post_verifier',{}).get('exit_code')==0,
        actual_four_physical_cards=receipt.get('physical_cards')==[2,5,6,7],
        raw_idle_before=raw_hardware_idle(receipt.get('devices_before')),raw_idle_after=raw_hardware_idle(receipt.get('devices_after')),
        owned_processes_cleared=receipt.get('owned_remaining')==[],
        experimental_sources_unchanged=receipt.get('experimental_sources_unchanged') is True,
        external_sources_unchanged=receipt.get('external_sources_unchanged') is True)
    return dict(passed=all(checks.values()),checks=checks,failed_checks=[k for k,v in checks.items() if not v],
        child=receipt.get('child'),worker_exit=exited,completed=descriptor(cp) if completed else None,
        actual_original_status=receipt['status'],original_errors=receipt.get('errors'),owned_remaining=receipt.get('owned_remaining'),
        devices_before=receipt.get('devices_before'),devices_after=receipt.get('devices_after'))


def expected_raw_artifacts(directory):
    directory=Path(directory)
    return [directory/name for rank in range(4) for name in
        ([f'rank{rank}-S35-bounded.json.gz']+
         [f'rank{rank}-{trace}-chain.json.gz' for trace in TRACES]+
         [f'rank{rank}-{trace}-scope{index}.json.gz' for trace in TRACES for index in range(3)])]


def analyze(tag,source=SOURCE):
    source=Path(source).resolve();directory=source/'local-output'/tag
    rp=source/f'local-receipt-{tag}.json';receipt=read_json(rp)
    need(receipt.get('schema')=='V98.mixed-S35-supervisor.v1' and receipt.get('status') in ('complete','failed'),
         'Wait for original actual terminal V98 receipt')
    result=dict(schema='ROOT.V98.actual-S35-independent-audit.v1',tag=tag,UTC=datetime.now(timezone.utc).isoformat(),
        original_status=receipt['status'],original_errors=receipt.get('errors'),original_receipt=descriptor(rp),
        execution=dict(argv=list(sys.argv),executable=sys.executable,python_version=sys.version,
            dont_write_bytecode=sys.dont_write_bytecode,model_device_imports_allowed=False,device_queries_performed=False),
        errors=[],artifacts=[],actual_local_S35_qualified=False,all_N17_to32_partitions_qualified=False,
        whole_workload_AR_qualified=False,independent_own_AR_qualified=False,online_qualified=False,formal_Goodput=False)
    def section(name,callback,retain=True):
        try:
            value=callback();result[name]=value if retain else dict(actual_file_read=True);return value
        except Exception as error:
            result[name]=dict(passed=False,error=str(error));result['errors'].append(dict(section=name,
                error=type(error).__name__+': '+str(error),traceback=traceback.format_exc()));return None
    closed=section('source_CPU_archive_environment',lambda:source_closure(source,receipt))
    section('genuine_source_inputs',lambda:genuine_inputs(source))
    section('cleanup',lambda:cleanup(receipt,directory))
    reports,bounded={},{}
    for rank in range(4):
        path=directory/f'rank{rank}-local-diagnostic.json'
        if path.exists():
            row=section('worker'+str(rank),lambda path=path:read_json(path),False)
            if row is not None:reports[rank]=row;result['artifacts'].append(descriptor(path)|dict(kind='original-worker-report'))
        path=directory/f'rank{rank}-S35-bounded.json.gz'
        if path.exists():
            def load(path=path,rank=rank):
                pointer=reports.get(rank,{}).get('bounded_artifact')
                if pointer is None:pointer,value=recover_gzip(path,directory)
                else:value=read_gzip(pointer,directory)
                result['artifacts'].append(pointer|dict(kind='original-bounded-chain',rank=rank,
                    descriptor_bound_to_original_worker=rank in reports));return value
            value=section('bounded'+str(rank),load,False)
            if value is not None:bounded[rank]=value
    # Raw per-scope files remain useful if the terminal worker report is absent.
    for rank in range(4):
        for trace_index,trace in enumerate(TRACES):
            chain_path=directory/f'rank{rank}-{trace}-chain.json.gz'
            chain=None
            def linked_chain(path=chain_path,rank=rank,trace_index=trace_index):
                pointer,value=recover_gzip(path,directory)
                if rank in bounded:
                    expected=bounded[rank]['qualification'] if trace_index==0 else bounded[rank]['fixed_point'][trace_index-1]['actor_chain']
                    need(value==expected,'Raw auxiliary chain differs from original bounded artifact')
                result['artifacts'].append(pointer|dict(kind='actual-auxiliary-chain',rank=rank,trace=trace,
                    content_linked_to_original_bounded=rank in bounded));return value
            if chain_path.exists():chain=section(f'chain_rank{rank}_{trace}',linked_chain,False)
            for scope_index in range(3):
                path=directory/f'rank{rank}-{trace}-scope{scope_index}.json.gz'
                if not path.exists():continue
                def linked_scope(path=path,chain=chain,index=scope_index):
                    pointer,value=recover_gzip(path,directory)
                    if chain is not None:need(value==chain['scopes'][index],'Raw scope differs from complete chain')
                    result['artifacts'].append(pointer|dict(kind='actual-auxiliary-scope',content_linked_to_chain=chain is not None))
                    return True
                section(f'scope_rank{rank}_{trace}_{scope_index}',linked_scope,False)
    expected=expected_raw_artifacts(directory)
    missing=[str(path) for path in expected if not path.is_file()]
    result['raw_artifact_inventory']=dict(expected_gzip_count=len(expected),
        present_gzip_count=len(expected)-len(missing),missing_paths=missing,
        complete_original_redundant_artifact_set=not missing,
        available_partial_scopes_retained_even_without_complete_chain=True)
    if missing:result['errors'].append(dict(section='raw_artifact_inventory',
        error='Incomplete original raw artifact set; available evidence retained',missing_paths=missing))
    result['available_original_worker_ranks']=sorted(reports);result['available_bounded_ranks']=sorted(bounded)
    if closed:
        with validators(source) as modules:
            section('actual_startup_inventory',lambda:startup_inventory(reports,modules,source))
            section('actual_HBM_checkpoints',lambda:hbm_observations(source,reports,bounded,modules))
            transitions=[]
            for rank,report in reports.items():
                value=section('actual_inventory_rank'+str(rank),lambda report=report:inventory_transition(report,modules['audit_s35_v98'].new_hidden_key()))
                if value is not None:transitions.append(value)
            result['one_new_S35_key_observations']=transitions
            if len(reports)==len(bounded)==4:
                section('original_frozen_complete_auditor',lambda:modules['audit_s35_v98'].audit_reports(
                    [reports[r] for r in range(4)],private_directory=directory))
                section('complete_numerical_transaction_boundary_supplement',lambda:compatible_numerical_audit(reports,bounded,modules))
                section('independent_actual_cost_samples',lambda:summarize_costs(bounded,modules))
            else:
                result['errors'].append(dict(section='original_completeness',error='Four original final workers/bounded chains absent; recovery does not fabricate them'))
                for rank,value in bounded.items():
                    for index,chain in enumerate([value['qualification']]+[t['actor_chain'] for t in value['fixed_point']]):
                        section(f'available_sameM_rank{rank}_{index}',lambda chain=chain,rank=rank:
                            modules['bounded_report_audit_v78'].chain_record(chain,'draft' if rank==0 else 'target',rank=rank))
    files=[path for path in source.rglob('*') if path.is_file()]
    storage=dict(bytes=sum(path.stat().st_size for path in files),maximum_file_bytes=max((p.stat().st_size for p in files),default=0),
                 private_limit_bytes=256*1024**2,file_limit_bytes=MAX_FILE)
    result['actual_private_storage']=storage
    if storage['bytes']>=storage['private_limit_bytes'] or storage['maximum_file_bytes']>MAX_FILE:
        result['errors'].append(dict(section='private_storage',error='Actual private/file limit exceeded'))
    result['actual_local_S35_qualified']=bool(receipt['status']=='complete' and not result['errors']
        and all(result.get(name,{}).get('passed') is True for name in ('source_CPU_archive_environment',
            'genuine_source_inputs','cleanup','actual_startup_inventory','actual_HBM_checkpoints','independent_actual_cost_samples'))
        and result.get('complete_numerical_transaction_boundary_supplement',{}).get('passed') is True)
    result['actual_raw_local_S35_numerical_proof_accepted']=result.get('complete_numerical_transaction_boundary_supplement',{}).get('passed') is True
    result['scope']='Only genuine N17B5 and N32B3 three local source scopes/five phases, complete same-M hidden/KV/head/history/changed-restored/commit/lifecycle proof and sequential local timing; no all-partition, workload, AR, production overlap or Goodput qualification'
    result['numerical_evidence_boundary']='Original source-bound actor matcher/exact-head observations, Native/MC2 counters and transaction records; full hidden/KV tensors were not retained for a new offline tensor comparison'
    result['self']=descriptor(Path(__file__))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'Invalid tag')
    need(sys.dont_write_bytecode,'Use retained Python3.11 -B')
    output=args.output or BASE/'budget-diagnostics'/f'ROOT_V98_ACTUAL_S35_{args.tag}_AUDIT.json'
    need(not output.exists(),'Refusing to replace an actual audit attempt')
    result=analyze(args.tag)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    output.chmod(0o600)
    print(json.dumps(dict(output=str(output),sha256=sha(output),original_status=result['original_status'],
        errors=len(result['errors']),actual_local_S35_qualified=result['actual_local_S35_qualified'],
        target_samples=result.get('independent_actual_cost_samples',{}).get('target_observations'),formal_Goodput=False)))
    return int(not result['actual_local_S35_qualified'])


if __name__=='__main__':raise SystemExit(main())
