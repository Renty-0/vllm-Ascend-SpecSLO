#!/usr/bin/env python3
"""Independent read-only V102 actual audit; no torch, model, subprocess or NPU.

Requires a genuinely terminal complete clean supervisor receipt. Does not turn
failed infrastructure or unfinished qualification into a completed experiment.
Only after all gates pass writes two new files in budget-diagnostics. Thirty
rank0 method calls form15 paired repeats across3 controlled contexts; the
other three ranks supply protocol evidence and have null cost, not extra trials.
"""
import argparse
import ast
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys
import tarfile
import textwrap

sys.dont_write_bytecode=True
BASE=Path('/root/SpecSLO/budget-calibration-20261015')
WORK=BASE/'tree-packed-eager01-cost-v102'
OUT=BASE/'budget-diagnostics'
FREEZE='PREPARED_SOURCE_HASHES_V102.json'
FREEZE_SHA='0165b07cdb674b1f2d635cde318853f0e9f3e90def1db8f661d1de50581d5c76'
CPU_SHA='5ed07fb0fc03bd8c64cc6442fe05856c1e5b74041bf5f7fce6a5c54f1f27f531'
PREAUDIT=OUT/'ROOT_V102_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
PREAUDIT_SHA='57a60de922376fcefcbf1cec79e05adc8a3fe261a1f96fac4839e1ba25e0d01a'
VOCAB=151936
METHODS=('original_fast4','packed_seed_tail3')
OUTCOMES=('observed_target','valid_promotion','parent_reject','same_token_changed_ancestor')
GATES=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
    'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',
    *(f'aclgraph_{role}_{field}' for role in ('generic','draft','target') for field in
      ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
FAILURES=tuple(k for k in GATES if any(s in k for s in ('failed_captures','fallback','disabled_entry','unclassified','validation_failures')))
EXECUTION_FIELDS={'total_calls','capture_replay_calls','replay_calls','eager_fallback_calls',
    'disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures',
    'changed_input_validation_calls','logical_row_expansion_validation_calls'}
MC2_FIELDS={'fused_attempt','fused_success','fallback','exception','full_fused_attempt','full_fused_success',
    'native_epilogue_attempt','native_epilogue_success','native_epilogue_chained_attempt','native_epilogue_chained_success','native_epilogue_chain_flush'}
MC2_ROWS=[40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197]
PROFILE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json')
PROFILE_SHA='ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c'
OPERATOR='tp3_native_matmul+allreduce_add_rmsnorm'
RUNTIME_SHELLS={
 '/root/data/specslo-cann-runtime/cann-9.0.0/set_env.sh':'f9c99ed3e6465e1dfa80caa24b759e6e993760f607ca0777bbe24f9e05f52ded',
 '/root/data/specslo-cann-runtime/nnal/atb/9.0.0/atb/set_env.sh':'8219b354c51441bca8f5ab3127522cd92161f97de0ffd95eee2a85812d689886'}
NUMERIC_FLAGS=('all_physical_hidden','exact_full_physical_encoded','exact_top2_confidence_roots',
    'argmax_frontier_exact','all_layer_query_KV','ledger','all_owner_source_prefix_bitwise',
    'reference_candidate_KV_never_used','original_whole_method_restored')
FUTURE_FLAGS=('passed','all_layer_query_KV','independent_perrow_ledger','no_candidate_KV_or_ledger_reference',
    'all_physical_first_and_tail_hidden','original_before_tail_real_first','immediate_first_snapshot',
    'exact_full_physical_encoded','whole_method_restored','independent_history_rebuilt',
    'per_method_known_prefix_before_missing_bitwise_protected')


def require(ok,label):
    if not ok:raise RuntimeError('V102 independent actual audit refused: '+label)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def true_fields(row,names,label):require(isinstance(row,dict) and all(row.get(k) is True for k in names),label)
def hexsha(value):return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def number(value,label,*,positive=False):
    require(type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0),label)
    return value


def stats(values):
    values=list(values);require(bool(values),'nonempty timing samples')
    for v in values:number(v,'finite positive draft latency',positive=True)
    return dict(samples_ms=values,median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values))


def bucket(value,maximum):
    require(type(value) is int and 0<value<=maximum,'physical bucket domain')
    return next(v for v in [v for v in (1,2,4,8,12,16,20,24,28,32,36,40,44,48,64) if v<maximum]+[maximum] if v>=value)


def cases():
    def c(p,r):return dict(prefix=p,normal_repairs=r,eager=1,selection=[0,1,2,3,4])
    return [c(p,[0]*7+[1]*6) for p in (127,128,255)]+[c(128,[0]*4+[1]*9),c(128,[0]*13),c(128,[0,1]*16)]


def cost_cases():return [c for c in cases() if c['prefix']==128 and sum(c['normal_repairs'])>0]
def protocol_cases():return [c for c in cases() if c['prefix']==128 and len(c['normal_repairs'])==13]
def label(c):return 'N%d-E%d-R%d-P%d'%(len(c['normal_repairs']),c['eager'],sum(c['normal_repairs']),c['prefix'])


def validate_case(c):
    require(c in cases(),'explicit independent bounded device fixture')
    n=len(c['normal_repairs']);e=c['eager'];raw=n+e+sum(c['normal_repairs'])
    require(1<=n<=32 and 1<=e<=32 and n+e<=64 and raw<=128,'actual arity/query domain')
    return dict(normal=n,eager=e,total=n+e,rawQ=raw,first_M=bucket(raw,128),head_M=bucket(n+e,64),
        draft_candidates=5*n+3*e,C96_service_feasible=5*n+3*e<=96,
        N32_allB5_is_local_boundary_not_production=(n==32))


def native_delta(before,after,*,zero=False):
    require(set(before)==set(after)==set(GATES),'all21 explicit original Native snapshots')
    require(all(type(v) is int and v>=0 for d in (before,after) for v in d.values()),'nonnegative integer Native snapshots')
    delta={k:after[k]-before[k] for k in GATES}
    require(all(v>=0 for v in delta.values()),'monotonic original Native counters')
    if zero:require(not any(delta.values()),'zero discovery/validation/fallback in measured trace')
    return delta


def no_failures(snapshot):
    native_delta(snapshot,snapshot)
    require(all(snapshot[k]==0 for k in FAILURES),'absolute Native failure fields remain zero')


def source_file(name):
    p=WORK/(name+'.py');require(p.is_file(),'explicit frozen HOST helper '+name);return p


def host_module(name):
    require(name in ('packed_scope_v102','native_counts_v102','packed_report_audit_v102'),'stdlib-only pinned HOST module')
    spec=importlib.util.spec_from_file_location(name,source_file(name));module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    require('torch' not in sys.modules,'external audit must never import torch')
    return module


class Pairs(list):pass


def fusion_output(path,*,started,completed):
    path=Path(path);require(path.name=='fusion_result.json' and path.is_file() and not path.is_symlink(),'exact regular runtime fusion name')
    raw=path.read_bytes();st=path.stat()
    require(0<len(raw)<=64*1024 and started<=st.st_mtime<=completed,'bounded fusion output in actual run interval')
    root=json.loads(raw,object_pairs_hook=Pairs);require(isinstance(root,Pairs) and 1<=len(root)<=128,'bounded fusion object pairs')
    allowed={'TransdataCastFusionPass','TransdataFz2FzgFusionPass','TransdataFzg2FzFusionPass'};observed=[]
    for key,record in root:
        require(isinstance(key,str) and re.fullmatch(r'session_and_graph_id_\d+_\d+',key),'known CANN session/graph key')
        require(isinstance(record,Pairs) and len(record)==1 and record[0][0]=='graph_fusion','only graph_fusion record')
        passes=record[0][1]
        require(isinstance(passes,Pairs) and 1<=len(passes)<=3 and len({k for k,v in passes})==len(passes) and {k for k,v in passes}<=allowed,'only known unique fusion passes')
        for name,counts in passes:
            require(isinstance(counts,Pairs) and len(counts)==2 and {k for k,v in counts}=={'effect_times','match_times'}
                and all((type(v) is int and v>=0) or (type(v) is str and re.fullmatch(r'0|[1-9][0-9]*',v)) for k,v in counts),'explicit nonnegative fusion counts')
        observed.append(dict(session_graph_key=key,graph_fusion={k:dict(v) for k,v in passes}))
    require(path.read_bytes()==raw and path.stat().st_mtime_ns==st.st_mtime_ns,'fusion unchanged while reading')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),mtime_ns=st.st_mtime_ns,
        object_records=len(observed),unique_session_graph_keys=len({r['session_graph_key'] for r in observed}),
        duplicate_top_level_keys_preserved=True,records=observed,source_input=False,writer_identity_inferred=False)


def top_level(work,bound,*,started,completed):
    outputs=[]
    for p in Path(work).iterdir():
        if not p.is_file() or p.name in bound:continue
        if p.name=='fusion_result.json':outputs.append(fusion_output(p,started=started,completed=completed))
        elif re.fullmatch(r'cost-receipt-packed-eager01-[A-Za-z0-9-]+\.json',p.name):continue
        elif p.suffix in ('.py','.sh','.json','.md','.pyc','.so','.yaml','.yml','.toml','.ini'):
            require(False,'unknown unbound top-level code/static: '+p.name)
    return outputs


def archive_sources(path,expected):
    result={}
    with tarfile.open(path,'r:gz') as tar:
        for member in tar.getmembers():
            require(member.isfile() and member.name==Path(member.name).name and member.name not in result,'flat unique regular source archive')
            result[member.name]=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
    require(result==expected,'exact archived file bytes and membership')
    return len(result)


def source_and_HOST(receipt):
    require(sha(WORK/FREEZE)==FREEZE_SHA and sha(WORK/'CPU_RESULT_V102.json')==CPU_SHA,'reviewed exact source/CPU identities')
    require(sha(PREAUDIT)==PREAUDIT_SHA,'ROOT actual pre-NPU audit identity')
    freeze=read(WORK/FREEZE);pre=read(PREAUDIT);cpu=read(WORK/'CPU_RESULT_V102.json')
    require(pre['all_checks_passed'] is True and pre['NPU_started'] is False and pre['readiness_exit_code']==0
        and pre['CPU_tests']==33 and pre['CPU_sha256']==CPU_SHA and pre['freeze_sha256']==FREEZE_SHA
        and pre['static_CPU_input_files']==86 and pre['static_files_including_CPU']==87 and pre['external_files']==620,'actual successful ROOT prelaunch evidence')
    expected_local=freeze['files']|{FREEZE:FREEZE_SHA};static={k:v for k,v in freeze['files'].items() if k!='CPU_RESULT_V102.json'}
    require(len(static)==86 and len(expected_local)==88 and len(freeze['external_files'])==620,'exact source count semantics')
    require(receipt['input_hashes']=={str(WORK/k):v for k,v in expected_local.items()}
        and receipt['external_input_hashes']==freeze['external_files'],'actual source/archive closure equals freeze')
    for name,wanted in expected_local.items():require(sha(WORK/name)==wanted,'unchanged local frozen '+name)
    for name,wanted in freeze['external_files'].items():require(sha(name)==wanted,'unchanged external/runtime '+name)
    require(cpu['schema']=='V102.ragged01-focused-HOST.v1' and cpu['status']=='passed' and cpu['tests_run']==33
        and cpu['failures']==cpu['errors']==[] and cpu['NPU_initialized'] is False and cpu['NPU_run'] is False
        and cpu['model_initialization_attempts']==cpu['NPU_initialization_attempts']==[]
        and cpu['modules']==['test_packed_scope_v102_host','test_packed_numeric_v102_host','test_packed_actor_v102_host','test_packed_supervisor_v102_host']
        and cpu['input_hashes']==cpu['input_hashes_after']==static
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],'actual passing unchanged HOST33 source/result evidence')
    readiness=json.loads(pre['readiness_stdout'])
    require(pre['readiness_argv']==['bash',str(BASE/'serial-v1/runtime.sh'),'--environment-only',
        '/root/miniconda3/envs/vllm-hust-dev/bin/python3.11','-B',str(WORK/'packed_readiness_v102.py')]
        and readiness==receipt['prerequisites'] and readiness['source_contract']['new_NPU_qualification_inherited'] is False
        and readiness['source_contract']['external_inputs']==620 and readiness['actual_HOST_tests']==33
        and readiness['final_source_manifest_sha256']==FREEZE_SHA and readiness['NPU_qualified'] is False,'actual original prelaunch readiness without re-running launch collector')
    for path,wanted in RUNTIME_SHELLS.items():
        require(freeze['external_files'][path]==pre['actual_sourced_runtime_shells_before'][path]==wanted
            and sha(path)==wanted,'both actually sourced CANN9.0 runtime shells')
    require(sha(PROFILE)==PROFILE_SHA and freeze['external_files'][str(PROFILE)]==PROFILE_SHA,'unchanged original MC2 profile')
    archive=Path(receipt['source_snapshot'])
    require(archive.parent==WORK/'source-snapshots' and sha(archive)==receipt['source_snapshot_sha256'],'actual source snapshot location/hash')
    archive_sources(archive,expected_local)
    reviewed=Path(pre['archive']);require(sha(reviewed)==pre['archive_sha256'],'ROOT reviewed prelaunch archive hash')
    archive_sources(reviewed,expected_local)
    require(read(WORK/'SOURCE_DELTA_V102.json')['Native_modified'] is False,'Native source was never edited')
    return freeze,pre,cpu,readiness


def source_protocol_hashes(external):
    matches=[Path(p) for p in external if Path(p).name=='native_engine.py' and 'vllm_ascend/spec_decode/pearl' in p]
    require(len(matches)==1,'one actual Native source in external closure')
    source=matches[0].read_text();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativePearlEngine')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_spec_rhythm_tree_decode')
    start=min([fn.lineno]+[d.lineno for d in fn.decorator_list]);lines=source.splitlines(keepends=True)
    original=textwrap.dedent(''.join(lines[start-1:fn.end_lineno]));canonical=original
    for name,cls_name in (('canonical_eager','DecodeRewrite'),('host_tree_transport','Rewrite')):
        tree=ast.parse(source_file(name).read_text())
        defs=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls_name or isinstance(n,ast.FunctionDef) and n.name=='patch_decode_source']
        require(len(defs)==2,'exact pure HOST canonical/transport AST definitions')
        ns=dict(ast=ast,Counter=Counter);exec(compile(ast.Module(body=defs,type_ignores=[]),str(source_file(name))+':HOST-AST','exec'),ns)
        canonical=ns['patch_decode_source'](canonical)
    require('torch' not in sys.modules,'protocol source parsing imported no torch')
    return dict(original_decode_sha256=hashlib.sha256(original.encode()).hexdigest(),
        canonical_decode_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        original_decorator_included=True,source_bound_native_protocol_adapter=True,
        per_protocol_record_function_hashes_persisted=False)


def static_mc2(row):
    true_fields(row,('passed',),'original MC2 static qualification')
    checks=('disk_raw_sha','normalized_manifest_canonical_sha','profile_operator','standalone_chained_flush',
        'all64layers','route_count128','routes_frozen','all3physical_rows_in_chained_flush_rows',
        'every_route_enabled_and_all3physical_rows_qualified')
    require(set(row['checks'])==set(checks) and all(row['checks'][k] is True for k in checks)
        and row['bad_routes']==[] and row['profile_raw_sha256']==PROFILE_SHA
        and row['profile_canonical_sha256']==row['manifest_canonical_sha256']=='67feca4c924ce95b252812d2db606a65a43942c472c429a71d12e5270442deb7'
        and row['chained_flush_rows']==MC2_ROWS and row['deferred_rows_diagnostic_only']==[]
        and row['deferred_read_done_chain_required'] is False,'all64 layers/128 frozen MC2 routes and original profile')


def target_geometry(child=False):
    return dict(N=1,B=3,real_Q=4,M=40,S=8,family='causal',scratch=[6,5,5,5,5,5,5]) if child else dict(N=14,B=5,real_Q=84,M=96,S=16,family='full9',scratch=[6,6])


def target_keys(g):
    s=g['S'];attention=('tree-fia-full:exact-length-task-update|buffers:'+repr((((s,1,9,4096),(s,32)),))
        if g['family']=='full9' else 'fia:exact-length-task-update|buffers:'+repr((((s,32),(s,32),(2048,2048)),)))
    return {kind:repr((f'target-{output}:{vocab}|steps:1|'+attention,(g['M'],)))
        for kind,output,vocab in (('greedy','greedy',VOCAB),('hidden','hidden',0))}


def raw_mc2(row,*,geometry=None,independent=False):
    true_fields(row,('passed',),'raw original MC2 passed')
    static_mc2(row['static_contract']);d=row['dispatch']
    true_fields(d,('passed',),'raw MC2 dispatch passed')
    before,after=d['before'],d['after']
    require(set(before)==set(after)==set(d['values'])==MC2_FIELDS,'all11 explicit MC2 counter fields')
    require(all(type(v) is int and v>=0 for snap in (before,after) for v in snap.values()),'nonnegative original MC2 snapshots')
    delta={k:after[k]-before[k] for k in before}
    require(delta==d['values'] and all(v>=0 for v in delta.values()) and delta['fallback']==delta['exception']==0,'raw MC2 subtraction/no fallback')
    require(type(d['native_success_required']) is bool,'explicit MC2 cold/reference qualification kind')
    if independent:require(d['native_success_required'] is True,'independent sameM eager requires native dispatch')
    if d['native_success_required']:require(delta['native_epilogue_success']>0,'positive MC2 cold/reference dispatch')
    mode='independent_eager'
    if not independent:
        require(geometry is not None and row['history']['geometry']==geometry,'actual sameM MC2 geometry')
        keys=target_keys(geometry);h=row['history']
        require(h['vocabulary']==VOCAB and h['keys']==keys and row['actual_greedy_key']==keys['greedy']
            and row['hidden_key']==keys['hidden'],'exact original Native target/hidden family keys')
        require(h['history_valid'] is True and row['checks'] and all(v is True for v in row['checks'].values()),'cold/replay history and all original task checks')
        cold=any(v['exists'] is False for v in h['entries'].values())
        require(h['cold_capture_required'] is cold and d['native_success_required'] is cold,'raw cold capture vs qualified replay')
        require(set(row['after_entries'])==set(h['entries'])=={'greedy','hidden'},'both Native inventory entries')
        for kind,value in row['after_entries'].items():
            true_fields(value,('exists','valid'),'actual original Native entry exists/qualified')
            require(type(value['object_id']) is int and value['object_id']>0 and value['checks'] and all(v is True for v in value['checks'].values()),'all64 task shapes/handles/events independent')
            prior=h['entries'][kind]
            if prior['exists']:require(prior['valid'] is True and prior['object_id']==value['object_id'],'qualified resident object identity unchanged')
        require(type(row['hidden_executed_in_interval']) is bool,'actual hidden execution scope')
        if cold:require(row['hidden_executed_in_interval'] is True,'cold target key backed by actual hidden qualification')
        require(row['hidden_key_evidence']==('actual_qualification_execution' if row['hidden_executed_in_interval'] else 'qualified_resident_inventory_only'),'real hidden execution vs inventory declaration')
        mode='cold_capture' if cold else 'qualified_replay'
    return dict(mode=mode,native_positive_dispatch=delta['native_epilogue_success'],fallback=0,exception=0)


def target_proof(proof,*,child=False):
    true_fields(proof,('original_allhidden_allKV_1e3','exact_tokens','native_changed_input_gate','all_owner_prefixes_protected'),'independent sameM all64 target hidden/KV/exact tokens')
    g=target_geometry(child);require(proof['geometry']==g,'actual new physical target layout')
    require(set(proof['MC2'])=={'candidate','fresh_eager_reference'},'candidate and independent MC2 qualification')
    raw_mc2(proof['MC2']['candidate'],geometry=g)
    raw_mc2(proof['MC2']['fresh_eager_reference'],independent=True)


def compaction(proof):
    true_fields(proof,('all_real_rows','all_layers','independent_source_KV','perrow_original_query_offset'),'all64 layers/original selected-branch independent compaction')
    require(proof['dense_AR_qualified'] is False,'sameM compaction does not claim dense AR')


def independent_mover(row,accepted):
    true_fields(row,('fresh_independent_source_only','clone_before_index_copy','candidate_KV_never_used_as_reference'),'independent parent mover snapshots source before any destination write')
    owners=row['per_owner'];require(len(owners)==14 and [o['owner'] for o in owners]==list(range(14)),'all14 independent parent owners')
    destinations=[]
    for o,path in zip(owners,accepted):
        nodes=[v for v in path if v>=0];src=o['source_slots'];dst=o['destination_slots']
        require(o['selected_nodes']==nodes and len(src)==len(dst)==len(nodes)+1 and all(type(v) is int and v>=0 for v in src+dst),'actual original accepted-branch source/destination rows')
        require(src==[src[0],*[src[0]+v+1 for v in nodes]] and dst==list(range(src[0],src[0]+len(nodes)+1)),'original query indices moved to contiguous committed prefix')
        destinations.extend(dst)
    require(len(set(destinations))==len(destinations),'independent owner destinations never alias')


def expected_native_keys(c,method):
    g=validate_case(c);n=g['total']
    if method=='packed_seed_tail3':
        return [('hidden',repr(('target-hidden:0|steps:1|paged',(g['first_M'],)))),
            ('draft',repr((f'draft-greedy:{VOCAB}|steps:3|paged|confidence',g['head_M'])))]
    repairs=sum(c['normal_repairs']);require(repairs>0,'original Fast control requires an actual repair stage')
    return [('hidden',repr(('target-hidden:0|steps:1|paged',(bucket(repairs,64),)))),
        ('greedy',repr((f'target-greedy:{VOCAB}|steps:1|paged',(bucket(c['eager'],64),)))),
        ('draft',repr((f'draft-greedy:{VOCAB}|steps:4|paged|confidence',g['head_M'])))]


def numeric_records(report):
    for field,points,method in (('numerical',cases(),'packed_seed_tail3'),('control_numerical',cost_cases(),'original_fast4')):
        rows=report[field]
        require([(r['case'],r['method'],r['changed_input_variant']) for r in rows]==[(c,method,v) for c in points for v in (False,True,False)],'independent original/changed/restored numerical schedule')
        for q in rows:
            c=q['case'];g=validate_case(c)
            true_fields(q,NUMERIC_FLAGS,'independent allphysical hidden/encoded/tokens/confidence/28KV/ledger/restoration')
            require(q['qualification_inherited'] is False and q['timing_qualified'] is False and q['prefix']==c['prefix']
                and q['normal']==g['normal'] and q['eager']==g['eager'] and q['changed']==q['changed_input_variant'],'true fresh source and unchanged numerical gate')
            require([(r['kind'],r['key']) for r in q['native_keys']]==expected_native_keys(c,method),'actual Native hidden/greedy/tail keys, not a pair whitelist')
            for key in q['native_keys']:
                require(key['runtime_validated'] is True,'actual Native changed-input flag')
                if key['kind']=='draft':require(type(key['validated_real_row_count']) is int and key['validated_real_row_count']>=g['total'],'all real owner rows qualified')
            if method=='packed_seed_tail3':
                true_fields(q,('original_before_tail_real_first','root_head_outside_graph','diagnostic_observer_scope_only',
                    'immediate_first_hidden_observer_numeric_only'),'real hidden snapshot before outside head and tail')
                require(q['first_M']==g['first_M'] and q['tail_M']==g['head_M'] and q['eager_uncomputed_indices']==[3,7]
                    and q['eager_aggregate_confidence_denominator']==6 and q['normal_aggregate_confidence_denominator']==8,'actual packed geometry and original eight-wide sparse ABI')
            else:
                true_fields(q,('frontier_capture_eager_storage_nonalias','diagnostic_observer_scope_only'),'independent original frontier storage')
                require(q['actual_method']=='V46-original-FastV5-repairPA-frontierPA-chain4','actual original Fast control, not packed replay')


def next_proof(row):
    true_fields(row,FUTURE_FLAGS,'actual nextdraft independent physical history/28KV/ledger/restoration')
    require(row['dense_new_prefix_prefill_used'] is False and row['candidate_reference_known_prefix_cross_bitwise_equality_claimed'] is False,'true physical history, no new dense/cross-reference bitexact claim')
    prefixes=row['prefixes'];starts=row['starts'];missing=row['missing_suffix_lengths']
    require(len(prefixes)==len(starts)==len(missing)==14 and all(type(p) is int and type(s) is int and p>=s>=128 and r==p-s for p,s,r in zip(prefixes,starts,missing)),'all14 real nextdraft per-owner repair extents')
    warm=row['native_warmup'];require(len(warm)==3 and [r['delta'] for r in warm]==[0,1,0],'original natural changed/restored nextdraft warmup')
    require(warm[0]['roots']==warm[2]['roots'] and warm[1]['roots']==[(r+1)%VOCAB for r in warm[0]['roots']]
        and len(warm[0]['roots'])==14,'actual nextdraft F/T/F roots')
    require(len({r['key'] for r in warm})==1 and len({r['root_mode'] for r in warm})==1,'same Native nextdraft tail key/mode across roots')
    for w in warm:
        true_fields(w,('source_history_physically_replayed',),'independent cold call restores original physical history')
        require(w['head_rows']==16 and type(w['root_mode']) is bool,'N14 actual physical head16')
    require(warm[-1]['runtime_validated'] is True and warm[-1]['validated_real_row_count']>=14,'original nextdraft Native all-real gate')
    initial=warm[0]['qualification_before_three_calls']
    if not (initial['runtime_validated'] is True and initial['validated_real_row_count']>=14):
        require(warm[0]['actual_seed_tokens']!=warm[1]['actual_seed_tokens'],'new tail key must receive changed actual seed')


def verdict(rows,accepted,count,*,child=False):
    require(len(rows)==len(accepted)==count,'all real output/accepted owner rows')
    for output,path in zip(rows,accepted):
        require(len(output)==5 and len(path)==4,'original public verdict shape')
        require(all(type(t) is int and -1<=t<VOCAB for t in output) and all(type(v) is int and -1<=v<=(2 if child else 4) for v in path),'actual token/accepted domains')
        tokens=[v for v in output if v>=0];nodes=[v for v in path if v>=0]
        require(output==tokens+[-1]*(len(output)-len(tokens)) and path==nodes+[-1]*(4-len(nodes)),'contiguous actual emitted/accepted prefix')
        require(1<=len(tokens)<=((3 if child else 4)+1) and len(tokens)==len(nodes)+1,'accepted path plus one actual target bonus')
        require(nodes==list(range(len(nodes))) or (not child and nodes==[4]),'genuine primary path or terminal side, not arbitrary accepted indices')


def protocol_records(report,rank):
    require([(q['case'],q['outcome']) for q in report['protocol']]==[(c,o) for c in protocol_cases() for o in OUTCOMES],'three genuine ragged contexts times four protocol outcomes')
    schedule=[]
    for q in report['protocol']:
        require(q['prefix']==128 and q['normal']==13 and q['eager']==1 and q['qualification_inherited'] is False,'new N13/E1 actual protocol scope')
        true_fields(q,('Native_original_finite_preflight_commit_promotion_guards',),'original Native finite/vote/commit/promotion')
        require(q['actual_method']=='V102-packedPA-normalB5-eagerB3-ragged01-tail3'
            and q['production_dual_home_overlap'] is False and q['dense_canonical_AR_qualified'] is False
            and q['cross_M_equality_qualified'] is False and q['acceptance_rate_gain_or_Goodput_measured'] is False
            and q['online_qualified'] is False,'bounded local protocol, not online/AR/Goodput')
        p=q['proposal_exchange'];true_fields(p,('original_WORLD_exchange','payload_ticket_identity_retained'),'original WORLD proposals retain payload/ticket identity')
        require(p['parent_rows']==14 and p['staged_child_rows']==1 and p['ordered_proposal_ids']==list(range(15)),'actual parent owners14 then staged child1')
        verdict(q['actual_observed_parent_output'],q['actual_observed_parent_accepted'],14)
        verdict(q['exercised_parent_output'],q['exercised_parent_accepted'],14)
        controlled=q['outcome']!='observed_target'
        require(q['parent_counterfactual_protocol_only'] is controlled,'actual observed parent retained separately from diagnostic controls')
        if controlled:
            require(q['actual_observed_parent_output'][:13]==q['exercised_parent_output'][:13]
                and q['actual_observed_parent_accepted'][:13]==q['exercised_parent_accepted'][:13],'controls modify only eager-parent row')
        else:require(q['actual_observed_parent_output']==q['exercised_parent_output'] and q['actual_observed_parent_accepted']==q['exercised_parent_accepted'],'observed outcome is never synthesized')
        if rank==0:
            require(q['parent_target'] is None and q['parent_compaction'] is None,'draft role does not invent target evidence')
            next_proof(q['actual_nextrepair'])
        else:
            require(q['actual_nextrepair'] is None,'target ranks do not invent draft oracle')
            target_proof(q['parent_target']);compaction(q['parent_compaction'])
            schedule.extend([('parent-candidate',q['parent_target']['MC2']['candidate'],target_geometry(),False),
                ('parent-reference',q['parent_target']['MC2']['fresh_eager_reference'],None,True)])
        child=q['real_promoted_child_consume'];promoted=q['promoted_owners']
        require(promoted in ([],[13]),'only the actual eager owner can be promoted')
        if q['outcome']=='valid_promotion':require(promoted==[13] and q['exercised_parent_accepted'][13]==[0,1,2,3],'controlled full genuine primary4 dependency promotion')
        if q['outcome']=='parent_reject':require(promoted==[] and q['exercised_parent_accepted'][13]==[-1]*4,'original parent rejection cleanup')
        if q['outcome']=='same_token_changed_ancestor':require(promoted==[] and q['exercised_parent_accepted'][13]==[4,-1,-1,-1],'same frontier cannot override changed ancestor')
        require((child is not None)==bool(promoted),'promoted child must be actually consumed')
        if child is None:continue
        true_fields(child,('actual_observed_only','normal_owner_prefixes_protected'),'real child target, original normal owners protected')
        require(child['consumed_owners']==[13],'original child consumption owns real eager request')
        verdict(child['observed_output'],child['observed_accepted'],1,child=True)
        if rank==0:
            require(child['target_proof'] is None and child['committed_KV'] is None and child['independent_parent_history']==[],'no target proof fabricated on draft rank')
        else:
            target_proof(child['target_proof'],child=True);compaction(child['committed_KV'])
            history=child['independent_parent_history'];require(len(history)==2,'independent then actual parent restoration epochs')
            require(history[0]['mode']=='fresh-independent-parent-and-own-index-copy','fresh parent reference uses independent index-copy')
            independent_mover(history[0],q['exercised_parent_accepted']);raw_mc2(history[0]['MC2'],independent=True)
            require(history[1]['mode']=='original-Native-parent-and-Native-compaction' and history[1]['reference_KV_copied_back_to_candidate'] is False,'actual parent+Native mover restore without reference KV copying')
            raw_mc2(history[1]['MC2'],geometry=target_geometry())
            schedule.extend([('promoted-child-candidate',child['target_proof']['MC2']['candidate'],target_geometry(True),False),
                ('child-independent-parent-replay',history[0]['MC2'],None,True),
                ('promoted-child-reference',child['target_proof']['MC2']['fresh_eager_reference'],None,True),
                ('child-original-parent-restoration',history[1]['MC2'],target_geometry(),False)])
    return schedule


def source_contexts(c):
    points=read(WORK/'inputs.json')['contexts'];categories=('coding','chat','summarization','coding')
    count=len(c['normal_repairs'])+c['eager']
    return [next(p for p in points if p['category']==categories[i%4] and len(p['token_ids'])>=264) for i in range(count)]


def float32(v):return struct.unpack('<f',struct.pack('<f',v))[0]


def public_signature(value,c,method,*,changed=False):
    g=validate_case(c);n=g['normal'];count=g['total'];P=c['prefix']
    require(set(value)=={'draft_token_ids','root_token_ids','eager_frontier_tokens','node_confidences','draft_confidence'},'all real public token/confidence ABI fields')
    rows=value['draft_token_ids'];scores=value['node_confidences'];roots=value['root_token_ids'];aggregate=value['draft_confidence']
    require(len(rows)==len(scores)==len(roots)==len(aggregate)==count and set(value['eager_frontier_tokens'])=={str(n)},'normal/eager owner rows and sparse frontier dictionary')
    require(all(type(t) is int and 0<=t<VOCAB for row in rows for t in row) and all(type(t) is int and 0<=t<VOCAB for t in roots),'real actual token domains')
    contexts=source_contexts(c)
    require(roots[:n]==[(int(contexts[i]['token_ids'][P+r])+int(changed))%VOCAB for i,r in enumerate(c['normal_repairs'])]
        and roots[n]==value['eager_frontier_tokens'][str(n)],'actual source normal roots and eager frontier seed')
    for i,(tokens,confidence,total) in enumerate(zip(rows,scores,aggregate)):
        require(len(tokens)==len(confidence)==8,'eight-wide original top2 ABI')
        for s in confidence:number(s,'finite node confidence');require(s<=1,'node confidence probability')
        quanta=[round(s*1_000_000) for s in confidence]
        require(all(float32(q/1_000_000)==s for q,s in zip(quanta,confidence)),'actual quantized float32 confidence ABI')
        divisor=6 if method=='packed_seed_tail3' and i>=n else 8
        require(total==float32(sum(q/1_000_000 for q in quanta)/divisor),'exact original aggregate confidence denominator')
        valid=3 if method=='packed_seed_tail3' and i>=n else 4
        if valid==3:require(tokens[3]==tokens[7]==0 and confidence[3]==confidence[7]==0,'uncomputed eager fourth slots are zero')
        for j in range(valid):require(tokens[j]!=tokens[j+4] and quanta[j]>=quanta[j+4],'actual distinct sorted top2 pair')


def cost_ledger(row,c,method):
    g=validate_case(c);n=g['normal'];count=g['total'];P=c['prefix'];out=row['signature'];ledger=row['final_known_ledger']
    require(set(ledger)=={str(i) for i in range(count)},'each actual owner retains independent known-KV ledger')
    for i,r in enumerate(c['normal_repairs']):
        p=P+r;state=ledger[str(i)];expected={str(p+j):t for j,t in enumerate([out['root_token_ids'][i],*out['draft_token_ids'][i][:3]])}
        require(state==dict(safe_before=p,tokens=expected),'actual normal root/primary3 ledger and committed boundary')
    state=ledger[str(n)];tokens=state['tokens'];length=8 if method=='packed_seed_tail3' else 9
    require(state['safe_before']==P and set(tokens)=={str(P+j) for j in range(length)},'uncommitted parent history retained, no nonexistent eager fourth KV')
    context=source_contexts(c)[n]
    require(tokens[str(P)]==(int(context['token_ids'][P])+int(row['changed']))%VOCAB,'true original B5 parent root retained')
    expected=[out['root_token_ids'][n],*out['draft_token_ids'][n][:2 if method=='packed_seed_tail3' else 3]]
    require([tokens[str(P+5+j)] for j in range(len(expected))]==expected,'actual eager frontier and generated primary tail ledger')
    require(all(type(t) is int and 0<=t<VOCAB for t in tokens.values()),'all actual parent/eager ledger tokens')


def execution_delta(row,method):
    before=row['execution_before'];after=row['execution_after']
    require(set(before)==set(after)=={'generic','target','draft'},'three genuine Native execution families')
    delta={}
    for family in before:
        a,b=before[family],after[family]
        require(set(a)==set(b)==EXECUTION_FIELDS and all(type(v) is int and v>=0 for d in (a,b) for v in d.values()),'all10 original execution fields')
        d={k:b[k]-a[k] for k in a};require(all(v>=0 for v in d.values()),'monotonic actual Native execution')
        calls=(1 if family=='draft' else (1 if method=='packed_seed_tail3' else 2) if family=='target' else 0)
        require(d['total_calls']==d['replay_calls']==calls and all(d[k]==0 for k in EXECUTION_FIELDS-{'total_calls','replay_calls'}),'only actual replay: one packed hidden+tail or original repair+frontier+tail')
        delta[family]=d
    return delta


def plain_row(row,rank,*,timed):
    c=row['case'];g=validate_case(c);m=row['method'];require(m in METHODS,'actual method name')
    require(type(row['changed']) is bool and row['actual_call'] is (rank==0)
        and row['target_or_whole_method_latency_ms'] is None and row['actual_delivered_g'] is None,'actual draft-only measurement ownership and nullable target/g fields')
    if rank:
        require(row['draft_latency_ms'] is None and row['signature'] is None and row['source_identity_sha256'] is None,'idle target rank has no cost/token sample')
        require(not {'Native_before','execution_before','final_known_ledger'}&set(row),'target ranks do not fabricate draft diagnostics')
        return None
    require(hexsha(row['source_identity_sha256']) and len(row['initial_source_prefix_hashes'])==56
        and all(hexsha(s) for s in row['initial_source_prefix_hashes']),'actual source identity and all28*2 KV prefix hashes')
    require(native_delta(row['Native_before'],row['Native_after'],zero=True)==row['Native_deltas']
        and native_delta(row['Native_after_fixture'],row['Native_after'],zero=True)==row['Native_timed_call_deltas'],'actual zero Native arithmetic across fixture and method')
    public_signature(row['signature'],c,m,changed=row['changed']);cost_ledger(row,c,m);d=execution_delta(row,m)
    q=g['first_M']+3*g['head_M'] if m=='packed_seed_tail3' else bucket(sum(c['normal_repairs']),64)+bucket(c['eager'],64)+4*g['head_M']
    calls=2 if m=='packed_seed_tail3' else 3
    require(row['actual_method_query_count']==q and row['actual_method_graph_calls']==row['actual_method_model_calls']==calls,'physical PA/head/tail cost footprint, no padded-query discount')
    if timed:
        require(row['changed'] is False and row['sealed'] is True,'unmodified original source only under original Native seal')
        number(row['draft_latency_ms'],'finite real draft latency',positive=True)
        require(row['untimed_replay_elapsed_ms'] is None,'one actual timer scope')
    else:
        require(row['draft_latency_ms'] is None,'zero traces do not masquerade as samples')
        number(row['untimed_replay_elapsed_ms'],'real untimed replay elapsed',positive=True)
    require(row['measured_method_includes']=='actual draft function: metadata, model graphs, outside head if any, original D2H and result packing; final NPU synchronize'
        and row['excluded_from_timer']=='source cache release/allocate/prefill, actual B5 eager parent, independent references, source/proof checks, rank coordination','actual source-bound timer boundaries')
    return d


def zero_and_cost(report,rank):
    no_failures(report['preseal_Native']);traces=report['fixed_point']
    require(len(traces)==2,'two complete measured-input zero traces')
    for sweep,trace in enumerate(traces):
        points=[(c,m) for c in cost_cases() for m in METHODS]
        if sweep:points.reverse()
        require(trace['sweep']==sweep and [(r['case'],r['method'],r['changed']) for r in trace['rows']]==[(c,m,ch) for c,m in points for ch in (False,True,False)],'all three measured cases, both real methods, F/T/F forward/reverse')
        require(native_delta(trace['Native_before'],trace['Native_after'],zero=True)==trace['Native_deltas'],'original all-rank full trace zero discovery')
        for row in trace['rows']:plain_row(row,rank,timed=False)
        if rank==0:
            for i in range(0,18,3):
                a,b,c=trace['rows'][i:i+3]
                require(a['source_identity_sha256']==c['source_identity_sha256']!=b['source_identity_sha256']
                    and a['signature']==c['signature'] and a['final_known_ledger']==c['final_known_ledger']
                    and a['initial_source_prefix_hashes']==b['initial_source_prefix_hashes']==c['initial_source_prefix_hashes'],'actual F/T/F source/output/ledger restoration with unchanged original prefix')
    rows=report['timing']
    expected=[(repeat,c,m) for repeat in range(5) for c in cost_cases() for m in (METHODS if repeat%2==0 else tuple(reversed(METHODS)))]
    require([(r['repeat'],r['case'],r['method']) for r in rows]==expected,'15 matched pairs/30 real methods in declared alternating order')
    for row in rows:plain_row(row,rank,timed=True)
    if rank==0:
        qualified={}
        for trace in traces:
            for r in trace['rows']:
                if not r['changed']:
                    key=(label(r['case']),r['method']);signature=(r['signature'],r['final_known_ledger'],r['source_identity_sha256'],r['initial_source_prefix_hashes'])
                    if key in qualified:require(qualified[key]==signature,'unobserved actual same-method fixed-point source signature')
                    qualified[key]=signature
        for r in rows:
            require(qualified[(label(r['case']),r['method'])]==(r['signature'],r['final_known_ledger'],r['source_identity_sha256'],r['initial_source_prefix_hashes']),'measured actual method retains own qualified source/output/ledger bridge')
        for i in range(0,30,2):
            a,b=rows[i:i+2]
            require(a['source_identity_sha256']==b['source_identity_sha256'] and a['initial_source_prefix_hashes']==b['initial_source_prefix_hashes'],'each paired real method rebuilds identical actual source/prefix KV')


def memory(row,rank):
    require(row['layers']==(28 if rank==0 else 64) and row['blocks']==544 and 60*1024**3<=row['total']<=64*1024**3
        and row['free']>=6*1024**3 and row['reserved']<=54*1024**3 and row['peak']<=54*1024**3 and row['entries']<=512,'actual original layer/page/memory/inventory bounds')


def reports_deep(reports,completed,directory=None):
    require(len(reports)==4 and completed['replies']==[['calibrated',rank,dict(rows=0,timings=30)] for rank in range(4)],'all4 original actual worker replies')
    mc2=[]
    for rank,r in enumerate(reports):
        require(r['schema']=='V102.ragged01-packed-qualification-and-local-draft-cost.v1' and r['rank']==rank and r['errors']==[],'real report identity/no local failures')
        true_fields(r,('sealed','graph_sealed','NPU_qualified_bounded_ragged01_samples_only','timing_qualified_local_draft_only'),'actual bounded qualification and original seal before cost')
        require(r['gain']==[] and r['actual_delivered_g'] is None and r['Goodput'] is None and r['production_dual_home_overlap'] is False and r['online_qualified'] is False and r['dense_AR_qualified'] is False,'no manufactured gain/Goodput/AR/production overlap')
        require((r['expected_packed_numerical'],r['expected_control_numerical'],r['expected_protocol'],r['expected_synchronized_cost_experiments'])==(18,9,12,30),'bounded planned record totals')
        warm=r['warmup'];points=[(c,'packed_seed_tail3') for c in cases()]+[(c,'original_fast4') for c in cost_cases()]
        require([(w['case'],w['method']) for w in warm]==points,'actual unpublished F/T/F warmup schedule before independent qualification')
        for w in warm:
            require(len(w['calls'])==3,'each actual warmup has original/changed/restored calls')
            for value,ch in zip(w['calls'],(False,True,False)):
                if rank:require(value is None,'target warmup does not invent a draft result')
                else:public_signature(value,w['case'],w['method'],changed=ch)
        numeric_records(r);schedule=protocol_records(r,rank);zero_and_cost(r,rank)
        memory(r['startup_memory'],rank);memory(r['final_memory'],rank)
        require(r['final_entry_states'] and all(e['runtime_validated'] is True for e in r['final_entry_states']),'all resident final target/draft entries genuinely qualified')
        no_failures({k:r['final_graph_metrics'][k] for k in GATES})
        if directory is not None and rank:
            paths=sorted(directory.glob(f'rank{rank}-MC2-*.json'))
            require(len(paths)==len(schedule),'all raw actual MC2 records; count derived from genuinely consumed children')
            last_dispatch=None
            for serial,(path,(suffix,expected,g,independent)) in enumerate(zip(paths,schedule,strict=True),1):
                require(path.name==f'rank{rank}-MC2-{serial:04d}-{suffix}.json' and read(path)==expected,'exact MC2 serial/schedule and raw report binding')
                observation=raw_mc2(expected,geometry=g,independent=independent)
                before=expected['dispatch']['before']
                if last_dispatch is not None:require(all(before[k]>=last_dispatch[k] for k in MC2_FIELDS),'raw MC2 chronological counters never reset between records')
                last_dispatch=expected['dispatch']['after']
                mc2.append(dict(path=str(path),sha256=sha(path),rank=rank,label=suffix,**observation))
    require(all(r['numerical']==reports[0]['numerical'] and r['control_numerical']==reports[0]['control_numerical'] for r in reports),'all-rank numerical evidence is the actual shared draft trace')
    fields=('case','outcome','proposal_exchange','actual_observed_parent_output','actual_observed_parent_accepted','exercised_parent_output','exercised_parent_accepted','parent_counterfactual_protocol_only','promoted_owners')
    witness=[[{k:q[k] for k in fields} for q in r['protocol']] for r in reports]
    require(all(w==witness[0] for w in witness),'all-rank exact observed/controlled verdict/promotion agreement')
    require('torch' not in sys.modules,'no torch imported during deep actual HOST audit')
    return mc2


def audit(tag):
    require(tag.startswith('packed-eager01-') and tag.replace('-','').isalnum(),'bounded exact diagnostic tag')
    receipt_path=WORK/f'cost-receipt-{tag}.json';receipt_bytes=receipt_path.read_bytes();receipt=json.loads(receipt_bytes)
    require(receipt.get('schema')=='V102.ragged01-packed-cost-receipt.v1' and receipt.get('status')=='complete','terminal complete supervisor receipt before any actual result audit')
    require(receipt['errors']==[] and receipt['physical_cards']==[2,5,6,7] and receipt['child']['exit_code']==0
        and receipt['child']['argv']==['bash',str(WORK/'launch_packed_cost_v102.sh'),tag]
        and receipt['pre_verifier']['exit_code']==receipt['post_verifier']['exit_code']==0,'actual child/launcher/original runtime lock verifiers')
    require(receipt['devices_before']['all_idle'] is True and receipt['devices_after']['all_idle'] is True and receipt['owned_remaining']==[],'owned processes cleaned and cards idle before/after')
    true_fields(receipt,('experimental_sources_unchanged','external_sources_unchanged','local_scope_complete'),'terminal source closure/local qualification')
    require(receipt['formal_Goodput'] is False and receipt['production_overlap_qualified'] is False and receipt['whole_workload_AR_equality_qualified'] is False,'local bounded scope only')
    require(number(receipt['elapsed_seconds'],'actual final child elapsed',positive=True)<2400 and number(receipt['private_bytes'],'actual bounded private bytes')<256*1024**2,'original supervisor time/storage bounds')
    freeze,pre,cpu,ready=source_and_HOST(receipt)
    started=datetime.datetime.fromisoformat(receipt['started_UTC']).timestamp();finished=receipt_path.stat().st_mtime
    runtime=top_level(WORK,set(freeze['files'])|{FREEZE},started=started,completed=finished)
    retained=receipt['known_runtime_outputs'];require(len(retained)==len(runtime),'all known runtime outputs retained by actual supervisor')
    for expected,observed in zip(retained,runtime):
        require(expected['path']==observed['path'] and expected['raw_sha256']==observed['sha256']
            and expected['bytes']==observed['bytes'] and expected['mtime_ns']==observed['mtime_ns']
            and expected['source_input'] is False and expected['writer_identity_inferred'] is False
            and expected['classification']=='known-CANN-format-runtime-output'
            and expected['parsed_structure']==read(expected['path']),'actual supervisor fusion metadata and independent raw object-pair witness')
    directory=WORK/'context-cost-sweep-v1'/tag
    require(read(directory/'prerequisites.json')==ready,'actual child readiness equals ROOT prelaunch contract')
    paths=[directory/f'rank{rank}-packed01-cost.json' for rank in range(4)]
    require(receipt['bounded_local_report_paths']==[str(p) for p in paths],'exact complete actual report locations')
    reports=[read(p) for p in paths];completed=read(directory/'completed.json')
    require(started<=completed['completed_unix']<=finished,'actual completed replies within supervisor interval')
    cfg=read(directory/'config.json')
    for key,wanted in dict(draft_tensor_parallel_size=1,target_tensor_parallel_size=3,max_num_seqs=64,max_num_queued_seqs=463,
        max_model_len=4096,num_kvcache_blocks=544,kvcache_block_size=128,max_aclgraph_entries=512,gamma=8,
        spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4,spec_rhythm_max_eager_tokens=3,
        target_use_paged_attention=True,draft_use_paged_attention=True,enable_mc2=True,
        spec_rhythm_linear_full_window=False,spec_rhythm_linear_eager_cross_graph_bucket=False,
        spec_rhythm_linear_idle_residual_eager=False,spec_rhythm_prefill_token_chunk_size=0,
        spec_rhythm_verification_budget=None,spec_rhythm_draft_token_budget=None).items():
        require(cfg[key]==wanted,'actual diagnostic config '+key)
    require(cfg['mc2_profile']==str(PROFILE) and cfg['draft_model_path']=='/root/data/shared-models/Qwen3-0.6B'
        and cfg['target_model_path']=='/root/data/shared-models/Qwen3-32B','same actual Qwen3 pair/original MC2 profile')
    require(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'original engine closed and all four workers exited')
    # Only execute the original stdlib report checker after all its exact source
    # bytes were checked. The deeper checks below are independent implementations.
    sys.path.insert(0,str(WORK));host_module('packed_scope_v102');host_module('native_counts_v102')
    checker=host_module('packed_report_audit_v102');original=checker.audit_reports(reports,completed)
    require(original==receipt['actual_report_audit'] and completed['replies']==receipt['completed_replies'],'frozen supervisor report audit independently recomputed')
    require(host_module('packed_scope_v102').cases()==cases(),'independent expected fixtures equal exact frozen bounded samples')
    mc2=reports_deep(reports,completed,directory)
    preflight=read(directory/'rank0-preflight.json');require(len(preflight)==4,'all4 actual preflight observations')
    evidence={str(p):sha(p) for p in [receipt_path,*paths,directory/'completed.json',directory/'config.json',
        directory/'prerequisites.json',directory/'worker-exit.json',WORK/FREEZE,WORK/'CPU_RESULT_V102.json',PREAUDIT]}
    for rank in range(4):
        p=directory/f'rank{rank}-preflight.json';require(read(p)==preflight,'same actual preflight all-gather on all ranks')
        memory(preflight[rank]['memory'],rank)
        if rank:static_mc2(preflight[rank]['MC2'])
        else:require(preflight[rank]['MC2'] is None,'draft has no invented MC2 target static record')
        evidence[str(p)]=sha(p)
        r=reports[rank]
        require(r['seal_status']['aclgraph_sealed']==1 and r['final_graph_metrics']['aclgraph_sealed']==1,'actual original Native seal reflected in metrics')
        for snap in (r['seal_status'],r['final_graph_metrics']):
            require(all(snap[k]==0 for k in ('aclgraph_unvalidated_entries','aclgraph_disabled_entries','aclgraph_pruned_unvalidated_entries')),'original all-resident qualification, no prune bypass')
            require(native_delta(r['preseal_Native'],{k:snap[k] for k in GATES},zero=True)=={k:0 for k in GATES},'seal and measured calls preserve all Native21 values')
        require(len(r['final_entry_states'])==r['final_graph_metrics']['aclgraph_target_entries']+r['final_graph_metrics']['aclgraph_draft_entries'],'actual final inventory cardinality')
    for row in mc2:evidence[row['path']]=row['sha256']
    for p in [WORK/f'cost-calibration-{tag}.log',WORK/f'{tag}-pre-verifier.log',WORK/f'{tag}-post-verifier.log']:
        require(p.is_file() and p.stat().st_size>0,'actual raw worker/verifier log retained');evidence[str(p)]=sha(p)
    for r in runtime:evidence[r['path']]=r['sha256']
    limits=read(WORK/'EAGER_GATE_LIMITS_V102.json')
    require(limits['V99_actual_shadow_executed'] is False and limits['N32_E1_allB5']['draft_candidates']==163
        and limits['N32_E1_allB5']['exceeds_original_candidate_roof96'] is True
        and all(limits[k] is None for k in ('actual_generated','actual_published','actual_controller_eager_candidates',
            'actual_gap_urgency_acceptance_gate_counts','actual_window_selection_counts','actual_need_or_grant_counts')),'missing online eager gates remain unknown; local N32 is not C96 service')
    require(receipt_path.read_bytes()==receipt_bytes,'actual terminal receipt unchanged while reading')
    require('torch' not in sys.modules,'independent external audit did not import torch')
    return dict(receipt=receipt,reports=reports,completed=completed,original_report_audit=original,
        raw_evidence=evidence,MC2=mc2,protocol_source=source_protocol_hashes(freeze['external_files']),
        runtime_outputs=runtime,preaudit=pre,HOST=cpu)


def signed_stats(values):
    values=list(values);require(values and all(type(v) in (int,float) and math.isfinite(v) for v in values),'finite signed paired deltas')
    return dict(values=values,median=statistics.median(values),min=min(values),max=max(values))


def signature_differences(a,b,c):
    n=len(c['normal_repairs']);differences=[];confidence=[]
    for owner,(x,y,sx,sy) in enumerate(zip(a['draft_token_ids'],b['draft_token_ids'],a['node_confidences'],b['node_confidences'])):
        indices=list(range(8)) if owner<n else [0,1,2,4,5,6]
        for pos in indices:
            if x[pos]!=y[pos]:differences.append(dict(owner=owner,original_node=pos,original_fast_token=x[pos],packed_token=y[pos]))
            if sx[pos]!=sy[pos]:confidence.append(dict(owner=owner,original_node=pos,original_fast=sx[pos],packed=sy[pos]))
    return dict(common_valid_node_token_differences=differences,common_valid_node_confidence_differences=confidence,
        expected_unused_packed_eager_slots=[3,7],cross_method_bitexact_required=False,
        actual_delivery_or_gain_equality_measured=False)


def summarize(reports):
    rows=reports[0]['timing'];summary=[]
    for c in cost_cases():
        by_method={m:[r for r in rows if r['case']==c and r['method']==m] for m in METHODS}
        require(all([r['repeat'] for r in v]==list(range(5)) for v in by_method.values()),'five actual paired repeat IDs')
        old,new=by_method['original_fast4'],by_method['packed_seed_tail3']
        g=validate_case(c)
        summary.append(dict(case=c,case_id=label(c),metadata_scope=g,
            draft_whole_method_ms={m:stats(r['draft_latency_ms'] for r in values) for m,values in by_method.items()},
            paired_packed_minus_original_ms=signed_stats(n['draft_latency_ms']-o['draft_latency_ms'] for o,n in zip(old,new)),
            paired_original_over_packed_ratio=signed_stats(o['draft_latency_ms']/n['draft_latency_ms'] for o,n in zip(old,new)),
            packed_lower_latency_pairs=sum(n['draft_latency_ms']<o['draft_latency_ms'] for o,n in zip(old,new)),
            actual_physical_query_count={m:values[0]['actual_method_query_count'] for m,values in by_method.items()},
            source_identity_sha256=old[0]['source_identity_sha256'],
            actual_output_differences=signature_differences(old[0]['signature'],new[0]['signature'],c),
            target_latency_ms=None,serving_round_latency_ms=None,actual_delivered_g=None,
            service_C96_valid=(g['draft_candidates']<=96),N32_boundary_only=(g['normal']==32)))
    return dict(schema='V102.actual-paired-local-draft-cost-summary.v1',paired_contexts=3,paired_repeats_per_context=5,
        actual_rank0_draft_method_calls=30,paired_comparisons=15,cost_participating_rank=0,
        target_rank_latency_is_null=True,four_rank_protocol_observations_are_not_four_independent_trials=True,
        repeated_contexts_and_repeats_statistical_independence_not_established=True,
        source_prefill_parent_construction_independent_reference_and_checks_outside_timer=True,
        timer='actual draft function through final NPU synchronize, including real metadata/head/D2H/result packing',
        measured_packed_route='one ragged PA + outside head + chain3',
        measured_original_control='original repairPA + frontierPA + chain4',
        cases=summary,all28_query_KV_and_ledger_qualified_before_timing=True,
        all64_target_KV_and_original_protocol_qualified_separately_not_timed=True,
        target_or_whole_serving_round_latency_measured=False,actual_delivered_g=None,
        online_eager_generation_publication_promotion_measured=False,N32_B5_C163_not_C96_service=True,
        independent_own_AR_qualified=False,production_overlap_qualified=False,formal_Goodput=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True);args=parser.parse_args()
    try:
        actual=audit(args.tag);summary=summarize(actual['reports']);receipt=actual['receipt']
        summary_path=OUT/'V102_ACTUAL_PACKED_EAGER01_DRAFT_COST.json';audit_path=OUT/'ROOT_V102_ACTUAL_PACKED_EAGER01_COST_AUDIT.json'
        require(not summary_path.exists() and not audit_path.exists(),'refuse overwriting any retained actual result')
        result=dict(schema='ROOT.V102.independent-actual-packed01-audit.v1',audited_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            passed=True,receipt_status='complete',receipt=str(WORK/f'cost-receipt-{args.tag}.json'),
            source_freeze_sha256=FREEZE_SHA,CPU_sha256=CPU_SHA,ROOT_prelaunch_audit_sha256=PREAUDIT_SHA,
            static_source_files_with_CPU=87,runtime_archive_members=88,external_source_files=620,
            actual_HOST_tests=33,original_report_audit_recomputed=actual['original_report_audit'],
            actual_numeric_packed_records_per_rank=18,actual_numeric_original_Fast_controls_per_rank=9,
            actual_protocol_records_per_rank=12,actual_all28_draft_and_all64_target_KV_ledger_commit_nextdraft_gates=True,
            actual_target_parent='N14/B5 FULL9 M96/S16,84 real queries,2*6 scratch',actual_child='E1/B3 causal M40/S8',
            two_complete_measured_F_T_F_zero_traces_per_rank=True,Native_fields_per_snapshot=21,
            zero_discovery_validation_failure_and_fallback_preserved_through_original_seal_and_cost=True,
            actual_cost_pairs=15,actual_rank0_draft_method_calls=30,target_rank_cost_values=None,
            rank_observations_are_not_independent_trials=True,N32_B5_C163_not_online_C96_scope=True,
            MC2_raw_records=actual['MC2'],MC2_raw_record_count=len(actual['MC2']),
            original_MC2_profile_sha256=PROFILE_SHA,original_protocol_source=actual['protocol_source'],
            runtime_shells_source_and_ROOT_actual_prelaunch_binding=RUNTIME_SHELLS,
            launch_readiness_collector_reexecuted_postrun=False,known_runtime_outputs=actual['runtime_outputs'],
            owned_remaining=[],devices_idle_before_after=True,original_lock_verifiers_exit0=True,
            supervisor_final_child_elapsed_seconds=receipt['elapsed_seconds'],supervisor_last_polled_private_bytes=receipt['private_bytes'],
            supervisor_start_to_completed_reply_seconds=actual['completed']['completed_unix']-datetime.datetime.fromisoformat(receipt['started_UTC']).timestamp(),
            evidence_paths_sha256=actual['raw_evidence'],source_archive=str(receipt['source_snapshot']),
            source_archive_sha256=receipt['source_snapshot_sha256'],
            numerical_evidence_limit='Persisted source-bound independent numerical gate results, actual Native keys/counters and all-layer counts; full hidden/KV tensors are not dumped/recomputed by this HOST parser.',
            source_identity_limit='Actual source identity/paired-prefix hashes are retained. Original parent full candidate payload is not persisted in cost rows, so its complete SHA preimage cannot be independently rebuilt by this parser.',
            environment_evidence_limit='Source/actual prelaunch readiness bind Python3.11,CANN9.0,MC2 and both sourced runtime shells; no process environment snapshot is invented.',
            resource_evidence_limit='Preflight/startup/final memory guards are retained; no per-cost device peak/free claim.',
            original_failed_or_incomplete_receipt_never_promoted_to_complete=True,
            independent_own_AR_qualified=False,production_overlap_qualified=False,formal_Goodput=False,
            analyzer_sha256=sha(__file__),HOST_argv=[sys.executable,*sys.argv],torch_imported=False,NPU_launched=False,
            cost_summary=str(summary_path))
        data=(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();result['cost_summary_sha256']=hashlib.sha256(data).hexdigest()
        audit_data=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        with summary_path.open('xb') as f:f.write(data)
        with audit_path.open('xb') as f:f.write(audit_data)
        print(json.dumps(dict(passed=True,audit=str(audit_path),summary=str(summary_path),actual_draft_calls=30,paired_comparisons=15,rank_cost_trials_not_multiplied=True)))
        return 0
    except (OSError,KeyError,ValueError,RuntimeError,AssertionError,StopIteration,tarfile.TarError) as error:
        print(json.dumps(dict(passed=False,result_files_written=False,error=str(error)),ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
