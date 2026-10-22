#!/usr/bin/env python3
"""Terminal-only, independent HOST audit of the three actual V103 controls.

No torch/model imports, process launch, device query, runtime installation or
waiter. Each case is one full463 execution observed by four ranks. Cap3/4 are
linear controls in the tree runtime, with no genuine branching qualification.
"""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile

sys.dont_write_bytecode = True
BASE = Path('/root/SpecSLO/budget-calibration-20261015')
WORK = BASE/'tree-active-cap-controls-v103'
OUT = BASE/'budget-diagnostics'
ACTION = 'active-cap-controls-full463'
CASES = (('active32-cap3', 32, 3), ('active32-cap4', 32, 4), ('active64-cap3', 64, 3))
FREEZE_SHA = '84eb51474a5ccad197217d3953177510914673cbb017263fa48ebab8aba77318'
CPU_SHA = '9de89be0426eb1a3a7b9607c4399bde7e8c73022e22528346d3c62d377cba2a0'
PREAUDIT = OUT/'ROOT_V103_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
PREAUDIT_SHA = 'c7d8374b175a2cb79651cb8bac60b4dd178b79885e3007a3da28b98a1ebc3b8a'
SHELLS = ('/root/data/specslo-cann-runtime/cann-9.0.0/set_env.sh',
          '/root/data/specslo-cann-runtime/nnal/atb/9.0.0/atb/set_env.sh')
MC2_ROWS = [40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197]
PROFILE_SHA = 'ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c'
STATIC_CHECKS = {'frozen_profile_disk_hash','profile_operator','profile_standalone_chained_flush',
    'profile_target_TP3','original_MC2_enabled','model_profile_is_native_normalized_profile',
    'model_profile_matches_frozen_profile','manifest_is_original_native_type',
    'manifest_profile_matches_frozen_profile','model_routes_frozen',
    'engine_and_model_share_original_manifest','original64layers','original128routes',
    'ordinary_chained_rows_equal_planned22','complete_attention_down_route_identity',
    'all_original_routes_enabled','all_original_route_operators','every_planned_row_qualified_on_every_route'}
SOURCE_SUFFIXES = {'.py','.sh','.json','.md','.so','.pyc','.yaml','.yml','.toml','.ini'}


def require(value, label):
    if not value:
        raise RuntimeError('V103 external actual audit refused: '+label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, maximum=64*1024**2):
    path=Path(path);require(path.is_file() and not path.is_symlink(), 'regular retained file: '+str(path))
    raw=path.read_bytes();require(0<len(raw)<=maximum, 'bounded retained JSON: '+str(path))
    return json.loads(raw)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def terminal_receipt(receipt, tag):
    """First gate: do not consume raw outputs of running/failed executions."""
    require(receipt.get('status')=='complete', 'terminal complete status required; failed/running retained unchanged')
    require(receipt.get('schema')=='V103.persistent-cap-controls-supervisor.v1'
        and receipt.get('action')==ACTION and receipt.get('errors')==[], 'exact successful supervisor schema/action')
    require(receipt.get('local_execution_complete') is True
        and receipt.get('formal_Goodput') is receipt.get('online_qualified') is False, 'complete diagnostic-only claims')
    child=receipt.get('child',{})
    require(child.get('exit_code')==0 and type(child.get('exit_code')) is int
        and child.get('argv')==['bash',str(WORK/'launch_local_qualification.sh'),'--action',ACTION,'--tag',tag],
        'actual child command and successful reap')
    require(type(child.get('pid')) is int and child['pid']>0 and receipt.get('owned_remaining')==[], 'owned-process cleanup')
    require(receipt.get('physical_cards')==[2,5,6,7]
        and receipt.get('limits')==dict(seconds=7200,private_bytes=256*1024**2,file_bytes=64*1024**2), 'original card/resource limits')
    require(finite(receipt.get('elapsed_seconds')) and 0<receipt['elapsed_seconds']<7200
        and type(receipt.get('private_bytes')) is int and 0<receipt['private_bytes']<256*1024**2, 'actual bounded session elapsed/storage')
    require(receipt.get('experimental_sources_unchanged') is receipt.get('external_sources_unchanged') is True,
        'actual final source checks required')
    for label in ('pre_verifier','post_verifier'):
        value=receipt.get(label,{})
        require(value.get('exit_code')==0 and value.get('argv')==['bash',str(BASE/'serial-v1/runtime.sh'),
            'python','-B','/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py',
            '/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json'], 'original '+label+' lock verification')
    for label in ('devices_before','devices_after'):
        value=receipt.get(label,{})
        require(value.get('all_idle') is value.get('four_910B2') is True and type(value.get('hardware')) is dict
            and type(value.get('processes')) is dict and set(value['processes'])=={'2','5','6','7'},
            'actual four-card idle evidence '+label)
        hardware=value['hardware']
        parsed=[(int(device),name) for device,name in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',hardware['output'],re.MULTILINE)
            if not name.isdecimal()]
        require(hardware['exit_code']==0 and hardware['timed_out'] is False
            and sorted(parsed)==[(i,'910B2') for i in (2,5,6,7)]
            and value['parsed_hardware']==[list(row) for row in parsed], 'raw four910B2 hardware evidence '+label)
        for card,row in value['processes'].items():
            require(row.get('idle') is True and row.get('exit_code')==0 and row.get('timed_out') is False
                and row.get('argv')==['npu-smi','info','-t','proc-mem','-i',card,'-c','0']
                and re.search(r'No process in device\.|No running processes found',row.get('output',''),re.IGNORECASE),
                'actual raw per-card process cleanup '+label)
    return receipt


def archive(path, expected):
    """Hash flat regular members without extracting or executing an archive."""
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular source archive')
    names={Path(name).name:value for name,value in expected.items()}
    require(len(names)==len(expected),'unique flat source basenames')
    with tarfile.open(path,'r:gz') as stream:
        members=stream.getmembers()
        require(len(members)==len(names) and len({m.name for m in members})==len(names)
            and all(m.isfile() and m.name in names and Path(m.name).name==m.name for m in members),
            'complete unique flat regular source archive')
        for member in members:
            require(member.size<=64*1024**2 and hashlib.sha256(stream.extractfile(member).read()).hexdigest()==names[member.name],
                'source archive bytes: '+member.name)
    return dict(path=str(path),sha256=sha(path),members=len(names),all_member_SHAs_matched=True)


def host_module(name):
    """Only already SHA-verified stdlib-only V103 audit helpers may load."""
    require('torch' not in sys.modules,'external HOST process must not import torch')
    if str(WORK) not in sys.path:sys.path.insert(0,str(WORK))
    module=importlib.import_module(name)
    require(Path(module.__file__).resolve()==WORK/(name+'.py'),'exact readonly helper origin '+name)
    require('torch' not in sys.modules,'readonly audit helper imported torch')
    return module


class ObjectPairs(list):
    pass


def fusion(path, started, ended):
    path=Path(path);require(path.name=='fusion_result.json' and path.is_file() and not path.is_symlink(), 'exact regular CANN diagnostic')
    raw=path.read_bytes();st=path.stat()
    require(0<len(raw)<=64*1024 and started<=st.st_mtime<=ended,'bounded CANN diagnostic from actual run interval')
    root=json.loads(raw,object_pairs_hook=ObjectPairs)
    require(isinstance(root,ObjectPairs) and 1<=len(root)<=128,'CANN object-pairs root')
    records=[];allowed={'TransdataCastFusionPass','TransdataFz2FzgFusionPass','TransdataFzg2FzFusionPass'}
    for name,row in root:
        require(type(name) is str and re.fullmatch(r'session_and_graph_id_\d+_\d+',name)
            and isinstance(row,ObjectPairs) and len(row)==1 and row[0][0]=='graph_fusion','CANN session and graph_fusion structure')
        passes=row[0][1]
        require(isinstance(passes,ObjectPairs) and 1<=len(passes)<=3
            and len({k for k,v in passes})==len(passes) and {k for k,v in passes}<=allowed,'known unique nested CANN passes')
        for kind,counts in passes:
            require(isinstance(counts,ObjectPairs) and len(counts)==2
                and {k for k,v in counts}=={'effect_times','match_times'}
                and all((type(v) is int and v>=0) or (type(v) is str and re.fullmatch(r'0|[1-9][0-9]*',v)) for k,v in counts),
                'typed CANN counters without nested duplicates')
        records.append(dict(session_and_graph_id=name,graph_fusion={k:dict(v) for k,v in passes}))
    require(path.read_bytes()==raw and path.stat().st_mtime_ns==st.st_mtime_ns,'stable CANN bytes while read')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),mtime_ns=st.st_mtime_ns,
        top_level_object_pair_records=len(records),unique_session_graph_names=len({r['session_and_graph_id'] for r in records}),
        duplicate_top_level_records_preserved=True,records=records,writer_identity_inferred=False,source_input=False)


def runtime_outputs(receipt_path, receipt, bound):
    started=datetime.datetime.fromisoformat(receipt['started_UTC']).timestamp();ended=receipt_path.stat().st_mtime
    require(started<=ended,'receipt start/terminal time order')
    outputs=[]
    for path in WORK.iterdir():
        if not path.is_file() or str(path) in bound:continue
        if path==receipt_path:continue
        if path.name=='fusion_result.json':outputs.append(fusion(path,started,ended))
        elif re.fullmatch(r'CPU_ATTEMPT_[A-Za-z0-9._-]+\.json',path.name):
            require(read(path).get('schema')=='V103.actual-HOST-source-and-contract-tests.v1','known actual HOST attempt evidence')
        elif path.suffix in SOURCE_SUFFIXES:require(False,'unknown unbound code/static input '+path.name)
    return outputs


def source_closure(receipt):
    freeze_path=WORK/'PREPARED_SOURCE_HASHES_V103.json';cpu_path=WORK/'CPU_RESULT_V103.json'
    require(sha(freeze_path)==FREEZE_SHA and sha(cpu_path)==CPU_SHA and sha(PREAUDIT)==PREAUDIT_SHA,
        'independently reviewed freeze/CPU/prelaunch exact SHA')
    freeze=read(freeze_path);cpu=read(cpu_path);pre=read(PREAUDIT)
    require(pre['all_checks_passed'] is True and pre['NPU_started'] is False
        and pre['freeze_sha256']==FREEZE_SHA and pre['CPU_sha256']==CPU_SHA
        and pre['static_files']==len(freeze['files'])==237
        and pre['external_files']==len(freeze['external_files'])==2321,'ROOT prelaunch actual source counts/identity')
    require(freeze['schema']=='V103.actual-HOST-prepared-active-cap-runtime-source-freeze.v1'
        and freeze['CPU_sha256']==CPU_SHA and freeze['actual_HOST_tests']==32,'exact V103 source freeze')
    require(cpu['schema']=='V103.actual-HOST-source-and-contract-tests.v1' and cpu['status']=='passed'
        and cpu['tests_run']==32 and cpu['failures']==cpu['errors']==0
        and cpu['NPU_initialized'] is cpu['model_initialized'] is cpu['torch_imported'] is False
        and cpu['initialization_attempts']==[] and cpu['Python_grammar_checks_passed'] is True
        and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],
        'actual32 HOST tests and unchanged before/after closure')
    sources=freeze['files']|{str(freeze_path):FREEZE_SHA,str(cpu_path):CPU_SHA}
    require(receipt['source_hashes']==sources and receipt['external_input_hashes']==freeze['external_files'],
        'actual receipt exact static/CPU/freeze and external input maps')
    for path,wanted in (sources|freeze['external_files']).items():
        require(sha(path)==wanted,'current source/input SHA '+path)
    shell_map=pre['actual_sourced_runtime_shells_before']
    require(set(shell_map)==set(SHELLS) and all(sha(path)==wanted for path,wanted in shell_map.items()),
        'actual CANN9.0 and ATB9.0 sourced runtime shells unchanged')
    expected_ready=json.loads(pre['readiness_stdout'])
    require(receipt['readiness']==expected_ready,'actual supervisor readiness equals independently executed prelaunch result')
    readiness=host_module('check_v103').require_local_launch(ACTION)
    require(readiness==expected_ready,'postrun actual source-bound HOST readiness')
    prepared=archive(pre['archive'],sources)
    require(prepared['sha256']==pre['archive_sha256'] and prepared['members']==pre['archive_members']==239,
        'ROOT-reviewed prepared archive')
    actual=archive(receipt['source_snapshot'],sources)
    require(actual['sha256']==receipt['source_snapshot_sha256'],'actual launch archive SHA')
    return dict(freeze_SHA256=FREEZE_SHA,CPU_SHA256=CPU_SHA,ROOT_prelaunch_audit_SHA256=PREAUDIT_SHA,
        static_files=237,actual_CPU_tests=32,external_files=2321,actual_CPU_argv=cpu['actual_HOST_argv'],
        runtime_shells_separately_SHA_bound=shell_map,prepared_archive=prepared,actual_source_archive=actual,
        runtime_environment_evidence='frozen actual launch scripts and separately pinned sourced CANN/ATB shells; no fabricated runtime environment dump',
        source_contract=readiness['source_contract']),sources


def memory(row, rank):
    require(row['rank']==rank and row['local_errors']==[], 'actual finite/memory local vote')
    value=row['memory']
    require(all(type(value[k]) is int and value[k]>=0 for k in ('npu_allocated_bytes','npu_reserved_bytes',
        'npu_peak_allocated_bytes','npu_free_bytes','npu_total_bytes','graph_entries')),'explicit actual HBM values')
    require(value['npu_reserved_bytes']<=54*1024**3 and value['npu_peak_allocated_bytes']<=54*1024**3
        and value['npu_free_bytes']>=6*1024**3 and value['graph_entries']<=512,'original actual HBM headroom')
    return value


def expected_native_startup_keys(rank):
    inventory=host_module('startup_receipts_v75')
    if rank==0:
        keys=inventory._explicit_draft_keys()
        keys['target_entries']|=inventory.original_draft_family_keys()
    else:
        plan=host_module('finite_inventory_plan')
        keys=dict(entries=set(),draft_entries=set(),target_entries=
            inventory.expected_prefill_keys(inventory._prefill_queries()))
        return dict(entries=set(),draft_entries=set(),target_entries=
            set(map(repr,keys['target_entries']))|{row['expected_key'] for row in plan.targets()})
    return {name:set(map(repr,values)) for name,values in keys.items()}


def startup(reports):
    summaries=[]
    for rank,report in enumerate(reports):
        s=report['startup'];require(s['rank']==rank,'startup rank')
        keys=s['actual_native_keys'];counts=s['actual_native_entry_counts']
        expected_keys=expected_native_startup_keys(rank);expected={k:len(v) for k,v in expected_keys.items()}
        require(counts==expected and set(keys)==set(counts)
            and all(len(keys[k])==counts[k] and set(keys[k])==expected_keys[k] for k in counts),
            'actual disjoint 86+28/201+44 Native startup inventory')
        require(len(set(x for values in keys.values() for x in values))==sum(counts.values()),'no inventory key aliases')
        m=s['actual_MC2_preflight'];require(m['rank']==rank and m['role']==('draft' if rank==0 else 'target')
            and m['status']=='passed' and m['local_passed'] is m['all4rank_contract_passed'] is True
            and m['failed_checks']==[] and m['native_qualification_flags_modified'] is False
            and m['planned_full_chain_rows']==MC2_ROWS and m['profile_disk_sha256']==PROFILE_SHA
            and m['profile_operator']=='tp3_native_matmul+allreduce_add_rmsnorm'
            and all(v is True for v in m['checks'].values()),'actual original MC2 static contract')
        votes=m['all4rank_votes'];require(sorted(v['rank'] for v in votes)==list(range(4))
            and all(v['local_passed'] is True and v['failed_checks']==[] for v in votes),'actual all4 MC2 preflight vote')
        require(read(WORK/'startup-progress'/f'MC2-preflight-rank{rank}.json')==m,'raw actual startup MC2 report identity')
        if rank:
            require(set(m['checks'])==STATIC_CHECKS and m['manifest_route_count']==128
                and m['model_chained_row_counts']==MC2_ROWS
                and all(m[k]==[] for k in ('missing_chained_rows','unexpected_chained_rows','disabled_routes',
                    'routes_with_missing_or_unqualified_rows')),'all64layers/128 frozen MC2 routes')
            routes=m['route_details'];require(len(routes)==128 and {(v['projection_kind'],v['layer_index']) for v in routes}
                =={(kind,layer) for kind in ('attention','down') for layer in range(64)}
                and all(v['enabled'] is True and v['operator_kind']=='tp3_native_matmul+allreduce_add_rmsnorm'
                    and v['planned_decisions_present']==22 and v['missing_rows']==v['unqualified_rows']==[] for v in routes),
                'actual all128 complete per-row static route decisions')
        receipts=s['original_startup_receipts']
        if rank:
            p,t=receipts['prefill'],receipts['target']
            require(p['status']=='complete' and p['completed']==p['total']==201 and p['transition_checks_passed']==8
                and t['status']=='complete' and t['fixtures']==44 and len(set(t['native_keys']))==44,'original 201+8 and44 actual startup')
            require(read(WORK/'startup-progress'/f'prefill-rank{rank}.json')==p
                and read(WORK/'startup-progress'/f'finite-target-rank{rank}.json')==t,'raw prefill/target startup report identity')
        else:
            d=receipts['draft'];require(d['status']=='complete' and d['explicit_keys']==86
                and sum(len(values) for values in d['native_keys'].values())==86,'original explicit86 draft keys')
            require(read(WORK/'startup-progress'/f'finite-draft-rank{rank}.json')==d,'raw actual draft report identity')
        family=s['original_draft_family']
        require(family['rank']==rank and family['actual_added_qualified_entries']==(28 if rank==0 else 0)
            and family['original_native_method_executed'] is (rank==0)
            and family['native_qualification_flags_modified'] is False,'original Native draft family qualification')
        for key in report['failure_native_counters']:
            require(s['actual_native_counters'][key]==report['native_counters'][key]==0,'actual cumulative Native failure counter '+key)
        summaries.append(dict(rank=rank,actual_native_entry_counts=counts,
            startup_memory=memory(s['actual_original_memory_finite_vote'],rank),
            final_memory=memory(report['after_all_controls_memory'],rank),MC2_original_static_passed=True))
    return summaries


def quantile(values,q):
    ordered=sorted(values);require(bool(ordered),'nonempty timing distribution')
    x=(len(ordered)-1)*q;lo=math.floor(x);hi=math.ceil(x)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)


def recompute_clocks(trace):
    """Rebuild retained output/stream arithmetic, preserve missing Native clocks."""
    requests=trace['requests'];results=trace['generate_results'];wall=trace['host_generate_call_seconds']
    require(len(requests)==len(results)==463 and finite(wall) and wall>0,'complete full463 output and original host timer')
    delivered=[[] for _ in requests];first=[None]*463;last=[None]*463;finished=[0]*463;previous=-1.
    for event in trace['stream_events']:
        i=event['request_index'];t=event['elapsed_seconds'];tokens=event['token_ids']
        require(type(i) is int and 0<=i<463 and event['request_id']==requests[i]['request_id']
            and type(tokens) is list and tokens and all(type(v) is int and v>=0 for v in tokens)
            and finite(t) and previous<=t<=wall+1e-6 and type(event['finished']) is bool and not finished[i],
            'actual client stream owner/order/time/token/finish')
        previous=t
        if first[i] is None:first[i]=t
        last[i]=t;delivered[i].extend(tokens);finished[i]+=int(event['finished'])
    samples={'legacy_Native_paper':[],'existing_stream_paper':[],'existing_stream_observed':[]}
    for i,(request,result) in enumerate(zip(requests,results,strict=True)):
        require(result['request_id']==request['request_id'] and result['slo_class']==request['slo_class']
            and result['slo_tpot_ms']==request['slo_tpot_ms'] and len(delivered[i])==256 and finished[i]==1
            and delivered[i]==result['completion_token_ids'],'actual complete per-owner256 output')
        paper,observed=result['paper_tpot_ms'],result['observed_tpot_ms']
        require(finite(paper) and finite(observed) and paper>=0 and observed>=0
            and math.isclose(paper,observed*255/256,rel_tol=1e-10,abs_tol=1e-8),'retained Native255/256 arithmetic')
        require(result['paper_slo_attained'] is (paper<=request['slo_tpot_ms'])
            and result['paper_slo_goodput_tokens']==(256 if paper<=request['slo_tpot_ms'] else 0),'actual Native paper SLO arithmetic')
        samples['legacy_Native_paper'].append(paper)
        samples['existing_stream_paper'].append((last[i]-first[i])*1000/256)
        samples['existing_stream_observed'].append((last[i]-first[i])*1000/255)
    def stats(values,indices):
        data=[values[i] for i in indices];attained=sum(values[i]<=requests[i]['slo_tpot_ms'] for i in indices)
        return dict(requests=len(data),attained=attained,good_tokens=256*attained,output_tokens=256*len(data),
            SLO_Goodput_tokens_per_second=256*attained/wall,raw_tokens_per_second=256*len(data)/wall,
            TPOT_ms_mean=statistics.mean(data),TPOT_ms_median=statistics.median(data),TPOT_ms_P90=quantile(data,.9))
    metrics={name:dict(**stats(values,list(range(463))),byclass={kind:stats(values,
        [i for i,row in enumerate(requests) if row['slo_class']==kind]) for kind in ('tight','normal','loose')})
        for name,values in samples.items()}
    require({k:metrics['legacy_Native_paper']['byclass'][k]['requests'] for k in ('tight','normal','loose')}
        ==dict(tight=271,normal=96,loose=96),'original full463 class counts')
    native=trace['native_elapsed_seconds']
    require(finite(native) and native>0 and all(r['elapsed_seconds']==native for r in results),'original common Native inner elapsed value')
    return dict(metrics=metrics,host_generate_call_seconds=wall,Native_inner_elapsed_seconds=native,
        host_minus_Native_inner_seconds=wall-native,first_last_stream_clock_is_Native_clock=False,
        Native_raw_finish_start_clocks_available=False,independent_Native_clock_reconstruction=False,
        GP_denominator='same measured host generate_batch entry-to-return wall seconds',
        Native_inner_elapsed_is_GP_denominator=False,formal_paired_baseline_clock_qualification=False)


def performance_case(traces, shared_clock_summary):
    require(len(traces)==4 and [r['rank'] for r in traces]==list(range(4)), 'four synchronized observations')
    leader=traces[1];case=leader['case'];active,cap={name:(n,b) for name,n,b in CASES}[case]
    require(all(t['role']=='full463' and t['scope_count']==463 and t['case']==case
        and t['active_limit']==active and t['cap']==cap and t['tree_runtime_linear_control'] is True
        and t['genuine_branched_tree_target_qualified'] is False for t in traces), 'actual full linear case identity')
    clocks=recompute_clocks(leader)
    # Compare independently recomputed numerical fields; the local summary also
    # has explicit interpretation labels which are kept as separate evidence.
    for method,values in clocks['metrics'].items():
        for key,value in values.items():
            if key=='TPOT_ms_mean':continue
            if key=='byclass':
                for cls,items in value.items():
                    for field,num in items.items():
                        if field!='TPOT_ms_mean':require(shared_clock_summary['metrics'][method]['byclass'][cls][field]==num,'independent byclass clock arithmetic')
            else:require(shared_clock_summary['metrics'][method][key]==value,'independent retained clock arithmetic')
    histogram=Counter();N=Counter();budgets=Counter();logical=physical=rows=0
    layout=host_module('target_layout')
    for call in leader['normal_target_calls']:
        n,q,family=call['N'],call['real_query_count'],call['family']
        require(family=='causal' and all(1<=b<=cap for b in call['budgets']),'cap3/4 actual causal target family')
        m,s,dummy=layout.envelope(n,q,family)
        require((m,s)==(call['physical_M'],call['physical_S'])
            and call['physical_query_partition']==[b+1 for b in call['budgets']]+dummy,'exact actual original physical mapper')
        histogram[f'{family}/M{m}/S{s}']+=1;N[str(n)]+=1;budgets.update(str(b) for b in call['budgets'])
        logical+=q;physical+=m;rows+=n
    phases=[]
    for rank,t in enumerate(traces):
        p=t['phase_seconds'];require(set(p)=={'draft','target','exchange','verify','broadcast','state_update'}
            and all(finite(v) and v>=0 for v in p.values()), 'original six actual phase timers')
        phases.append(dict(rank=rank,host_generate_call_seconds=t['host_generate_call_seconds'],
            Native_inner_elapsed_seconds=t['native_elapsed_seconds'],phase_seconds=p,
            decode_counters=t['decode_counters'],allocator=t['allocator']))
    counters=leader['decode_counters']
    keys=('spec_rhythm_tree_rounds','spec_rhythm_tree_nodes','spec_rhythm_tree_accepted_nodes',
        'spec_rhythm_verified_tokens','spec_rhythm_tree_eager_promoted','spec_rhythm_tree_eager_invalidated',
        'spec_rhythm_target_causal_tree_rounds','spec_rhythm_target_full_tree_rounds')
    require(all(type(counters[k]) is int and counters[k]>=0 for k in keys),'explicit actual throughput/eager/family counters; missing is not zero')
    verification_rows=sum(r['verification_rounds'] for r in leader['generate_results'])
    require(type(verification_rows) is int and verification_rows>0,'actual original per-owner verification rounds')
    return dict(case=case,active_limit=active,cap=cap,actual_full463_executions=1,
        synchronized_rank_observations=4,independent_repeated_trials=1,clocks=clocks,
        normal_target=dict(calls=len(leader['normal_target_calls']),row_verifications=rows,
            logical_queries=logical,physical_queries=physical,padding_queries=physical-logical,
            N_histogram=dict(sorted(N.items(),key=lambda item:int(item[0]))),
            candidate_budget_histogram=dict(sorted(budgets.items())),physical_family_histogram=dict(sorted(histogram.items()))),
        original_counters={k:counters[k] for k in keys},
        actual_total_decode_tokens_after_first=463*255,actual_per_owner_verification_rounds=verification_rows,
        actual_decode_tokens_per_verification_round_including_terminal_clipping=463*255/verification_rows,
        rank_observations=phases,per_phase_rank_max_seconds={k:max(r['phase_seconds'][k] for r in phases) for k in phases[0]['phase_seconds']},
        phase_rank_maxima_are_not_added=True,phase_times_are_original_wall_intervals=True,
        pure_CPU_or_NPU_decomposition_claimed=False,local_numerics_do_not_prove_whole_workload_AR=True,
        genuine_branched_tree_target_qualified=False,production_overlap_qualified=False,formal_Goodput=False)


def warmup_profile_evidence(reports,directory):
    path=OUT/'V103_ACTUAL_FIRST_WARMUP_HOST_PROFILE.json';value=read(path)
    parser=OUT/'analyze_v103_actual_warmup_profile.py'
    require(value['schema']=='ROOT.V103.actual-warmup-only-cProfile-analysis.v1'
        and value['source_script_sha256']==sha(parser)
        and value['cumulative_times_overlap'] is value['C_wait_wall_included'] is True
        and value['pure_CPU_cycles_claimed'] is value['representative_full463_timing']
            is value['device_cost_gain_proved'] is value['formal_Goodput'] is False,
        'completed root warmup-only profile interpretation/source')
    row=next(t for t in reports[1]['traces'] if t['case']=='active32-cap3' and t['role']=='warmup65')
    require(value['trace']==dict(path=row['artifact']['path'],sha256=row['artifact']['sha256']),
        'root profile binds the actual retained first-warmup trace')
    raw=host_module('audit_caps_v103').artifact(row['artifact'],directory)
    require(value['profile']==raw['actual_HOST_profile'] and value['actual_profiled_original_generate_seconds']==raw['host_generate_call_seconds']
        and value['actual_normal_target_calls']==len(raw['normal_target_calls'])
        and value['phase_seconds']==raw['phase_seconds'] and value['decode_counters']==raw['decode_counters'],
        'root warmup parser actual raw scope/counters/profile')
    profile=value['profile'];require(sha(profile['path'])==profile['sha256']
        and Path(profile['path']).stat().st_size==profile['bytes'],'actual pstats raw SHA/size')
    return dict(path=str(path),sha256=sha(path),source_parser=dict(path=str(parser),sha256=sha(parser)),
        actual_profiled_original_generate_seconds=value['actual_profiled_original_generate_seconds'],
        actual_normal_target_calls=value['actual_normal_target_calls'],
        total_profile_exclusive_wall_seconds=value['total_profile_exclusive_wall_seconds'],
        selected_functions=value['selected_functions'],top_exclusive=value['top_exclusive'][:20],
        cumulative_times_overlap=True,C_wait_wall_included=True,pure_CPU_cycles_claimed=False,
        representative_full463_timing=False,device_cost_gain_proved=False)


def run(tag):
    require(sys.version_info[:2]==(3,11),'retained Python3.11 AST ABI required; use the original runtime interpreter')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',tag),'safe exact tag')
    receipt_path=WORK/('local-receipt-'+tag+'.json');receipt=read(receipt_path)
    terminal_receipt(receipt,tag)  # Must precede every raw report/profile read.
    original_receipt_SHA=sha(receipt_path)
    require(Path(receipt['source_snapshot'])==WORK/'source-snapshots'/('local-'+tag+'.tar.gz'),
        'exact actual per-tag source archive path')
    closure,bound=source_closure(receipt)
    runtime=runtime_outputs(receipt_path,receipt,bound)
    require(receipt['actual_runtime_outputs']==host_module('cap_source_snapshot_v103').actual_runtime_outputs(WORK),
        'actual final runtime diagnostic observations unchanged')
    directory=WORK/'local-output'/tag
    paths=[directory/f'rank{rank}-local-diagnostic.json' for rank in range(4)]
    require(receipt['report_paths']==list(map(str,paths)), 'exact four retained worker report paths')
    reports=[read(path) for path in paths]
    completed=read(directory/'completed.json');expected_replies=[['calibrated',rank,dict(rows=0,timings=0)] for rank in range(4)]
    require(completed['action']==ACTION and completed['replies']==receipt['completed_replies']==expected_replies
        and completed['formal_Goodput'] is False,'actual four-rank terminal completion')
    exited=read(directory/'worker-exit.json')
    require(exited==dict(closed=True,all_exited=True),'actual all original workers exited')
    audit=host_module('audit_caps_v103');original=audit.audit_reports(reports,directory)
    require(original==completed['actual_cap_controls_audit'],'independently rerun complete original raw report/transaction audit')
    startup_evidence=startup(reports)
    full=[];artifacts=[]
    for case,active,cap in CASES:
        traces=[]
        for rank,report in enumerate(reports):
            selection=[t for t in report['traces'] if t['role']=='full463' and t['case']==case]
            require(len(selection)==1,'one actual full control per case/rank')
            artifact=selection[0]['artifact'];trace=audit.artifact(artifact,directory)
            traces.append(trace);artifacts.append(artifact)
        full.append(performance_case(traces,original['separate_Native_and_existing_stream_clock_summaries'][case]))
    require(sha(receipt_path)==original_receipt_SHA,'terminal receipt changed during external audit')
    require('torch' not in sys.modules,'external actual audit did not import model/device runtime')
    return dict(schema='ROOT.V103.independent-actual-active-cap-controls-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_run_status='complete',
        audit_passed=True,actual_HOST_argv=[sys.executable,*sys.argv],source_script_SHA256=sha(__file__),
        actual_receipt=dict(path=str(receipt_path),sha256=original_receipt_SHA,elapsed_seconds=receipt['elapsed_seconds'],
            private_bytes=receipt['private_bytes'],started_UTC=receipt['started_UTC'],
            child=receipt['child'],owned_remaining=receipt['owned_remaining'],
            devices_before=receipt['devices_before'],devices_after=receipt['devices_after'],
            pre_verifier=receipt['pre_verifier'],post_verifier=receipt['post_verifier']),
        source_and_HOST_closure=closure,known_runtime_outputs=runtime,
        report_artifacts=[dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for path in paths],
        completed=dict(path=str(directory/'completed.json'),sha256=sha(directory/'completed.json')),
        worker_exit=dict(path=str(directory/'worker-exit.json'),sha256=sha(directory/'worker-exit.json'),value=exited),
        original_complete_raw_audit=original,actual_startup_and_memory=startup_evidence,
        actual_full_trace_artifacts=artifacts,cases=full,
        experimental_design=dict(cases=3,full463_executions_per_case=1,synchronized_rank_observations_per_case=4,
            rank_observations_are_not_independent_trials=True,model_initializations_per_rank=1,
            target_prefill201_inventory_initializations_per_rank=1,complete_zero65_traces=6,seal_calls=1,
            fixed_order_confounding_retained=True,cap3_and_cap4_are_tree_runtime_linear_controls=True),
        Native_raw_finish_start_clocks_reconstructed=False,C_wait_is_not_pure_CPU=True,
        completed_warmup_cProfile=warmup_profile_evidence(reports,directory),
        NPU_initialized=False,model_initialized=False,genuine_branched_tree_target_qualified=False,
        independent_AR_all_requests_qualified=False,production_overlap_qualified=False,formal_Goodput=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    value=run(args.tag);path=OUT/('ROOT_V103_ACTUAL_ACTIVE_CAP_CONTROLS_'+args.tag+'_AUDIT.json')
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path),audit_passed=True,
        cases=[dict(case=row['case'],host_generate_seconds=row['clocks']['host_generate_call_seconds'],
            native_inner_seconds=row['clocks']['Native_inner_elapsed_seconds'],
            metrics=row['clocks']['metrics']) for row in value['cases']]),ensure_ascii=False))


if __name__=='__main__':
    main()
