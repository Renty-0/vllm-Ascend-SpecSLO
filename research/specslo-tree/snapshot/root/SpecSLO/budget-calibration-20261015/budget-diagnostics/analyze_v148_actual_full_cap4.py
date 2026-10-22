"""Terminal-only V148 HOST audit of the standalone production control. Never launches, queries, or imports a device.

Prepared before terminal; ROOT supplies the final receipt SHA. Failed/partial
artifacts remain failed/partial and cannot enter the original full-run auditor.
"""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
from datetime import datetime,timezone

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-high-normal-cap4-native-families-v148'
MANIFEST=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl')
# Deliberately unbound: no actual identity is invented before candidate freeze.
CPU_SHA='9c701049b3a798aef184f38b7644380e7b85719af7cde8384639995917307872'
FREEZE_SHA='40303ccec611b6cd5c8834933cd03137d6ec2da38a3551b009e03564913e8dce'
ARCHIVE_SHA='64f3bf65451ceee2c14b8e80c446b573681d03195bdfbd59991581e2ae4b848b'
PRE_SHA='7a0f78942989267093b2ad7eff166f7dca47a4719714d60cb059ac60c963e97f'
PRE_PATH=str(D/'ROOT_V148_PREPARED_SOURCE_HOST_AUDIT.json')
CPU_SCHEMA='V148.actual-native-inventory-family-fix-HOST.v1'
ACTUAL_NEW_HOST_TESTS=10
LOCAL_SOURCE_COUNT=377
EXTERNAL_SOURCE_COUNT=6007
MODE_CONTRACT_BOUND=True
PARENT_AUDITOR_PATH=D/'analyze_v140_actual_normal_admission_revision2.py'
PARENT_AUDITOR_SHA='8505318726ab4665aa8137523e9b295120732b7d38cf5710fd23dc3778918944'
HELPERS={'analyze_v132_actual_failed_metadata':'3b35d0fbe799558dc15295c8337f3cdb556075fe958cadf6f20dc2a1826e7173',
 'analyze_v96_actual_capacity64':'b9e420f70d84966f0c1b5edef2721b6213756506c27e3fd9ec27be3e774ade6b',
 'analyze_v119_actual_combined':'903ae516280dacf68c1458358d4019fefa2a0a245819a4cf2e9fd5d1a768da5e'}
COMPONENT_HELPER_SHA='64520cd9a9e243c086dcb9faefcf8f18ffae1f59e0ed64af42572773cced35e1'
HELPERS['analyze_v144_actual_variable_B4']=COMPONENT_HELPER_SHA
NATIVE_COUNTERS=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
    'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',
    *(f'aclgraph_{role}_{field}' for role in ('generic','draft','target') for field in
      ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
ROLES=('warmup65','zero65-1','zero65-2','full463')
PRIORITY={'tight':3,'normal':2,'loose':1}
RULE='V140.normal-class-priority-defer-age-owner-order.v1'


def need(value,message):
    if not value:raise ValueError('V148 external: '+message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def desc(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def finite(value):
    if type(value) is float:need(math.isfinite(value),'nonfinite JSON scalar')
    elif type(value) is dict:
        for item in value.values():finite(item)
    elif type(value) is list:
        for item in value:finite(item)
    else:need(value is None or type(value) in (str,bool,int),'non-JSON value')


def read(path):
    value=json.loads(Path(path).read_bytes());finite(value);return value


def compressed(path,*,artifact=None,directory=None):
    path=Path(path).resolve()
    if directory is not None:need(path.is_relative_to(Path(directory).resolve()),'artifact escaped actual directory')
    need(0<path.stat().st_size<=64*1024**2,'compressed artifact bound')
    if artifact is not None:need(path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256'],'artifact bytes/SHA')
    with gzip.open(path,'rb') as stream:raw=stream.read(256*1024**2+1)
    need(len(raw)<=256*1024**2,'expanded artifact bound')
    if artifact is not None:need(len(raw)==artifact['uncompressed_bytes'],'expanded artifact size')
    value=json.loads(raw);finite(value);return value


@contextmanager
def host_modules():
    need(sys.version_info[:2]==(3,11),'retained3.11 required for original AST contracts')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):need(os.environ.get(key)=='1','HOST thread cap '+key)
    need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','device backend autoload must be disabled')
    previous=list(sys.path);sys.path[:]=[str(D),*(s for s in sys.path if Path(s or '.').resolve()!=D)]
    try:
        helpers={}
        for name,wanted in HELPERS.items():
            path=D/(name+'.py');need(sha(path)==wanted,'external HOST helper source changed: '+name)
            helpers[name]=importlib.import_module(name)
            need(Path(helpers[name].__file__).resolve()==path,'external HOST helper origin differs: '+name)
        provenance=helpers['analyze_v132_actual_failed_metadata']  # Installs backend import guard.
        sys.path.insert(0,str(P))
        names=('audit_candidate_v132','audit_production_v101','bounded_report_audit_v78','shape_scope_v101',
            'startup_receipts_v75','finite_inventory_plan','local_qualification_scope','source_snapshot_v132',
            'private_storage_v132','check_v132','audit_v146','scope_v146','full_process_v146')
        mods={name:importlib.import_module(name) for name in names}
        need(all(Path(module.__file__).resolve()==P/(name+'.py') for name,module in mods.items()),'actual frozen validator origin')
        yield mods,helpers
        need(not provenance.ATTEMPTS and not any(name.split('.')[0] in provenance.FORBIDDEN for name in sys.modules),'model/backend import attempted')
    finally:sys.path[:]=previous


def nonnegative_hist(value,size,label):
    need(type(value) is list and len(value)==size and all(type(n) is int and n>=0 for n in value),label)
    return list(value)


def owner_list(value,count,label):
    need(type(value) is list and len(value)==len(set(value)) and all(type(i) is int and 0<=i<count for i in value),label)
    return list(value)


def audit_admission(pair,*,generation,count,classes,policy):
    """Replay sparse defer-age records without inventing successful owner lists."""
    need(set(pair)=={'before','after'},'admission phase pair')
    before,after=pair['before'],pair['after']
    for side in (before,after):
        need(side['available'] is True and side['rule']==RULE and side['generation']==generation,'actual admission generation/rule')
        need(type(side['call_count']) is int and side['call_count']>=0,'admission call count')
    need(before['call_count']==0 and before['ages']==before['calls']==[] and before['overflow_calls']==0
        and not any(before['raw_N_histogram']+before['admitted_N_histogram']+before['positive_grant_N_histogram']),
        'fresh-generation admission reset')
    raw_hist=nonnegative_hist(after['raw_N_histogram'],65,'raw0..64 histogram')
    admitted_hist=nonnegative_hist(after['admitted_N_histogram'],33,'admitted0..32 histogram')
    positive_hist=nonnegative_hist(after['positive_grant_N_histogram'],33,'positive0..32 histogram')
    calls=after['call_count'];need(sum(raw_hist)==sum(admitted_hist)==sum(positive_hist)==calls,'histogram call conservation')
    need(after['overflow_calls']==sum(raw_hist[33:]),'overflow histogram/count')
    need(after['Native_home_or_KV_mapping_changed'] is False and after['next_cycle_service_guaranteed'] is False,'admission scope boundary')
    # Possible values account exactly for unlogged all-positive calls: they may
    # remove a prior positive age, and cannot create or increment any age.
    possible={};previous=-1;defer_owner_counts=Counter();overflow_rows=0
    event_raw=[0]*65;event_admitted=[0]*33;event_positive=[0]*33
    for event in after['calls']:
        call=event['call'];need(type(call) is int and previous<call<calls,'sparse call monotonicity')
        if call>previous+1:
            possible={i:values|{0} for i,values in possible.items()}
        raw=owner_list(event['raw_normal'],count,'raw normal owners')
        eager=owner_list(event['post_window_eager'],count,'post-window eager owners')
        admitted=owner_list(event['admitted_normal'],count,'admitted normal owners')
        need(len(raw)<=64 and len(eager)<=32 and len(raw)+len(eager)<=64 and not set(raw)&set(eager)
            and len(admitted)<=32 and set(admitted)<=set(raw),'physical row/owner bounds')
        explicit=event['explicit_active_batch_size']
        need(explicit is None or type(explicit) is int and len(raw)+len(eager)<=explicit<=64,'optional actual active bound')
        klass,ages=event['classes'],event['ages_before']
        need(klass==[classes[i] for i in raw] and len(ages)==len(raw)
            and all(type(age) is int and age>=0 for age in ages),'actual class/age owner alignment')
        for i,age in zip(raw,ages,strict=True):need(age in possible.get(i,{0}),'age increased/reset without eligible or positive evidence')
        if len(raw)>32:
            chosen=set(sorted(raw,key=lambda i:(-PRIORITY[classes[i]],-ages[raw.index(i)],i))[:32])
            need(admitted==[i for i in raw if i in chosen],'class/defer-age/owner priority admission')
        else:need(admitted==raw,'at-most32 original owner order')
        grants=event['positive_normal_grants'];need(type(grants) is list,'positive grants list')
        positive=owner_list([entry['owner'] for entry in grants],count,'actual positive grant IDs')
        need(len(positive)==len(set(positive)) and set(positive)<=set(admitted)
            and positive==[i for i in raw if i in set(positive)]
            and all(type(entry['B']) is int and 1<=entry['B']<=(4 if len(admitted)>8 else 5) for entry in grants),
            'actual positive grant owners/caps/order')
        deferred=[i for i in raw if i not in set(positive)]
        need(len(raw)>32 or bool(deferred),'ordinary all-positive call must not have full event')
        need(event['cap_deferred_normal']==[i for i in raw if i not in set(admitted)]
            and event['zero_grant_deferred_normal']==[i for i in admitted if i not in set(positive)]
            and event['all_deferred_normal']==deferred,'all raw owners retained in explicit actual defer partitions')
        expected=[0 if i in set(positive) else age+1 for i,age in zip(raw,ages,strict=True)]
        need(event['ages_after']==expected,'only positive grants clear age; every actual defer increments')
        for i,age in zip(raw,expected,strict=True):possible[i]={age}
        for i in deferred:defer_owner_counts[i]+=1
        need((event['raw_N'],event['admitted_N'],event['positive_grant_N'],event['eager_N'])==(len(raw),len(admitted),len(positive),len(eager)),'event row-count metadata')
        need(event['overflow'] is (len(raw)>32) and event['actual_work_normal_row_limit']==32
            and event['actual_work_combined_row_limit']==64,'event physical contract')
        event_raw[len(raw)]+=1;event_admitted[len(admitted)]+=1;event_positive[len(positive)]+=1
        overflow_rows+=len(raw)>32;previous=call
    if previous<calls-1:possible={i:values|{0} for i,values in possible.items()}
    final=after['ages'];owners=owner_list([entry['owner'] for entry in final],count,'final sparse age owners')
    need(owners==sorted(owners) and all(type(e['age']) is int and e['age']>0 for e in final),'positive age map only; no stored zero-age keys')
    actual=dict((e['owner'],e['age']) for e in final)
    for i in set(possible)|set(actual):need(actual.get(i,0) in possible.get(i,{0}),'final ages outside sparse event/release evidence')
    remaining_raw=[a-b for a,b in zip(raw_hist,event_raw,strict=True)]
    remaining_admitted=[a-b for a,b in zip(admitted_hist,event_admitted,strict=True)]
    remaining_positive=[a-b for a,b in zip(positive_hist,event_positive,strict=True)]
    need(all(v>=0 for v in remaining_raw+remaining_admitted+remaining_positive)
        and not any(remaining_raw[33:]) and remaining_raw[:33]==remaining_admitted==remaining_positive,
        'unlogged calls are exclusively at-most32 all-positive calls')
    need(overflow_rows==after['overflow_calls'],'every overflow has complete owner event')
    need(all(admitted_hist[n]==policy[f'draft_normal_N{n}_shape_calls'] for n in range(33)),
        'actual admission histogram versus original shape counters')
    need(all(positive_hist[n]==policy[f'actual_draft_normal_N{n}_work_calls'] for n in range(1,33))
        and policy['actual_draft_normal_N0_work_calls']<=positive_hist[0]
        and sum(n*positive_hist[n] for n in range(33))==policy['actual_draft_normal_rows'],
        'actual positive grant rows versus actual model work counters')
    return dict(generation=generation,calls=calls,raw_N_histogram=raw_hist,admitted_N_histogram=admitted_hist,
        positive_grant_N_histogram=positive_hist,overflow_calls=after['overflow_calls'],
        full_owner_events=len(after['calls']),unlogged_at_most32_all_positive_calls=calls-len(after['calls']),
        total_cap_deferred=sum(len(e['cap_deferred_normal']) for e in after['calls']),
        total_actual_zero_grant_deferred=sum(len(e['zero_grant_deferred_normal']) for e in after['calls']),
        distinct_deferred_owners=len(defer_owner_counts),remaining_historical_aged_owners=len(final),
        max_final_defer_age=max((e['age'] for e in final),default=0),
        sparse_age_replay_passed=True,unlogged_success_owner_identity_reconstructed=False,
        next_cycle_service_or_cross_class_fairness_qualified=False)


def audit_gain(pair,*,generation,proposal_counts,previous_after=None):
    need(set(pair)=={'before','after'},'gain phase pair')
    before,after=pair['before'],pair['after']
    need(before['actual_engine_generation']==generation-1 and after['actual_engine_generation']==generation,
        'actual gain engine generation; admission before anticipates next, gain before does not')
    need(before['base']==after['base'],'unchanged calibrated prior')
    if previous_after is not None:need(before==previous_after,'previous actual generation snapshot continuity')
    empirical=[]
    for snapshot in (before,after):
        need(snapshot['prior_weight']==12 and snapshot['unobserved_B_filled_with_zero'] is False
            and snapshot['actual_candidate_topology_B']==[1,2,3,4,5] and snapshot['unused_base_prior_B']==[6,7,8]
            and snapshot['counter_delta_or_cross_generation_subtraction'] is False
            and snapshot['before_fresh_decode_may_describe_previous_generation'] is True
            and snapshot['base_is_calibrated_prior_not_current_generation_measurements'] is True,'gain observation/prior scope')
        base={}
        for row in snapshot['base']:
            cls=row['slo_class'];need(cls in PRIORITY and cls not in base,'base class domain/unique')
            need([r['B'] for r in row['values']]==list(range(1,9))
                and all(type(r['g']) in (int,float) and math.isfinite(r['g']) and r['g']>=0 for r in row['values']),
                'actual base B1..8 domain')
            base[cls]={r['B']:r['g'] for r in row['values']}
        need(set(base)==set(PRIORITY),'all actual prior classes')
        seen=set()
        for row in snapshot['observed']:
            key=(row['slo_class'],row['B']);rounds,total=row['rounds'],row['delivered']
            need(key not in seen and key[0] in PRIORITY and type(key[1]) is int and 1<=key[1]<=5
                and type(rounds) is int and rounds>0 and type(total) is int and rounds<=total<=rounds*(min(key[1],4)+1),
                'actual selected class/B rounds/delivered domain')
            seen.add(key)
            if snapshot is after:
                need(rounds<=proposal_counts[key],'gain observations exceed actual class/B target proposals')
                empirical.append(dict(slo_class=key[0],B=key[1],rounds=rounds,delivered=total,
                    empirical_g=total/rounds,prior_g=base[key[0]][key[1]],
                    posterior_intrinsic_g=(12*base[key[0]][key[1]]+total)/(12+rounds)))
    return dict(actual_source_generation=generation,observed_class_B=empirical,
        actual_observed_rounds=sum(r['rounds'] for r in empirical),
        prior_weight=12,unobserved_B_not_imputed=True,prior_B6_to8_not_measured=True,
        cross_generation_subtraction=False,request_local_remaining_clip_not_reconstructed=True,
        adaptive_correlated_observations_not_heldout_or_causal=True)


def audit_sources(receipt,mods,helpers):
    cp,fp=P/'CPU_RESULT_V132.json',P/'PREPARED_SOURCE_HASHES_V132.json'
    need(sha(cp)==CPU_SHA and sha(fp)==FREEZE_SHA,'pinned V148 CPU/freeze')
    cpu,frozen=read(cp),read(fp)
    need(cpu['schema']==CPU_SCHEMA and cpu['preparation_epoch']=='V148' and cpu['status']=='passed'
        and cpu['tests_run']==ACTUAL_NEW_HOST_TESTS and cpu['errors']==cpu['failures']==0,
        'actual V148 ten-case corrected Native namespace HOST proof')
    need(all(cpu['parent'+str(n)+'_actual_HOST_proof_verified'] is True for n in (7,9,13,8,23))
        and cpu['original7_or13_or8_or23_suite_executed'] is False
        and cpu['Torch_imported'] is cpu['NPU_initialized'] is cpu['model_initialized'] is cpu['NPU_or_ACL_queried'] is False
        and cpu['forbidden_import_attempts']==cpu['forbidden_modules_loaded']==[],
        'prior suites are SHA evidence; no backend imports/device qualification')
    need(cpu['input_hashes']==cpu['input_hashes_after']==frozen['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==frozen['external_files']
        and frozen['CPU_sha256']==CPU_SHA
        and len(frozen['files'])==LOCAL_SOURCE_COUNT and len(frozen['external_files'])==EXTERNAL_SOURCE_COUNT,
        'actual HOST before/after source closure')
    expected={**frozen['files'],**frozen['external_files'],str(cp):CPU_SHA,str(fp):FREEZE_SHA}
    provenance=helpers['analyze_v132_actual_failed_metadata']
    count=provenance.verify_map(expected)
    need(mods['source_snapshot_v132'].collect()==frozen['files'],'strict current static collector')
    need(receipt['source_hashes']==expected
        and receipt['readiness']['identity']==dict(CPU_sha256=CPU_SHA,freeze_sha256=FREEZE_SHA),
        'actual runtime snapshot/readiness identity')
    contract=mods['check_v132'].source_contract()
    need(receipt['readiness']['source_contract']==contract,'unchanged actual readiness source contract')
    need(receipt['readiness']==mods['check_v132'].require_local_launch('split192-candidate-full463'),
        'terminal readiness exactly equals current full original source/prior-component binding')
    legacy=read(P/'ORIGINAL_PRIOR_BINDING_V78.json')['observed']
    need(receipt['external_input_hashes']=={**legacy,**frozen['external_files']},'receipt external plus historical closure')
    provenance.verify_map(receipt['external_input_hashes'])
    need(sha(receipt['source_snapshot'])==receipt['source_snapshot_sha256'],'actual runtime archive SHA')
    runtime=provenance.archive(receipt['source_snapshot'],expected)
    attempts=frozen['actual_HOST_attempts']
    need(type(attempts) is dict and attempts and attempts.get(cpu['attempt_log'])==cpu['attempt_log_sha256'],
        'prepared proof binds exact successful HOST log')
    provenance.verify_map(attempts)
    need(attempts=={str(path):sha(path) for path in sorted((P/'host-attempts').glob('*')) if path.is_file()},
        'all original HOST attempts retained without additions or deletion')
    prepared_path=P/'host-archive/V148_FINAL_CPU_SOURCE.tar.gz'
    need(sha(prepared_path)==ARCHIVE_SHA,'exact prepared archive SHA')
    prepared=provenance.archive(prepared_path,{**expected,**attempts})
    pre_path=Path(PRE_PATH);need(sha(pre_path)==PRE_SHA,'ROOT independent before-freeze audit identity')
    pre=read(pre_path)
    need(pre['status']=='passed' and pre['CPU_sha256']==CPU_SHA
        and pre['freeze_sha256']==FREEZE_SHA and pre['archive_sha256']==ARCHIVE_SHA
        and pre['local_sources']==LOCAL_SOURCE_COUNT and pre['external_inputs']==EXTERNAL_SOURCE_COUNT
        and pre['all_archive_member_bytes_verified'] is pre['all_retained_HOST_attempt_bytes_verified'] is pre['require_local_launch_passed'] is True
        and pre['NPU_or_device_queried'] is False and pre['actual_component_or_full463_not_yet_run'] is True,
        'ROOT independent source/inverse/HOST review scope')
    runtime_shell=B/'serial-v1/runtime.sh'
    sourced=re.findall(r'^source (/[^ \n]+)',runtime_shell.read_text(),re.M)
    need(len(sourced)==2 and all(expected.get(path)==sha(path) for path in (str(runtime_shell),*sourced)),
        'real retained runtime wrapper and sourced scripts bound to source closure')
    need(receipt['actual_sourced_runtime_shells_before']==receipt['actual_sourced_runtime_shells_after']
        and [row['path'] for row in receipt['actual_sourced_runtime_shells_after']]==sourced
        and all(sha(row['path'])==row['sha256'] for row in receipt['actual_sourced_runtime_shells_after']),
        'actual sourced runtime before/after')
    need(receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'runtime aliases before/after')
    for row in receipt['runtime_aliases_after']:
        path=Path(row['path']);need(path.is_symlink() and path.exists() is row['exists']
            and str(path.resolve())==row['resolved'] and os.readlink(path)==row['link'],'retained runtime alias')
    outputs=mods['source_snapshot_v132'].actual_runtime_outputs()
    need(outputs==receipt['actual_runtime_outputs'],'strict known-fusion runtime output closure')
    storage=mods['private_storage_v132'].snapshot(receipt['_audit_tag']);mods['private_storage_v132'].require_limits(storage)
    mods['private_storage_v132'].require_limits(receipt)
    return dict(passed=True,CPU=desc(cp),freeze=desc(fp),ROOT_before_freeze_audit=desc(pre_path),
        local_sources=LOCAL_SOURCE_COUNT,external_sources=EXTERNAL_SOURCE_COUNT,all_current_inputs_SHA_checked=count,
        runtime_source_archive=runtime,prepared_source_archive=prepared,prepared_HOST_attempts=attempts,
        actual_sourced_runtime_shells=receipt['actual_sourced_runtime_shells_after'],runtime_aliases=receipt['runtime_aliases_after'],
        known_runtime_outputs=outputs,unknown_source_outputs_rejected=True,
        storage=dict(terminal_receipt=dict(attempt_private_bytes=receipt['attempt_private_bytes'],
            aggregate_private_bytes=receipt['aggregate_private_bytes'],storage_limits=receipt['storage_limits']),
            current_read_only_HOST_snapshot=storage,matching_measurement_time_not_assumed=True),
        parent_suites_reused_by_SHA_not_rerun=True,device_numerics_not_reexecuted_by_HOST=True,
        launch_guard_did_not_itself_recheck_prepared_attempt_bytes=True)


def proposal_counts(trace,rows):
    """Count real owner/B observations; graph scratch rows are not requests."""
    result=Counter();records=[]
    for call in trace['normal_target_calls']:
        n=call['N'];budgets=call['budgets'];owners=call['sequence_ids']
        partition=call['physical_query_partition'];contexts=call['contexts']
        need(type(n) is int and 1<=n<=32 and type(budgets) is list and len(budgets)==n
            and all(type(b) is int and 1<=b<=5 for b in budgets),
            'actual normal target real N/B alignment')
        need(type(contexts) is list and len(contexts)==n
            and all(type(p) is int and p>=0 for p in contexts),'actual normal target real contexts')
        need(type(owners) is list and type(partition) is list and n<=len(owners)<=n+32
            and len(owners)==len(partition)==call['physical_S']
            and all(type(q) is int and q>0 for q in partition)
            and partition[:n]==[b+1 for b in budgets]
            and all(2<=q<=9 for q in partition[n:])
            and call['physical_M']==sum(partition)
            and call['real_query_count']==sum(partition[:n])
            and sum(budgets)<=96 and call['real_query_count']<=128,
            'actual normal target physical/real query geometry')
        need(all(type(i) is int for i in owners) and len(set(owners))==len(owners)
            and owners[n:]==list(range(463,463+len(owners)-n)),
            'actual normal target dedicated scratch suffix')
        owner_list(owners[:n],len(rows),'actual normal target real owner prefix')
        need(call['family']==('causal' if max(budgets)<=4 else 'full9'),
            'actual normal target real plan family')
        records.append((owners[:n],budgets))
    # Mixed logical events already contain only real normal owners. Their exact
    # length contract is retained; no physical-prefix slicing applies here.
    records.extend((event['target_ids'],event['budgets']) for event in trace['logical_events'] if event['kind']=='plan')
    for owners,budgets in records:
        need(len(owners)==len(budgets),'actual target proposal owner/B alignment')
        for i,b in zip(owners,budgets,strict=True):
            need(type(i) is int and 0<=i<len(rows) and type(b) is int and 1<=b<=5,'actual class/B proposal domain')
            result[rows[i]['slo_class'],b]+=1
    return result


def augment_metrics(metric):
    records=metric['request_metrics']
    names=dict(native_paper='native_paper_ms',native_observed='native_observed_ms',
        event_wall_paper='event_wall_paper_ms',event_wall_observed='event_wall_observed_ms')
    def percentile(values,p):
        values=sorted(values);point=(len(values)-1)*p;lo=math.floor(point);hi=math.ceil(point)
        return values[lo]+(values[hi]-values[lo])*(point-lo)
    groups=[(metric,records)]+[(metric['by_class'][cls],[r for r in records if r['slo_class']==cls]) for cls in metric['by_class']]
    for group,items in groups:
        for label,column in names.items():
            values=[r[column] for r in items];attained=sum(r[column]<=r['slo_ms'] for r in items)
            group[label]=dict(attained=attained,good_tokens=256*attained,
                goodput_tokens_per_second=256*attained/metric['elapsed_seconds'],
                median_tpot_ms=statistics.median(values),P90_tpot_ms=percentile(values,.9),max_tpot_ms=max(values))
    metric['quantile_method']='median conventional midpoint; P90 linear interpolation at zero-based (n-1)*0.9'
    metric['timing_scope']='actual host Native generate_batch entry to return including original arrivals; excludes startup/three65; no supervisor elapsed substitution'
    metric['clock_methods']=dict(native_paper='actual reported charged cycle-clock TPOT /256; omitted raw start/finish not reconstructed',
        native_observed='actual reported charged cycle-clock TPOT /255; denominator relationship checked',
        event_wall_paper='existing first-to-last token delivery wall seconds *1000 /256',
        event_wall_observed='same existing stream wall interval *1000 /255')
    return metric


def component_native_pair(value,before,*,zero=False):
    """Exact 21 Native counters; no cold flag is fabricated by this audit."""
    need(set(before)==set(NATIVE_COUNTERS),'complete21 Native counters required')
    need(set(value)=={'before','after','delta'} and value['before']==before,'component Native before continuity')
    need(set(value['before'])==set(value['after'])==set(value['delta'])==set(before),'component complete Native namespace')
    need(all(type(value[side][k]) is int and value[side][k]>=0 for side in value for k in before),
        'component finite nonnegative integer Native counters')
    need(value['delta']=={k:value['after'][k]-value['before'][k] for k in before}
        and all(v>=0 for v in value['delta'].values()),'component Native raw arithmetic')
    bad=[k for k in before if any(word in k for word in ('failed_captures','fallback','disabled_entry','unclassified','validation_failures'))]
    need(not any(value['delta'][k] for k in bad),'component Native failure/fallback counter')
    if zero:need(not any(value['delta'].values()),'component full protocol zero discovers or validates nothing')
    return value['after']


def component_lease(lease,reference=None):
    need(lease['pages']==140 and lease['storage_rows']==527
        and lease['candidate_owners']==list(range(32)) and lease['reference_owners']==list(range(32,64))
        and lease['dummy_owners']==list(range(463,475)) and lease['source_KV_copied'] is False
        and type(lease['allocation_id']) is int and re.fullmatch('[0-9a-f]{64}',lease['tables_sha256']),
        'actual own-full527 simultaneous140-page lease/owners')
    if reference is not None:need(lease==reference,'actual immutable lease across commit/reference history')


def component_boundary_labels(helper):
    original=helper.expected_stages()
    need(original[:4]==['configuration','source-origins','minimal-construction','MC2-before'],
        'pinned local-stage producer prefix')
    need(original[-7:]==['exact-observed-inventory','MC2-after','MC2-positive-original-route',
        'original-unified-seal','final-Native','final-exact-inventory','all4-component-required-coverage'],
        'pinned local-stage producer suffix')
    return ['V146/'+name for name in ('full-configuration','fresh-fixture-gain',
        'capture-fixture-only-state','preserve-inherited-Native-entries')]+[
        'configuration','source-origins','full-process-construction','MC2-before']+original[4:-7]+[
        'observed-full-process-inventory','MC2-after','MC2-positive-original-route','final-Native',
        'all4-component-required-coverage']+['V146/'+name for name in ('only-two-new-hidden-keys',
        'fresh-production-gain','discard-fixture-ledger-and-restore-defer-flags',
        'qualification-cache-released-before-production')]


def component_boundaries(events,helper):
    labels=component_boundary_labels(helper)
    need(len(events)==len(labels),'exact full component pre-save boundary prefix count')
    for seq,(event,label) in enumerate(zip(events,labels,strict=True)):
        need(event['sequence']==seq and event['label']==label
            and [row['rank'] for row in event['ranks']]==list(range(4))
            and all(row['sequence']==seq and row['label']==label and row['error'] is None for row in event['ranks']),
            'exact nonnested own-full component common-stage order/errors at '+str(seq))
    return len(labels)


def component_inventory_keys(record,rank,*,after):
    counts=dict(entries=0,draft_entries=60,target_entries=54) if rank==0 else dict(
        entries=0,draft_entries=0,target_entries=249 if after else 247)
    need(set(record)==set(counts) and {name:len(keys) for name,keys in record.items()}==counts,
        'actual Native dictionaries distinct from logical report purpose labels')
    for keys in record.values():need(keys==sorted(set(keys)),'actual Native keys unique sorted repr')
    return counts


def component_target_flags(value,key,flags):
    """Replay the actual Native flag transition; inherited greedy is already qualified."""
    need(value['phase'] in ('first','changed','restored'),'declared actual target phase')
    was=flags.get(key,False);now=was or value['phase']!='first'
    need(value['key']==key and value['runtime_validated_before'] is was
        and value['runtime_validated_after'] is now,'natural flags distinguish inherited greedy from new hidden')
    flags[key]=now


def audit_own_full_component(reports,out,mods,helpers=None):
    """Reconstruct retained typed local transactions inside this full process.

    This is additional qualification. The original21/26, two production65
    zeros, unified seal and463 workload remain separately mandatory.
    """
    if helpers is None:
        name='analyze_v144_actual_variable_B4';need(sha(D/(name+'.py'))==COMPONENT_HELPER_SHA,'component pure HOST helper SHA')
        helper=importlib.import_module(name)
    else:helper=helpers['analyze_v144_actual_variable_B4']
    directory=out/'variable-B4-own-full-process'
    strict=mods['audit_v146'].audit_reports(directory)
    need(strict['status']=='passed' and strict['component_numerical_qualified'] is True,
        'actual frozen full527 component original raw audit')
    raw_reports=[compressed(directory/f'rank{rank}-component.json.gz',directory=directory) for rank in range(4)]
    initial=helper.fixtures();repair_buckets=set();protocol_missing={0:False,1:False};final_missing={0:False,1:False}
    summaries=[];expected_private=set();native_after=[];stage_count=None
    sources={**read(P/'PREPARED_SOURCE_HASHES_V132.json')['files'],**read(P/'PREPARED_SOURCE_HASHES_V132.json')['external_files']}
    for rank,(outer,report) in enumerate(zip(reports,raw_reports,strict=True)):
        expected_config=dict(outer['original_inventory_and_qualification_config'],
            spec_rhythm_draft_token_budget=192,spec_rhythm_max_eager_tokens=3,spec_rhythm_eager_reserve_tokens=32)
        need(report['actual_config']==expected_config,'same full engine actual config with original three candidate-control fields')
        construction=report['construction']
        need(construction['maxseq']==64 and construction['maxqueued']==463 and construction['storage']==527
            and construction['block_size']==128 and construction['pages']==544
            and construction['layers']==(28 if rank==0 else 64) and construction['same_full_process'] is True
            and construction['tree_scratch']==list(range(463,527))
            and construction['candidate_owners']==list(range(32)) and construction['reference_owners']==list(range(32,64))
            and construction['candidate_dummy_owners']==list(range(463,469))
            and construction['reference_dummy_owners']==list(range(469,475))
            and construction['ordinary_mixed_graph_enabled'] is False,'actual own527 constructor geometry, no new local96 engine')
        expected_env=dict(SPECSLO_VISIBLE_DEVICES='2,5,6,7',ASCEND_RT_VISIBLE_DEVICES='0,1,2,3',
            SPECSLO_TARGET_DEVICES='1,2,3',RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='4')
        need(all(report['actual_environment'].get(k)==v for k,v in expected_env.items())
            and report['physical_cards']==[2,5,6,7] and report['physical_rank_card']==[2,5,6,7][rank]
            and report['actual_target_leader_rank']==1 and report['actual_logical_device']=='npu',
            'actual own-process physical/logical role metadata')
        need(report['initial_source_root_teacher_forced_not_own_AR'] is True
            and report['subsequent_draft_verdict_commit_actual'] is True
            and report['future_full_requires_21scope26phase_and_two65zero'] is True,
            'honest initial fixture and genuine later recurrence')
        stage_count=component_boundaries(report['boundary_events'],helper)
        need(report['boundary_events']==raw_reports[0]['boundary_events'],'all4 actual shared stage packets identical')
        for name,row in report['actual_source_origins'].items():
            need(sources.get(row['path'])==row['sha256'],'actual own component imported source '+name)
        chain=compressed(outer['mixed_same_M_qualification']['artifact']['path'],
            artifact=outer['mixed_same_M_qualification']['artifact'],directory=out)
        inherited=chain['actual_inventory']['after'];component_inventory_keys(inherited,rank,after=False)
        delta=report['inventory_delta'];final=report['inventory_after_local_protocol']
        need(construction['existing_inventory']==delta['before']==inherited,'old21/26 keys continue into own full component')
        component_inventory_keys(final,rank,after=True)
        need(delta['after']==final,'actual component final key record equals wrapper finish record')
        mods['full_process_v146'].validate_inventory_delta(inherited,final,is_draft=rank==0)
        need(delta['inherited_entry_identity_preserved'] is delta['disabled_keys_unchanged'] is True
            and delta['Native_flags_assigned'] is False and delta['new_hidden_keys']==(0 if rank==0 else 2)
            and delta['new_draft_keys']==delta['new_greedy_keys']==delta['new_generic_keys']==0,
            'original entry objects survive; only own two hidden counterparts new')
        states=delta['inherited_states_before']
        need(set(states)==set(inherited) and all(set(states[family])==set(inherited[family]) for family in states),
            'all inherited key flag/bound records present')
        flags={}
        for family in states:
            for key,value in states[family].items():
                need(type(value) is list and len(value)==2 and value[0] is True
                    and (value[1] is None or type(value[1]) is int and value[1]>=0),'inherited natural Native qualification flags')
                flags[key]=value[0]
        native=construction['initial_native']
        need(native==outer['mixed_same_M_qualification']['native']['after'],
            'Native continuity original21/26 after to actual component construction before')
        first_native=dict(native)
        for trace in report['traces']:
            p,c=trace['pass_index'],trace['case'];budgets=helper.VECTORS[c];m,s,dummy=helper.GEOMETRIES[c]
            need((trace['M'],trace['S'])==(m,s),'actual variable B maps original physical M/S')
            bank=trace['lease'];component_lease(bank['lease']);contexts=[list(owner['token_ids']) for owner in initial]
            need(len(bank['prefill'])==64 and bank['initial_root_teacher_forced_not_own_AR'] is True,'64 independent prefix evaluations')
            for owner,prefill in enumerate(bank['prefill']):
                tokens=initial[owner%32]['token_ids'][:-1]
                need(prefill==dict(owner=owner,tokens=len(tokens),sha256=helper.digest(tokens),actual_eager_PA=True,
                    candidate_KV_copied=False),'actual separately evaluated prefix owner '+str(owner))
            hashes=bank['initial_prefix']
            need(hashes['tokens']==helper.digest({i:initial[i%32]['token_ids'][:-1] for i in range(64)})
                and len(hashes['KV'])==(56 if rank==0 else 128)
                and all(re.fullmatch('[0-9a-f]{64}',h) for h in hashes['KV']),
                'protected prefix token digest and complete28/64 KV hash count')
            local_delivered=0;repair_commits=0;fingerprints=[]
            for cycle in range(4):
                path=directory/f'rank{rank}-pass{p}-case{c}-cycle{cycle}.json.gz';expected_private.add(path)
                row=compressed(path,directory=directory);helper.check_source(row['source'],initial,contexts)
                need(row['work_plan_id']==(p*2+c)*5+cycle and row['selected_normal_candidates']==sum(budgets)
                    and row['controlled_typed_fixture_grants_not_policy_allocation'] is True,
                    'original controlled typed local grants; not live allocator policy inference')
                component_lease(row['independent_commit']['lease'],bank['lease'])
                commit=row['commit']
                need(commit['cycle_id']==cycle and commit['target_home']==cycle%2
                    and commit['actual_prefix_epochs']=={str(i):cycle+1 for i in range(32)}
                    and commit['original_decode_sha256']==helper.ORIGINAL_DECODE
                    and commit['canonical_decode_sha256']==helper.CANONICAL_DECODE
                    and commit['extracted_structural_and_numerical_lanes_unchanged'] is True,
                    'actual original protocol/epoch/rotation/source identity')
                fingerprints.append(dict(source=row['source'],selected=row['selected'],rows=row['rows'],confidences=row['confidences'],
                    commit={k:commit[k] for k in ('committed_tokens','accepted_paths','actual_prefix_epochs',
                        'actual_logical_tokens','actual_structural_failed','actual_numerical_failed')}))
                leader=compressed(directory/f'rank1-pass{p}-case{c}-cycle{cycle}.json.gz',directory=directory)
                query=leader['target']['greedy'][2]['full_physical_tokens'][:32+sum(budgets)]
                emitted,accepted=helper.causal_verdict(row['rows'],query,budgets)
                need(commit['committed_tokens']==emitted and commit['accepted_paths']==accepted,
                    'independently reconstructed causal B1..4 verdict from actual full returned target tokens')
                contexts=[old+tokens for old,tokens in zip(contexts,emitted,strict=True)]
                need(commit['actual_logical_tokens']==contexts,'actual own recurrence from committed outputs')
                local_delivered+=sum(map(len,emitted))
                if rank==0:
                    proof=row['draft'];helper.draft_ABI(proof,budgets)
                    need(proof['roots']==[v['token_ids'][-1] for v in row['source']['contexts']]
                        and row['rows']==[raw[:b] for raw,b in zip(proof['raw_pool'],budgets,strict=True)],
                        'actual draft roots/selected primary rows from complete8 raw pool')
                    need(row['confidences']==[sum(scores[:b])/b for scores,b in zip(proof['node_confidences'],budgets,strict=True)],
                        'exact original selected mean confidence')
                    if proof['natural_missing_suffix_observed']:
                        repair_buckets.add(proof['first_PA_physical']);protocol_missing[c]=True;repair_commits+=1
                else:
                    for arm in ('greedy','hidden'):
                        phases=row['target'][arm];need([v['phase'] for v in phases]==['first','changed','restored'],
                            'actual original F/T/F target qualification sequence')
                        key=repr(helper.target_key(m,s,arm=='hidden'))
                        for value in phases:
                            component_target_flags(value,key,flags)
                            execution=value['execution']
                            need(set(execution)=={'mode','fallback_reason','capture_attempted','replay_executed'}
                                and execution['mode'] in ('capture_replay','replay') and execution['fallback_reason'] is None
                                and execution['replay_executed'] is True,'original Native serialized actual graph execution')
                        if arm=='greedy':need(phases==leader['target']['greedy'],'all3 target ranks retain exact full physical target return')
                    need(row['reference']['same_M']==m and row['reference']['same_S']==s
                        and row['reference']['real_query_tokens']==query
                        and row['independent_commit']['own_reference_query_KV_compacted'] is True,
                        'independently evaluated own same-M query identity and KV compaction')
            nextpath=directory/f'rank{rank}-pass{p}-case{c}-next-draft.json.gz';expected_private.add(nextpath)
            nxt=compressed(nextpath,directory=directory);helper.check_source(nxt['source'],initial,contexts)
            need(nxt['work_plan_id']==(p*2+c)*5+4 and trace['fingerprints']==fingerprints
                and trace['next_draft']=={k:nxt[k] for k in ('source','selected','rows','confidences')},
                'actual postcommit next draft and every zero replay fingerprint reconstructed')
            need(bank['current_context_sha256']==[helper.digest(v) for v in contexts],'actual final independent history digests')
            if rank==0:
                helper.draft_ABI(nxt['draft'],budgets)
                if nxt['draft']['natural_missing_suffix_observed']:
                    repair_buckets.add(nxt['draft']['first_PA_physical']);final_missing[c]=True
                summaries.append(dict(pass_index=p,case=c,C=sum(budgets),Q=32+sum(budgets),M=m,S=s,
                    dummy_lengths=list(dummy),actual_commits=4,actual_delivered_tokens=local_delivered,
                    repaired_draft_then_committed_cycles=repair_commits,
                    final_next_draft_repair=nxt['draft']['natural_missing_suffix_observed']))
            native=component_native_pair(trace['Native'],native,zero=p>0)
        need(report['final_Native']==native,'component final Native equals all six real trace deltas')
        expected_new=0 if rank==0 else 2
        need(native['aclgraph_captures']-first_native['aclgraph_captures']==expected_new
            and native['aclgraph_capture_attempts']-first_native['aclgraph_capture_attempts']==expected_new,
            'actual captures/attempts exactly match the two new hidden keys, or zero draft keys')
        native_after.append(native)
        need(report['inventory_delta']['after']=={family:sorted(keys) for family,keys in final.items()},'canonical final inventory')
        need(report['retained_boundary_prefix']==dict(stage_count=stage_count,
            last_label=report['boundary_events'][-1]['label'],saving_stage_and_later_audit_not_retained=True),
            'retained boundary ends before save/audit future stages')
        finalpath=directory/f'rank{rank}-component.json.gz';expected_private.add(finalpath)
        component=outer['variable_B4_own_full_process']
        need(component['audit']==json.loads(json.dumps(strict,allow_nan=False))
            and component['directory']==str(directory) and component['full527_own_process_actual_component_qualified'] is True
            and component['component_sealed'] is component['production_full463_qualified'] is False
            and component['production_two65zero_still_required'] is True,
            'original same-process component result binds its own retained audit and cannot replace full65 zeros')
    need(len(expected_private)==124 and {path for path in directory.rglob('*') if path.is_file()}==expected_private,
        'exact own component4reports/96cycle/24next artifacts, no missing or unrelated evidence')
    need(all(protocol_missing.values()) and strict['independently_recomputed_protocol_missing_coverage']=={str(k):v for k,v in protocol_missing.items()},
        'both vectors naturally repaired then published/verified/committed; final-only draft does not qualify')
    return dict(status='passed',original_frozen_component_audit=strict,
        actual_full527_own_component_qualified=True,actual_local96_component_not_substituted=True,
        Native_after_by_rank=native_after,original21scope26phase_to_component_Native_continuity=True,
        exact_actual_artifacts=124,exploration_commits=8,fixed_protocol_replay_commits=16,next_draft_probes=6,
        all4_fixed_stage_count=stage_count,actual_trace_summary=summaries,
        natural_protocol_repair_coverage=protocol_missing,final_next_draft_repair_coverage=final_missing,
        observed_repair_PA_buckets=sorted(repair_buckets),target_actual_keys_before=247,target_actual_keys_after=249,
        draft_actual_key_families=dict(entries=0,draft_entries=60,target_entries=54),
        initial_source_root_teacher_forced_not_own_AR=True,subsequent_actual_recurrence_recomputed=True,
        full65zeros_and_original463_remain_required=True,
        evidence_limits=['Full hidden/KV tensors were not retained; numerical matcher gates are actual source-bound records, not a new CPU tensor recomputation.',
            'Full block tables were not retained; page count/lease immutability/hashes do not constitute a new CPU raw-page enumeration.',
            'Own same-M tests do not prove arbitrary cross-M or own AR output equivalence.'])


def complete_run(receipt,out,mods,helpers):
    need(receipt['errors']==[] and receipt['child']['exit_code']==0 and receipt['local_execution_complete'] is True,'original terminal completion')
    paths=[out/f'rank{rank}-local-diagnostic.json' for rank in range(4)]
    need(receipt['report_paths']==list(map(str,paths)),'original four final worker report paths')
    reports=[read(path) for path in paths]
    expected_combination=read(P/'CANDIDATE_V132.json')
    for report in reports:
        need(report['ordinary_mixed_graph_enabled'] is False and report['actual_mixed_scratch_sequence_ids']==[]
            and report['actual_tree_scratch_sequence_ids']==list(range(463,527))
            and report['combination']==expected_combination,'actual flag0/scratch463..526 and declared candidate identity')
    # This is the original immutable whole-chain + four-phase + transaction /
    # Native / output auditor, not a relaxed reimplementation.
    strict=mods['audit_candidate_v132'].audit_worker_reports(reports,MANIFEST,out)
    completed=read(out/'completed.json')
    replies=[['calibrated',rank,dict(rows=0,timings=0)] for rank in range(4)]
    need(completed['replies']==receipt['completed_replies']==replies and completed['actual_capacity64_audit']==json.loads(json.dumps(strict)),
        'actual completed record agrees with strict frozen HOST re-audit')
    common=helpers['analyze_v119_actual_combined'];capacity=helpers['analyze_v96_actual_capacity64']
    startup=capacity.audit_startup(dict(enumerate(reports)),mods,P)
    formula=capacity.production_tpot_contract(P)
    chains=[compressed(r['mixed_same_M_qualification']['artifact']['path'],artifact=r['mixed_same_M_qualification']['artifact'],directory=out) for r in reports]
    same_m=common.same_m(chains,reports,mods,out)
    need(same_m['passed'] is True,'actual all4 original21scope26phase chain')
    component=audit_own_full_component(reports,out,mods,helpers)
    phases=[];previous_gain=None;previous_native=component['Native_after_by_rank']
    for index,role in enumerate(ROLES):
        envelopes=[r['traces'][index] for r in reports]
        raw=[compressed(e['artifact']['path'],artifact=e['artifact'],directory=out) for e in envelopes]
        count=463 if role=='full463' else 65;rows,prefix=mods['audit_production_v101'].manifest_rows(MANIFEST,count)
        generation=raw[0]['source_generation'];need(all(t['source_generation']==generation for t in raw),'four-rank raw source generation')
        admission=[e['actual_normal_admission'] for e in envelopes];gains=[e['actual_gain_state'] for e in envelopes]
        need(all(value==admission[0] for value in admission) and all(value==gains[0] for value in gains),
            'cross-rank actual admission/gain snapshots')
        classes={i:r['slo_class'] for i,r in enumerate(rows)}
        policies=[e['actual_split_work_counts']['delta']['policy'] for e in envelopes]
        need(all(value==policies[0] for value in policies),'four-rank actual typed allocation counters')
        admitted=audit_admission(admission[0],generation=generation,count=count,classes=classes,policy=policies[0])
        gain=audit_gain(gains[0],generation=generation,proposal_counts=proposal_counts(raw[1],rows),previous_after=previous_gain)
        previous_gain=gains[0]['after']
        for rank,(trace,envelope) in enumerate(zip(raw,envelopes,strict=True)):
            need(trace['rank']==rank and trace['role']==role and trace['native']==envelope['native'],'raw actual role/rank/envelope')
            need(trace['native']['before']==previous_native[rank],'Native changes hidden between original generations')
            previous_native[rank]=trace['native']['after']
            allocator=trace['allocator']
            need(trace['fresh_allocation'] is trace['cache_released'] is trace['wrappers_restored'] is trace['class_delegate_restored'] is True
                and allocator['allocation_storage_rows']==[527] and allocator['pool_blocks']==544
                and 0<allocator['referenced_page_highwater']<=544 and allocator['final_referenced_pages']==0
                and allocator['acquired_pages']==allocator['released_pages'],'actual fresh allocation/lease/page cleanup')
            mods['audit_production_v101'].native_delta(trace,zero=role!='warmup65')
        metric=augment_metrics(common.output_metrics(raw[1],rows))
        need(math.isclose(metric['native_paper']['goodput_tokens_per_second'],strict['traces'][index]['output']['goodput_tokens_per_second'],rel_tol=1e-12),
            'independent Native paper GP arithmetic versus original strict auditor')
        phases.append(dict(role=role,count=count,source_generation=generation,artifacts=[e['artifact'] for e in envelopes],
            actual_normal_admission=admitted,actual_gain_state=gain,cross_rank_admission_gain_and_policy_equal=True,
            output=metric,proposals=common.proposal_metrics(raw,envelopes),
            Native_counters_by_rank=[t['native'] for t in raw],
            allocator_by_rank=[t['allocator'] for t in raw],
            phase_seconds_by_rank=[t['phase_seconds'] for t in raw],
            actual_window_cost_state=[e['actual_window_cost_state'] for e in envelopes],
            actual_task_update_counts=[e['actual_task_update_counts'] for e in envelopes],
            first_token_paths=strict['traces'][index]['transactions'],
            actual_peak_active=strict['traces'][index]['actual_peak_active']))
    for rank,report in enumerate(reports):
        cp=compressed(out/f'rank{rank}-candidate-checkpoint.json.gz',directory=out)
        need(cp['rank']==rank and cp['traces']==report['traces'] and cp['phase_control_evidence']==report['phase_control_evidence']
            and cp['mixed_same_M_qualification']==report['mixed_same_M_qualification'],'last actual checkpoint/final report equality')
    return dict(full463_completed=True,original_frozen_worker_audit_reexecuted=True,
        original_frozen_audit=strict,startup=startup,same_M=same_m,Native_TPOT_source_formula=formula,
        own_full527_component=component,phases=phases,full463=phases[-1],final_worker_reports=[desc(p) for p in paths],completed_record=desc(out/'completed.json'),
        offline_hidden_KV_tensor_recomparison_available=False,
        numerical_proof='Source-bound actual all28/all64/hidden/head/changed-restored/MC2 actor records pass original strict chain validator; complete tensors are not retained for new HOST comparison',
        own_AR_all_requests_qualified=False,global_factory_equivalence_proved=False,
        formal_same_card_baseline_ratio_qualified=False,goal_achieved=False)


def failed_run(receipt,out,mods):
    need(bool(receipt['errors']) and receipt['child']['exit_code']!=0,'actual failed status remains failed')
    errors=[];partial=[]
    for path in sorted(out.glob('rank*-last-error.json')):
        row=read(path);errors.append(dict(**desc(path),rank=row['rank'],error=row['error'],failure_evidence=row.get('failure_evidence')))
        index=row.get('failure_evidence',{})
        for item in index.get('artifacts',[]):
            need(item['bytes']<=8*1024**2 and item['uncompressed_bytes']<=64*1024**2,'failure-only private bounds')
            value=compressed(item['path'],artifact=item,directory=out)
            need(value['failure_only'] is True and value['formal_Goodput'] is False,'failure artifact scope cannot become completed full')
            summary=dict(artifact=item,schema=value['schema'],rank=value['rank'],source_generation=value['source_generation'],
                full_or_Goodput_qualified=False,current_plan_binding_not_inferred=True)
            if value['schema']=='V140.failed-trace-existing-HOST-observations.v1':
                need(value['generate_returned'] is value['full463_completed'] is False
                    and value['returned_generate_results'] is value['elapsed_seconds'] is value['Goodput'] is None,
                    'partial trace cannot have synthetic results/timing/GP')
                summary.update(role=value['role'],scope_count=value['scope_count'],
                    retained_stream_events=len(value.get('stream_events',[])),
                    retained_stream_tokens=sum(len(e['token_ids']) for e in value.get('stream_events',[])),
                    missing_sections=value.get('missing_sections',{}))
            elif value['schema']=='V140.failed-decode-current-HOST-state.v1':
                need(value['local_loop_fields_current_round_binding_proved'] is False
                    and value['engine_last_worker_decode_fields_used_as_current'] is False,
                    'failed decode must retain currentness boundary')
                if 'budget_plan_binding' in value:
                    need(value['budget_plan_binding']['current_round_binding_proved'] is False
                        and value['budget_plan_binding']['current_round_grants_missing'] is True,
                        'stale plan ID equality is not current grant proof')
            partial.append(summary)
    checkpoints=[]
    for path in sorted(out.glob('rank*-candidate-checkpoint.json.gz')):
        cp=compressed(path,directory=out)
        checkpoints.append(dict(**desc(path),rank=cp['rank'],completed_phase_roles=[t['role'] for t in cp['traces']],
            original_worker_full_auditor_not_called=True,final_worker_report_not_synthesized=True))
    return dict(full463_completed=False,original_whole_four_phase_auditor_not_called_or_relaxed=True,
        final_worker_reports_available=[desc(p) for p in sorted(out.glob('rank*-local-diagnostic.json'))],
        last_errors=errors,retained_partial_evidence=partial,completed_phase_checkpoints=checkpoints,
        retained_qualification_phase_artifacts=[desc(p) for p in sorted(out.glob('rank*-qualification-*-phase*.json.gz'))],
        full463_performance=None,full_or_Goodput_qualified=False,
        missing_or_possibly_stale_current_fields_not_filled_from_engine_last_worker_values=True)



def require_bound_identity():
    for value in (CPU_SHA,FREEZE_SHA,ARCHIVE_SHA,PRE_SHA):
        need(type(value) is str and re.fullmatch('[a-f0-9]{64}',value),'draft CPU/freeze/archive/ROOT preaudit identity is still pending')
    need(type(PRE_PATH) is str and Path(PRE_PATH).is_absolute(),'draft ROOT preaudit path is pending')
    need(all(type(n) is int and n>0 for n in (ACTUAL_NEW_HOST_TESTS,LOCAL_SOURCE_COUNT,EXTERNAL_SOURCE_COUNT)),
        'draft actual focused HOST/source counts are pending')
    need(MODE_CONTRACT_BOUND is True,'draft production mode evidence field contract is pending')
    need(sha(PARENT_AUDITOR_PATH)==PARENT_AUDITOR_SHA,'source-bound revision2 parent auditor')



PRODUCTION_MODE='standalone-whole-prompt-no-new-reservation'

def production_contract_from_frozen_source():
    path=P/'no_mixed_prefill_v143.py';frozen=read(P/'PREPARED_SOURCE_HASHES_V132.json')
    need(frozen['files'][str(path)]==sha(path),'exact candidate production control helper source')
    parsed=ast.parse(path.read_text());values={}
    for node in parsed.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('MODE','ORIGINAL_SOURCES'):
            values[node.targets[0].id]=ast.literal_eval(node.value)
    need(values['MODE']==PRODUCTION_MODE,'frozen actual production mode source')
    for name,wanted in values['ORIGINAL_SOURCES'].items():
        need(frozen['files'][str(P/name)]==wanted==sha(P/name),'untouched original qualification class source '+name)
    return dict(schema='V143.production-prefill-contract.v1',mode=PRODUCTION_MODE,
        original_class_source_sha256=values['ORIGINAL_SOURCES'],
        original_qualification_class_method_unchanged=True,
        common_guard='existing untimed install-original-live stage',new_clock_calls=0,
        new_collective_calls=0,new_fence_calls=0,mixed_qualification_actors_modified=False,
        per_callback_observer_or_counter_installed=False)

def audit_standalone_phase(trace,count):
    """Additional production-only gate; independent qualification stays mixed."""
    need(trace['scope_count']==count and trace['role'] in ROLES,'production phase identity')
    need(trace['production_prefill_mode']==PRODUCTION_MODE,'actual raw production mode')
    control=trace['production_prefill_control']
    need(control['contract']==production_contract_from_frozen_source(),'actual exact production source/untimed-install contract')
    need(all(control[name] is True for name in ('instance_callback_installed','initial_empty_guard_passed','callback_owned_at_trace_end','terminal_empty_guard_passed','current_generation_only'))
        and control['terminal_guard_error'] is control['admission_callback_invocations'] is None
        and control['reservation_callback_result'] is control['native_or_class_method_replaced'] is False,
        'actual instance callback ownership/end empty state; missing invocation count stays None')
    for name in ('logical_events','begin_inputs','close_abort_identities'):
        need(type(trace[name]) is list and trace[name]==[],'production must have zero mixed reservation/submit/commit/publication/abort: '+name)
    terminal=trace['terminal_evidence']
    need(terminal['reservation_remaining'] is False and terminal['logical_terminal_request_indices']==[]
        and all(terminal[name] is None for name in ('committed_cursor','prepared_cursor','last_committed_cycle_id','initial_fixture_cursor','initial_fixture_scope')),
        'no live production mixed reservation or cursor')
    prompts=trace['tokenized_inputs'];need(type(prompts) is list and len(prompts)==count,'all original actual tokenized prompts')
    owners=[];queries=0
    for call in trace['standalone_prefill_calls']:
        ids=call['sequence_ids'];need(type(ids) is list and ids and len(ids)==len(set(ids))
            and all(type(i) is int and 0<=i<count for i in ids),'actual standalone owner rows')
        lengths=[len(prompts[i]) for i in ids]
        digests=[hashlib.sha256(json.dumps(prompts[i],separators=(',',':')).encode()).hexdigest() for i in ids]
        need(call['cached_tokens']==[0]*len(ids) and call['prompt_lengths']==lengths and call['prompt_sha256']==digests
            and call['uncached_queries']==sum(lengths) and call['precomputed_target_tokens'] is False
            and call['draft_prefill_completed'] is False,'every actual standalone call recomputes full untouched genuine prompt')
        owners.extend(ids);queries+=sum(lengths)
    need(sorted(owners)==list(range(count)),'each production owner gets one and only one whole-prompt standalone admission')
    counter=trace['decode_counters']
    need(counter['spec_rhythm_prefill_requests']==count and counter['spec_rhythm_prefill_batches']==len(trace['standalone_prefill_calls']),
        'production prefill accounting has no mixed publications')
    if trace['rank']!=0:
        need(counter['spec_rhythm_tree_rounds']==len(trace['normal_target_calls']),
            'every production target tree round is an ordinary target call, with no hidden mixed route')
    return dict(phase=trace['role'],generation=trace['source_generation'],count=count,
        actual_mixed_calls=0,actual_reservations=0,actual_mixed_publications=0,
        whole_prompt_standalone_owner_count=len(owners),whole_prompt_standalone_batches=len(trace['standalone_prefill_calls']),
        actual_prompt_queries=queries,each_owner_exactly_once=True,original_prompt_tokens_or_hashes_not_modified=True)


def audit_production_mode_envelopes(reports):
    need(MODE_CONTRACT_BOUND is True,'producer standalone mode schema/fields not yet bound')
    need(len(reports)==4 and {r['rank'] for r in reports}==set(range(4)), 'four actual producer reports')
    need(all(report['combination']['production_mode']==PRODUCTION_MODE for report in reports),
        'all actual producer combination fields explicitly select standalone mode')
    return dict(mode=PRODUCTION_MODE,four_rank_mode_selected=True,
        actual_control_fields_are_audited_in_raw_not_synthesized_from_selection=True)


def audit_standalone_run(receipt,out,mods,actual):
    paths=[out/f'rank{rank}-local-diagnostic.json' for rank in range(4)]
    reports=[read(path) for path in paths]
    mode=audit_production_mode_envelopes(reports)
    need(actual['original_frozen_audit']['production_prefill_mode']==PRODUCTION_MODE,'original frozen audit bound production mode')
    phases=[]
    for index,role in enumerate(ROLES):
        # Original complete_run already checked all-rank common raw equality,
        # original transaction arithmetic and all four trace/source/Native gates.
        item=reports[1]['traces'][index];need(item['role']==role,'exact production phase order')
        trace=compressed(item['artifact']['path'],artifact=item['artifact'],directory=out)
        phase=audit_standalone_phase(trace,463 if role=='full463' else 65)
        old=actual['original_frozen_audit']['traces'][index]['transactions']
        need(old['mixed_commits']==old['mixed_publications']==old['aborted_reservations']==0
            and old['standalone_prefill_batches']==phase['whole_prompt_standalone_batches'],
            'new production control agrees with unchanged original transaction auditor')
        original_mode=actual['original_frozen_audit']['traces'][index]['production_prefill']
        need(original_mode['mode']==PRODUCTION_MODE and original_mode['reservations']==original_mode['mixed_commits']==original_mode['mixed_publications']==0
            and original_mode['whole_prompt_standalone_owners']==phase['whole_prompt_standalone_owner_count']
            and original_mode['standalone_batches']==phase['whole_prompt_standalone_batches'],
            'original producer standalone auditor agrees with independent whole-prompt arithmetic')
        phases.append(phase)
    return dict(production_control='standalone-whole-prompt-prefill',production_mixed_disabled=True,
        same_M_mixed_21_scope_26_phase_still_required_and_not_relaxed=True,
        producer_mode_evidence=mode,phases=phases)

def audit_terminal_runtime(receipt,tag,out,helper):
    """Inspect retained original runtime-verifier logs; execute no verifier/device."""
    need(receipt['schema']=='V132.single-candidate-runtime-supervisor.v1'
        and receipt['action']=='split192-candidate-full463' and receipt['formal_Goodput'] is False
        and receipt['online_qualified'] is False,'terminal original supervisor schema/scope')
    expected=['bash',str(B/'serial-v1/runtime.sh'),'python','-B',
        '/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py',
        '/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json']
    lock=read(expected[-1]);verifiers={}
    for side in ('pre','post'):
        record=receipt[side+'_verifier'];path=P/f'{tag}-{side}-verifier.log'
        need(record==dict(exit_code=0,argv=expected),'actual '+side+' original runtime verifier command/exit')
        rows=[]
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if type(row) is dict and 'baseline_id' in row:rows.append(row)
        need(len(rows)==1,'one original '+side+' verifier JSON result')
        row=rows[0]
        need(row['status']=='pass' and row['errors']==[] and row['baseline_id']==lock['baseline_id']
            and row['checked_frozen_files']==len(lock['frozen_files_sha256'])
            and row['checked_external_files']==len(lock['external_files_sha256'])
            and row['checked_packages']==len(lock['package_versions']),'actual '+side+' original lock verification scope')
        helper.idle_record(receipt['devices_'+('before' if side=='pre' else 'after')],side)
        verifiers[side]=desc(path)
    need(receipt['child']['argv']==['bash',str(P/'launch_candidate_v132.sh'),
        '--action','split192-candidate-full463','--tag',tag],'actual original full launcher/tag')
    need(read(out/'worker-exit.json')==dict(closed=True,all_exited=True),'actual all4 worker exit')
    return dict(original_runtime_verifiers=verifiers,retained_raw_four_cards_idle_before_after_recomputed=True,
        device_query_by_this_auditor=False,recorded_child_command=receipt['child']['argv'],
        execution_lock='ROOT outer flock is an execution-log contract; this postrun audit does not remeasure an ended OS lock')


def audit_cli(args):
    require_bound_identity()  # Before opening any actual receipt or raw.
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'tag')
    need(re.fullmatch(r'[a-f0-9]{64}',args.expected_receipt_sha256),'ROOT final receipt SHA')
    rp=P/f'local-receipt-{args.tag}.json'
    need(sha(rp)==args.expected_receipt_sha256,'ROOT terminal receipt exact identity')
    receipt=read(rp)
    need(receipt['status'] in ('complete','failed') and receipt['child'].get('exit_code') is not None
        and 'devices_after' in receipt,'terminal receipt required; running data is not audited')
    need(receipt['action']=='split192-candidate-full463','exact local action')
    receipt=dict(receipt,_audit_tag=args.tag)
    out=P/'local-output'/args.tag
    with host_modules() as (mods,helpers):
        source=audit_sources(receipt,mods,helpers)
        cleanup=helpers['analyze_v119_actual_combined'].cleanup(receipt,out)
        cleanup.update(audit_terminal_runtime(receipt,args.tag,out,helpers['analyze_v144_actual_variable_B4']))
        actual=complete_run(receipt,out,mods,helpers) if receipt['status']=='complete' else failed_run(receipt,out,mods)
        if receipt['status']=='complete':actual['production_standalone_prefill_control']=audit_standalone_run(receipt,out,mods,actual)
        result=dict(schema='ROOT.V148.actual-full-cap4-audit.v1',status='passed',actual_run_status=receipt['status'],
            receipt=desc(rp),script=desc(__file__),parent_revision2_auditor=desc(PARENT_AUDITOR_PATH),helpers={name:desc(D/(name+'.py')) for name in HELPERS},
            actual_HOST_argv=sys.orig_argv,source_provenance=source,cleanup=cleanup,actual=actual,
            scope_gaps=['Four ranks cooperate in one run; they are not four independent trials.',
                'Sparse successful admission calls have aggregate counters only; owner identities are not reconstructed.',
                'Age counts eligible attempts, and is not wall waiting time or guaranteed next-cycle service.',
                'Current-generation gain observations are adaptive and repeated owner histories; no causal/heldout cap inference.',
                'Existing Native charged clocks and existing client stream wall clocks remain separate.',
                'No full own-M1 AR, arbitrary factory equivalence, historical-KV reconstruction or formal paired-baseline ratio follows.'],
            Torch_imported=False,NPU_initialized=False,physical_cards_requeried=False,model_or_query_executed=False,
            formal_Goodput=False,goal_achieved=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();destination=args.output.resolve()
    need(destination.is_relative_to(D.resolve()) and not destination.exists(),'fresh diagnostics-only audit output')
    started=time.perf_counter();code=0
    try:result=audit_cli(args)
    except BaseException:
        code=1
        result=dict(schema='ROOT.V148.actual-full-cap4-audit.v1',status='failed',tag=args.tag,
            expected_receipt_sha256=args.expected_receipt_sha256,error=traceback.format_exc(),
            actual_full463_qualified=False,formal_Goodput=False,goal_achieved=False)
    result.update(auditor=desc(__file__),audit_UTC=datetime.now(timezone.utc).isoformat(),
        audit_elapsed_seconds=time.perf_counter()-started,actual_argv=sys.orig_argv,
        actual_interpreter=dict(executable=sys.executable,version=sys.version),
        actual_environment={name:os.environ.get(name) for name in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TORCH_DEVICE_BACKEND_AUTOLOAD')},
        model_or_device_launched_by_auditor=False)
    with destination.open('x') as stream:stream.write(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    console=dict(status=result['status'],actual_run_status=result.get('actual_run_status'),
        output=str(destination),sha256=sha(destination))
    actual=result.get('actual',{})
    if actual.get('full463_completed'):
        full=actual['full463'];console.update(legacy_Native_paper_GP=full['output']['native_paper']['goodput_tokens_per_second'],
            existing_stream_wall_paper_GP=full['output']['event_wall_paper']['goodput_tokens_per_second'],
            raw_tokens_per_second=full['output']['raw_tokens_per_second'],overflow_calls=full['actual_normal_admission']['overflow_calls'])
    print(json.dumps(console))
    if code:raise SystemExit(code)


if __name__=='__main__':main()
