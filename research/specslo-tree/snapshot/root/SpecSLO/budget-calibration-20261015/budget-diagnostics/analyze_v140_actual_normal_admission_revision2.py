"""Terminal-only V140 HOST audit. Never launches, queries, or imports a device.

Prepared before terminal; ROOT supplies the final receipt SHA. Failed/partial
artifacts remain failed/partial and cannot enter the original full-run auditor.
"""
import argparse
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

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-normal-admission-fix-v140'
MANIFEST=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl')
CPU_SHA='50b0077a75b6adbe740983868ae82f48c4c116108bd546bb5e68527097b49a3b'
FREEZE_SHA='72ed95be9afe492543a616b0e4e587f2f4019cad9b4e666f92130dae19713f80'
ARCHIVE_SHA='b52032ea16188bedda2d3dfd9ec794ee225e8d94e8a11f055ef73b219a053e3f'
PRE_SHA='06732c8d458fe2a066742b8c2d46ec1a803dd264b73dbc5e5e79220567d95ad6'
HELPERS={'analyze_v132_actual_failed_metadata':'3b35d0fbe799558dc15295c8337f3cdb556075fe958cadf6f20dc2a1826e7173',
 'analyze_v96_actual_capacity64':'b9e420f70d84966f0c1b5edef2721b6213756506c27e3fd9ec27be3e774ade6b',
 'analyze_v119_actual_combined':'903ae516280dacf68c1458358d4019fefa2a0a245819a4cf2e9fd5d1a768da5e'}
ROLES=('warmup65','zero65-1','zero65-2','full463')
PRIORITY={'tight':3,'normal':2,'loose':1}
RULE='V140.normal-class-priority-defer-age-owner-order.v1'


def need(value,message):
    if not value:raise ValueError('V140 external: '+message)


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
            'private_storage_v132','check_v132')
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
            and all(type(entry['B']) is int and 1<=entry['B']<=(3 if len(admitted)>8 else 5) for entry in grants),
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
    need(sha(cp)==CPU_SHA and sha(fp)==FREEZE_SHA,'pinned V140 CPU/freeze')
    cpu,frozen=read(cp),read(fp)
    need(cpu['preparation_epoch']=='V140' and cpu['status']=='passed' and cpu['tests_run']==13
        and cpu['errors']==cpu['failures']==0 and cpu['parent8_actual_HOST_proof_verified'] is True
        and cpu['parent23_actual_HOST_proof_verified'] is True and cpu['original8_or23_suite_executed'] is False
        and cpu['Torch_imported'] is cpu['NPU_initialized'] is cpu['model_initialized'] is cpu['NPU_or_ACL_queried'] is False,
        'actual13 HOST versus inherited8+23 scope')
    need(cpu['input_hashes']==cpu['input_hashes_after']==frozen['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==frozen['external_files']
        and frozen['CPU_sha256']==CPU_SHA and len(frozen['files'])==343 and len(frozen['external_files'])==4752,
        'actual HOST before/after source closure')
    expected={**frozen['files'],**frozen['external_files'],str(cp):CPU_SHA,str(fp):FREEZE_SHA}
    provenance=helpers['analyze_v132_actual_failed_metadata']
    count=provenance.verify_map(expected)
    need(mods['source_snapshot_v132'].collect()==frozen['files'],'strict current static collector')
    need(receipt['source_hashes']==expected and receipt['readiness']['identity']==dict(CPU_sha256=CPU_SHA,freeze_sha256=FREEZE_SHA),
        'actual source snapshot/readiness identity')
    legacy=read(P/'ORIGINAL_PRIOR_BINDING_V78.json')['observed']
    need(receipt['external_input_hashes']=={**legacy,**frozen['external_files']},'receipt external plus historical reference closure')
    provenance.verify_map(receipt['external_input_hashes'])
    need(sha(receipt['source_snapshot'])==receipt['source_snapshot_sha256'],'actual runtime source archive SHA')
    runtime=provenance.archive(receipt['source_snapshot'],expected)
    pre_path=D/'ROOT_V140_PRE_NPU_SOURCE_AND_HOST_AUDIT.json';need(sha(pre_path)==PRE_SHA,'ROOT independent preaudit identity')
    pre=read(pre_path)
    need(pre['status']=='passed' and pre['CPU_sha256']==CPU_SHA and pre['freeze_sha256']==FREEZE_SHA
        and pre['actual_HOST_tests']==13 and pre['local_source_count']==343 and pre['external_source_count']==4752
        and pre['readiness_exit_code']==0 and pre['archive_sha256']==ARCHIVE_SHA,'ROOT preaudit source/readiness scope')
    need(sha(pre['archive'])==ARCHIVE_SHA,'prepared archive original bytes')
    prepared=provenance.archive(pre['archive'],expected)
    runtime_shell=B/'serial-v1/runtime.sh'
    sourced=re.findall(r'^source (/[^ \n]+)',runtime_shell.read_text(),re.M)
    need(len(sourced)==2 and pre['runtime_shells']=={str(runtime_shell):sha(runtime_shell),**{p:sha(p) for p in sourced}},
        'real runtime source paths and ROOT before-SHA')
    need(receipt['actual_sourced_runtime_shells_before']==receipt['actual_sourced_runtime_shells_after']
        and [row['path'] for row in receipt['actual_sourced_runtime_shells_after']]==sourced
        and all(sha(row['path'])==row['sha256'] for row in receipt['actual_sourced_runtime_shells_after']),
        'actual sourced runtime shell before/after')
    need(receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'runtime aliases before/after')
    for row in receipt['runtime_aliases_after']:
        path=Path(row['path']);need(path.is_symlink() and path.exists() is row['exists']
            and str(path.resolve())==row['resolved'] and os.readlink(path)==row['link'],'actual retained runtime alias')
    outputs=mods['source_snapshot_v132'].actual_runtime_outputs()
    need(outputs==receipt['actual_runtime_outputs'],'strict known-fusion runtime output closure')
    storage=mods['private_storage_v132'].snapshot(receipt['_audit_tag']);mods['private_storage_v132'].require_limits(storage)
    mods['private_storage_v132'].require_limits(receipt)
    return dict(passed=True,CPU=desc(cp),freeze=desc(fp),ROOT_pre_NPU_audit=desc(pre_path),
        local_sources=343,external_sources=4752,all_current_inputs_SHA_checked=count,
        runtime_source_archive=runtime,prepared_source_archive=prepared,
        parent8_and23_actual_HOST_proof_reused_by_SHA_not_rerun=True,
        actual_sourced_runtime_shells=receipt['actual_sourced_runtime_shells_after'],runtime_aliases=receipt['runtime_aliases_after'],
        known_runtime_outputs=outputs,unknown_source_outputs_rejected=True,
        storage=dict(terminal_receipt=dict(attempt_private_bytes=receipt['attempt_private_bytes'],
            aggregate_private_bytes=receipt['aggregate_private_bytes'],storage_limits=receipt['storage_limits']),
            current_read_only_HOST_snapshot=storage,matching_measurement_time_not_assumed=True),
        historical_reference_SHA_does_not_prove_dynamic_import_events=True,
        device_numerics_not_reexecuted_by_HOST=True)


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
    phases=[];previous_gain=None;previous_native=[r['mixed_same_M_qualification']['native']['after'] for r in reports]
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
        phases=phases,full463=phases[-1],final_worker_reports=[desc(p) for p in paths],completed_record=desc(out/'completed.json'),
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
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
        actual=complete_run(receipt,out,mods,helpers) if receipt['status']=='complete' else failed_run(receipt,out,mods)
        result=dict(schema='ROOT.V140.actual-normal-admission-audit.v1',status='passed',actual_run_status=receipt['status'],
            receipt=desc(rp),script=desc(__file__),helpers={name:desc(D/(name+'.py')) for name in HELPERS},
            actual_HOST_argv=sys.orig_argv,source_provenance=source,cleanup=cleanup,actual=actual,
            scope_gaps=['Four ranks cooperate in one run; they are not four independent trials.',
                'Sparse successful admission calls have aggregate counters only; owner identities are not reconstructed.',
                'Age counts eligible attempts, and is not wall waiting time or guaranteed next-cycle service.',
                'Current-generation gain observations are adaptive and repeated owner histories; no causal/heldout cap inference.',
                'Existing Native charged clocks and existing client stream wall clocks remain separate.',
                'No full own-M1 AR, arbitrary factory equivalence, historical-KV reconstruction or formal paired-baseline ratio follows.'],
            Torch_imported=False,NPU_initialized=False,physical_cards_requeried=False,model_or_query_executed=False,
            formal_Goodput=False,goal_achieved=False)
    destination=Path(args.output)
    with destination.open('x') as stream:stream.write(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    console=dict(status='passed',actual_run_status=receipt['status'],output=str(destination),sha256=sha(destination),full463_completed=actual['full463_completed'])
    if actual['full463_completed']:
        full=actual['full463'];console.update(legacy_Native_paper_GP=full['output']['native_paper']['goodput_tokens_per_second'],
            existing_stream_wall_paper_GP=full['output']['event_wall_paper']['goodput_tokens_per_second'],
            raw_tokens_per_second=full['output']['raw_tokens_per_second'],overflow_calls=full['actual_normal_admission']['overflow_calls'])
    print(json.dumps(console))

if __name__=='__main__':main()
