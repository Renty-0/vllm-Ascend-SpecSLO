"""Read completed tree traces only: exact owner/prefix join and frozen-path bounds.

This is not a runtime hook, workload qualification, or causal estimate of g(B).
It imports only the Python standard library and never opens a tensor artifact.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys


def need(value,message):
    if not value:raise ValueError('retained stream identification: '+message)


def desc(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def read(path):return json.loads(Path(path).read_bytes())


def unpack(artifact,directory):
    path=Path(artifact['path']).resolve();need(path.is_relative_to(directory),'artifact escaped terminal run')
    need(desc(path)=={key:artifact[key] for key in ('path','bytes','sha256')},'actual artifact descriptor differs')
    need(path.stat().st_size<=64*1024**2,'compressed artifact bound')
    with gzip.open(path,'rb') as stream:raw=stream.read(256*1024**2+1)
    need(len(raw)==artifact['uncompressed_bytes'] and len(raw)<=256*1024**2,'expanded artifact bound')
    return json.loads(raw)


def analyze(trace,envelope):
    count=trace['scope_count'];prompts=trace['tokenized_inputs'];requests=trace['requests']
    need(len(prompts)==len(requests)==len(trace['sampling'])==count,'complete actual input scope')
    need(all(s['max_tokens']==256 and s['ignore_eos'] is True for s in trace['sampling']),
         'identification currently requires actual complete256/ignoreEOS sampling')
    calls=defaultdict(list);mixed_issues=[];route_rows=Counter()
    def add(owner,budget,prefix,route,identity):
        need(type(owner) is int and 0<=owner<count and type(budget) is int and 1<=budget<=5
            and type(prefix) is int and prefix>=0,'real owner/B/prefix domain')
        calls[owner,prefix].append(dict(B=budget,route=route,identity=identity));route_rows[route]+=1
    for index,call in enumerate(trace['normal_target_calls']):
        n=call['N'];ids=call['sequence_ids'];budgets=call['budgets'];contexts=call['contexts']
        need(type(n) is int and 1<=n<=32 and len(budgets)==len(contexts)==n
            and len(ids)==call['physical_S'] and ids[n:]==list(range(463,463+len(ids)-n))
            and call['physical_query_partition'][:n]==[b+1 for b in budgets],
            'normal real prefix versus graph scratch geometry')
        for i,b,p in zip(ids[:n],budgets,contexts,strict=True):add(i,b,p,'ordinary-target-call',index)
    begins=defaultdict(list)
    for begin in trace['begin_inputs']:begins[begin['cycle_id']].append(begin)
    commits=Counter(event['cycle_id'] for event in trace['logical_events'] if event['kind']=='commit')
    for event in trace['logical_events']:
        if event['kind']!='plan':continue
        matches=[begin for begin in begins[event['cycle_id']]
                 if begin['target_ids']==event['target_ids'] and begin['budgets']==event['budgets']]
        if len(matches)!=1 or commits[event['cycle_id']]!=1:
            mixed_issues.append(dict(cycle_id=event['cycle_id'],matching_begin_inputs=len(matches),
                                     actual_commits=commits[event['cycle_id']]))
            continue
        begin=matches[0]
        for i,b,p in zip(begin['target_ids'],begin['budgets'],begin['contexts'],strict=True):
            add(i,b,p,'mixed-target-call',event['cycle_id'])
    seen=Counter();finished=Counter();used=Counter();joined=[];unmatched=[];initial=[]
    for index,event in enumerate(trace['stream_events']):
        need(set(event)=={'request_index','request_id','token_ids','finished','elapsed_seconds'},'actual stream ABI')
        i=event['request_index'];need(type(i) is int and 0<=i<count and event['request_id']==requests[i]['request_id'],
                                     'stream owner identity')
        before=seen[i];delivered=len(event['token_ids']);seen[i]+=delivered
        need(delivered>0 and seen[i]<=256 and not finished[i]
            and event['finished'] is (seen[i]==256),'complete output/terminal accounting')
        finished[i]+=int(event['finished'])
        if before==0:
            need(delivered==1,'first-token admission event must be isolated from target verdict')
            initial.append(index);continue
        key=(i,len(prompts[i])+before-1);matches=calls.get(key,[])
        if len(matches)!=1:
            unmatched.append(dict(event_index=index,owner=i,context=key[1],candidate_matches=len(matches)))
            continue
        used[key]+=1;call=matches[0];budget=call['B']
        need(delivered<=min(budget,4)+1,'delivered verdict exceeds original tree depth/budget')
        joined.append(dict(owner=i,event_index=index,B=budget,delivered=delivered,
            finished=event['finished'],remaining_before=256-before,slo_class=requests[i]['slo_class'],
            route=call['route']))
    need(len(initial)==count and all(seen[i]==256 and finished[i]==1 for i in range(count)),
         'all actual requests need complete256 and exactly one terminal event')
    duplicates=[dict(owner=i,context=p,records=len(values)) for (i,p),values in calls.items() if len(values)!=1]
    unused=[dict(owner=i,context=p) for i,p in calls if used[i,p]!=1]
    complete=not(mixed_issues or duplicates or unmatched or unused)
    observed=defaultdict(lambda:[0,0]);censored=Counter();budget_rows=defaultdict(list)
    for row in joined:
        b=row['B'];budget_rows[b].append(row)
        if row['remaining_before']<min(b,4)+1:censored[b]+=1
        else:
            value=observed[row['slo_class'],b];value[0]+=1;value[1]+=row['delivered']
    gain=envelope['actual_gain_state']['after']
    need(gain['actual_engine_generation']==trace['source_generation'],'gain source generation')
    actual={(entry['slo_class'],entry['B']):[entry['rounds'],entry['delivered']] for entry in gain['observed']}
    equal=complete and dict(observed)==actual
    tables=[]
    for cls in ('tight','normal','loose'):
        rows=[r for r in joined if r['slo_class']==cls and r['B'] in (3,4) and not r['finished']]
        n=len(rows)
        tables.append(dict(slo_class=cls,events=n,B_counts={str(b):sum(r['B']==b for r in rows) for b in (3,4)},
            actual_delivered_sum=sum(r['delivered'] for r in rows),
            fixed_observed_path_B1_truncation_sum=sum(min(r['delivered'],2) for r in rows),
            fixed_observed_path_B2_truncation_sum=sum(min(r['delivered'],3) for r in rows),
            actual_delivered_mean=sum(r['delivered'] for r in rows)/n if n else None,
            fixed_observed_path_B1_truncation_mean=sum(min(r['delivered'],2) for r in rows)/n if n else None,
            fixed_observed_path_B2_truncation_mean=sum(min(r['delivered'],3) for r in rows)/n if n else None))
    return dict(role=trace['role'],count=count,source_generation=trace['source_generation'],
        identification=dict(exact_join_complete=complete,initial_prefill_events_excluded=len(initial),
            actual_target_stream_events=len(trace['stream_events'])-len(initial),unique_target_owner_prefix_keys=len(calls),
            matched_target_events=len(joined),route_rows=dict(route_rows),duplicate_keys=duplicates,
            missing_or_ambiguous_events=unmatched,unused_or_multiply_used_call_keys=unused,mixed_source_issues=mixed_issues),
        actual_B_table=[dict(B=b,events=len(rows),finished_events=sum(r['finished'] for r in rows),
            horizon_censored_events=censored[b],delivered_histogram=dict(sorted(Counter(r['delivered'] for r in rows).items())))
            for b,rows in sorted(budget_rows.items())],
        fixed_observed_path_only=dict(filter='actual B3/B4 and finished=False; exclude every B5 fork and all terminal events',
            events=sum(t['events'] for t in tables),by_class=tables,
            not_actual_g1_or_g2=True,not_new_model_graph_or_KV_history_execution=True,
            current_budget_or_current_context_selection_bias_removed=False),
        actual_gain_reconstruction=dict(exact_equal=equal,whole_join_needed_for_equality=True,
            source_rule='remaining_before >= min(B,tree_depth4)+1; aggregate actual delivered, not accepted nodes',
            recomputed=[dict(slo_class=cls,B=b,rounds=v[0],delivered=v[1]) for (cls,b),v in sorted(observed.items())],
            actual_saved_observed=gain['observed'],missing_actual_B=[b for b in (1,2) if not any(key[1]==b for key in actual)],
            absent_actual_class_B_not_filled_with_zero=True,untruncated_accepted_nodes_not_needed_for_this_source_rule=True),
        remaining_gaps=['No changed-B model, kernel, MC2, cohort or downstream KV history was executed.',
            'Stream lacks B/topology; budget identity comes from exact observed owner/prefix plus pinned producer source.',
            'Per-row raw parent arrays and B5 side-anchor choices were not retained; B1..4 linearity follows the deterministic nested-primary source.',
            'Ordinary/mixed target execution route does not identify normal/eager origin of each proposal.',
            'Terminal untruncated verifier length and accepted-node vector were not retained; fixed-path table excludes all terminal events.',
            'The fixed-path sums do not identify a causal gain curve or justify replacing the current initial prior.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch-directory',type=Path,required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--role',default='full463')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    need(sys.version_info[:2]==(3,11),'retained Python3.11 required')
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'tag domain')
    epoch=args.epoch_directory.resolve();receipt_path=epoch/('local-receipt-'+args.tag+'.json')
    need(desc(receipt_path)['sha256']==args.expected_receipt_sha256,'ROOT terminal receipt identity')
    receipt=read(receipt_path);need(receipt['status']=='complete' and receipt['local_execution_complete'] is True
        and receipt['errors']==[] and receipt['child']['exit_code']==0,'completed original receipt only')
    directory=(epoch/'local-output'/args.tag).resolve();paths=[directory/f'rank{r}-local-diagnostic.json' for r in range(4)]
    need(receipt['report_paths']==[str(p) for p in paths],'original final four report paths')
    reports=[read(path) for path in paths];envelopes=[];traces=[]
    for rank,report in enumerate(reports):
        need(report['rank']==rank,'actual worker rank')
        selected=[t for t in report['traces'] if t['role']==args.role];need(len(selected)==1,'unique completed phase')
        envelopes.append(selected[0]);traces.append(unpack(selected[0]['artifact'],directory))
    leader=traces[1]
    common=('requests','tokenized_inputs','sampling','source_generation','logical_events','begin_inputs')
    need(all(all(trace[key]==leader[key] for key in common) for trace in traces),'four-rank actual join inputs')
    need(traces[0]['normal_target_calls']==[] and all(t['normal_target_calls']==leader['normal_target_calls'] for t in traces[1:]),
         'actual draft/three-target call record ownership')
    need(all(e['actual_gain_state']==envelopes[1]['actual_gain_state'] for e in envelopes),'four-rank actual gain snapshots')
    freeze_path=epoch/'PREPARED_SOURCE_HASHES_V132.json';frozen=read(freeze_path)
    need(receipt['readiness']['identity']['freeze_sha256']==desc(freeze_path)['sha256'],'actual frozen epoch identity')
    names=('runtime_tree_decode.py','production_probe_v101.py','poisson_coverage_probe.py','live_poisson_probe.py',
           'backbone_selection.py','budget_policy.py','tree_entry.py','gain_state_v140.py','tree-gain-model.json')
    sources={}
    for name in names:
        path=epoch/name;item=desc(path);need(frozen['files'][str(path)]==item['sha256'],'run-pinned producer source '+name)
        sources[name]=item
    native=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
    item=desc(native);need(frozen['external_files'][str(native)]==item['sha256'],'run-pinned Native stream producer')
    sources['Native.native_engine.py']=item
    model=read(epoch/'tree-gain-model.json');need(model.get('tree_depth',4)==4,'original depth4 gain censor scope')
    result=dict(schema='HOST.retained-tree-stream-budget-identifiability.v1',status='analysis-complete',
        actual_HOST_argv=sys.orig_argv,script=desc(__file__),receipt=desc(receipt_path),freeze=desc(freeze_path),
        reports=[desc(path) for path in paths],artifacts=[e['artifact'] for e in envelopes],source_subset=sources,
        four_rank_actual_inputs_and_gain_equal=True,source_line_anchors=dict(runtime_tree_decode=[479,854,921,950,969],
            poisson_coverage_probe=[186,198],live_poisson_probe=[228,240],backbone_selection=[7,27],
            budget_policy=[44,57],tree_entry=[217,221],Native_native_engine=[6320,6350]),
        observation=analyze(leader,envelopes[1]),not_a_replacement_for_original_full_chain_audit=True,
        policy_or_actual_gain_files_modified=False,Torch_imported=False,NPU_initialized=False,
        physical_cards_queried=False,model_or_query_executed=False,causal_gain_or_performance_qualified=False)
    need(not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','numpy') for n in sys.modules),'HOST-only imports')
    with args.output.open('x') as stream:stream.write(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(output=desc(args.output),exact_join_complete=result['observation']['identification']['exact_join_complete'],
        actual_gain_reconstruction_equal=result['observation']['actual_gain_reconstruction']['exact_equal']),sort_keys=True))


if __name__=='__main__':main()
