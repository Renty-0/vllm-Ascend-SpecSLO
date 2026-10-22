#!/usr/bin/env python3
"""One-shot, bounded, read-only progress from existing private files (stdlib only).

No candidate imports, Torch, model calls, subprocesses, device queries, waits or
output-file writes. Completed local phases do not establish whole-run success.
"""
import argparse
import datetime
import gzip
import json
from pathlib import Path
import re

MIB=1024*1024
ROLES=('warmup65','zero65-1','zero65-2','full463')


def short(value,limit=200):
    if value is None:return None
    return str(value).splitlines()[0][:limit]


def error_lines(value):
    if not isinstance(value,str):return None if value is None else short(value)
    lines=[line.strip() for line in value.splitlines() if line.strip()]
    exception=next((line for line in reversed(lines) if re.match(r'^[A-Za-z_][\w.]*?(?:Error|Exception):',line)),None)
    return dict(first_line=lines[0][:200] if lines else '',exception_line=exception[:260] if exception else None)


def load_json(path,max_raw):
    path=Path(path)
    if not path.is_file():return None,dict(state='absent')
    try:
        size=path.stat().st_size
        if size>max_raw:return None,dict(state='size-limit',bytes=size)
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rb') as stream:raw=stream.read(max_raw+1)
        if len(raw)>max_raw:return None,dict(state='expanded-size-limit',bytes=size)
        value=json.loads(raw)
        return value,dict(state='parsed-complete-JSON',bytes=size)
    except (OSError,EOFError,ValueError) as error:
        return None,dict(state='unreadable-or-partial',reason=short(type(error).__name__+': '+str(error)))


def last_jsonl(path):
    """Only a newline-terminated record counts; ignore a currently partial tail."""
    path=Path(path)
    if not path.is_file():return dict(state='absent')
    try:
        with path.open('rb') as stream:
            stream.seek(0,2);size=stream.tell();start=max(0,size-MIB);stream.seek(start);tail=stream.read(MIB)
        rows=tail.splitlines(keepends=True)
        if start and rows:rows=rows[1:]
        pending_tail=bool(rows and not rows[-1].endswith(b'\n'))
        if pending_tail:rows.pop()
        for row in reversed(rows):
            try:value=json.loads(row)
            except (ValueError,UnicodeError):continue
            if not isinstance(value,dict):continue
            memory=value.get('memory') or {}
            errors=value.get('local_errors')
            return dict(state='last-complete-record',phase=value.get('phase'),completed=value.get('completed'),
                total=value.get('total'),rank=value.get('rank'),updated_unix=value.get('updated_unix'),
                graph_entries=memory.get('graph_entries'),
                local_error_count=len(errors) if isinstance(errors,list) else None,
                first_local_error=short(errors[0]) if isinstance(errors,list) and errors else None,
                pending_partial_tail_ignored=pending_tail)
        return dict(state='no-complete-JSON-record-in-bounded-tail',bytes=size,pending_partial_tail_ignored=pending_tail)
    except OSError as error:return dict(state='unreadable',reason=short(error))


def histogram(value):
    if not isinstance(value,list):return None
    if any(type(v) is not int or v<0 for v in value):return dict(state='invalid-histogram')
    return {str(i):n for i,n in enumerate(value) if n}


def phase_summary(trace,rank):
    result={key:trace.get(key) for key in ('role','scope_count','source_generation','elapsed_seconds',
        'completed_output_tokens','active_peak','sealed_before','sealed_after','cache_released')}
    result['rank']=rank;result['scope_note']='retained local phase; no whole-run GP is inferred'
    artifact=trace.get('artifact') or {};path=Path(artifact['path']) if isinstance(artifact.get('path'),str) else None
    result['artifact']=dict(name=path.name if path else None,present=path.is_file() if path else False,
        declared_bytes=artifact.get('bytes'),bytes_match=(path.stat().st_size==artifact.get('bytes')) if path and path.is_file() else None)
    native=trace.get('native') or {};delta=native.get('delta')
    if isinstance(delta,dict):result['nonzero_Native_delta']={k:v for k,v in delta.items() if v!=0}
    if rank!=0:return result
    admission=(trace.get('actual_normal_admission') or {}).get('after')
    if isinstance(admission,dict):
        ages=admission.get('ages');age_values=[row['age'] for row in ages if isinstance(row,dict) and type(row.get('age')) is int] if isinstance(ages,list) else None
        result['normal_admission']=dict(available=admission.get('available'),generation=admission.get('generation'),
            call_count=admission.get('call_count'),overflow_calls=admission.get('overflow_calls'),
            raw_N_histogram=histogram(admission.get('raw_N_histogram')),
            admitted_N_histogram=histogram(admission.get('admitted_N_histogram')),
            positive_grant_N_histogram=histogram(admission.get('positive_grant_N_histogram')),
            retained_overflow_or_zero_grant_call_records=len(admission['calls']) if isinstance(admission.get('calls'),list) else None,
            owners_with_recorded_defer_age=len(ages) if isinstance(ages,list) else None,
            max_recorded_defer_age=max(age_values,default=None) if age_values is not None else None,
            age_scope='historical eligible attempts without positive grant, not current active owners')
    else:result['normal_admission']=dict(available=False,reason='no completed phase snapshot retained')
    gain=(trace.get('actual_gain_state') or {}).get('after')
    if not isinstance(gain,dict):result['gain_state']=dict(available=False,reason='no completed phase snapshot retained');return result
    generation=gain.get('actual_engine_generation');observed=gain.get('observed');table=[];invalid=0
    if isinstance(observed,list):
        for row in observed[:24]:
            if (not isinstance(row,dict) or row.get('slo_class') not in ('tight','normal','loose')
                or type(row.get('B')) is not int or not 1<=row['B']<=8
                or type(row.get('rounds')) is not int or row['rounds']<=0
                or type(row.get('delivered')) is not int or row['delivered']<0):invalid+=1;continue
            table.append(dict(slo_class=row['slo_class'],B=row['B'],rounds=row['rounds'],delivered=row['delivered'],
                empirical_g=round(row['delivered']/row['rounds'],6)))
    matches=generation==trace.get('source_generation')
    result['gain_state']=dict(available=isinstance(observed,list),actual_engine_generation=generation,
        phase_generation_matches=matches,observed=table if matches else None,
        invalid_sample_rows=invalid,not_a_heldout_or_independent_trial_curve=True,
        note='after snapshot only; no prior mixing, cross-generation subtraction, missing-B zero fill or cross-rank sum')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory','--candidate-directory',dest='directory',required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--max-json-mib',type=int,default=16,choices=range(1,65),metavar='1..64')
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag):parser.error('invalid private tag')
    p=Path(args.directory).resolve();maximum=args.max_json_mib*MIB
    candidates=[p/f'local-receipt-{args.tag}.json',p/f'receipt-{args.tag}.json']
    rp=next((path for path in candidates if path.is_file()),candidates[0]);receipt,rstate=load_json(rp,maximum)
    if not isinstance(receipt,dict):receipt={}
    out=p/'local-output'/args.tag
    if not out.is_dir() and (p/'actual'/args.tag).is_dir():out=p/'actual'/args.tag
    progress=receipt.get('startup_progress') or {}
    startup=Path(progress['actual_directory']) if isinstance(progress.get('actual_directory'),str) else p/'startup-progress-by-tag'/args.tag
    # Never show another tag through the mutable legacy startup symlink.
    legacy=p/'startup-progress'
    if not startup.is_dir() and legacy.is_symlink() and legacy.resolve()==startup.resolve():startup=legacy
    summary=dict(schema='HOST.readonly-tree-progress.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        directory=str(p),tag=args.tag,receipt_file=str(rp),receipt_read=rstate,
        receipt=dict(status=receipt.get('status'),terminal_status=receipt.get('status') in ('complete','failed'),
            elapsed_seconds=receipt.get('elapsed_seconds'),child_pid=(receipt.get('child') or {}).get('pid'),
            child_exit_code=(receipt.get('child') or {}).get('exit_code'),
            error_count=len(receipt['errors']) if isinstance(receipt.get('errors'),list) else None,
            first_error=error_lines(receipt['errors'][0]) if isinstance(receipt.get('errors'),list) and receipt['errors'] else None,
            owned_remaining_count=len(receipt['owned_remaining']) if isinstance(receipt.get('owned_remaining'),list) else None,
            all_idle_from_existing_terminal_receipt=(receipt.get('devices_after') or {}).get('all_idle'),
            experimental_sources_unchanged=receipt.get('experimental_sources_unchanged'),
            external_sources_unchanged=receipt.get('external_sources_unchanged')),
        startup_directory=str(startup),ranks=[],rank0_completed_phase_snapshots=[],
        scope='Progress only. Existing complete JSON is not an independent qualification audit. No Goodput is computed; warm/zero are not full463.')
    phase_rank_sets={role:[] for role in ROLES}
    for rank in range(4):
        checkpoint,cstate=load_json(out/f'rank{rank}-candidate-checkpoint.json.gz',maximum)
        final,fstate=load_json(out/f'rank{rank}-local-diagnostic.json',maximum)
        source=final if isinstance(final,dict) else checkpoint
        traces=source.get('traces',[]) if isinstance(source,dict) else []
        if not isinstance(traces,list):traces=[]
        names=[t.get('role') for t in traces if isinstance(t,dict)]
        for role in ROLES:
            if role in names:phase_rank_sets[role].append(rank)
        err,estate=load_json(out/f'rank{rank}-last-error.json',min(maximum,2*MIB));index=err.get('failure_evidence') if isinstance(err,dict) else None
        artifacts=index.get('artifacts') if isinstance(index,dict) else None
        current,sstate=load_json(out/f'rank{rank}-current-stage.json',min(maximum,2*MIB))
        row=dict(rank=rank,startup=last_jsonl(startup/f'finite-inventory-rank{rank}.jsonl'),
            qualification_phase_files_present=len(list(out.glob(f'rank{rank}-qualification-*.json.gz'))),
            pre_validator_chain_file_present=(out/f'rank{rank}-pre-audit-same-M-qualification.json.gz').is_file(),
            post_validator_chain_file_present=(out/f'rank{rank}-same-M-qualification.json.gz').is_file(),
            checkpoint_read=cstate,final_report_read=fstate,retained_phase_names=names,
            current_stage={k:current.get(k) for k in ('stage','label','status','sequence')} if isinstance(current,dict) else None,
            last_error_read=estate,error=error_lines(err.get('error')) if isinstance(err,dict) else None,
            failure_index=dict(source_generation=index.get('source_generation'),missing=index.get('missing'),
                retention_error_count=len(index['retention_errors']) if isinstance(index.get('retention_errors'),list) else None,
                artifacts=[{k:item.get(k) for k in ('kind','source_generation','bytes')} for item in artifacts[:8]] if isinstance(artifacts,list) else None) if isinstance(index,dict) else None)
        summary['ranks'].append(row)
        if rank==0:summary['rank0_completed_phase_snapshots']=[phase_summary(trace,rank) for trace in traces if isinstance(trace,dict) and trace.get('role') in ROLES]
    summary['retained_phase_ranks']=phase_rank_sets
    exited,exit_state=load_json(out/'worker-exit.json',min(maximum,2*MIB))
    summary['worker_exit']=dict(read=exit_state,closed=exited.get('closed'),all_exited=exited.get('all_exited')) if isinstance(exited,dict) else dict(read=exit_state,closed=None,all_exited=None)
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=None if args.compact else 2))

if __name__=='__main__':main()
