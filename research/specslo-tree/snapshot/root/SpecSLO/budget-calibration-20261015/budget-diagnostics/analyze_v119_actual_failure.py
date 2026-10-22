"""Read-only terminal V119 failure audit; no Torch, NPU or evidence rewriting."""
from __future__ import annotations
import argparse, copy, gzip, hashlib, importlib.util, json, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

D=Path(__file__).resolve().parent
HELPER=D/'analyze_v119_actual_combined.py'
HELPER_SHA='903ae516280dacf68c1458358d4019fefa2a0a245819a4cf2e9fd5d1a768da5e'

def load_helper():
    if hashlib.sha256(HELPER.read_bytes()).hexdigest()!=HELPER_SHA:
        raise ValueError('Pinned independent provenance helper changed')
    spec=importlib.util.spec_from_file_location('_v119_terminal_source_helper',HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def read_phase(h,path,rank,scope,index):
    with gzip.open(path,'rb') as stream: raw=stream.read(256*1024**2+1)
    h.need(len(raw)<=256*1024**2,'phase expansion limit')
    value=json.loads(raw);h.finite(value)
    h.need(set(value)=={'schema','rank','scope','phase_index','complete_structure_qualification',
        'online_qualified','formal_Goodput','completed_actual_phase'},'exact partial phase schema')
    h.need(value['schema']=='V101.partial-local-phase-evidence.v1' and value['rank']==rank
        and type(value['rank']) is int and value['scope']==scope.name and value['phase_index']==index
        and value['complete_structure_qualification'] is value['online_qualified'] is value['formal_Goodput'] is False,
        'phase checkpoint scope/qualification boundary')
    return value['completed_actual_phase'],dict(**h.desc(path),uncompressed_bytes=len(raw),
        uncompressed_sha256=hashlib.sha256(raw).hexdigest())


def negative_checks(h,validator,draft,target,scope):
    cases=[('draft_nonfinite_marker',0,lambda d:d['draft_numeric'].__setitem__('all28_query_KV_original_matcher',False)),
        ('count_scope',0,lambda d:d['draft_numeric'].__setitem__('normal_request_count',17)),
        ('cursor',1,lambda d:d.__setitem__('actual_committed_cursor',0)),
        ('commit_verdict',1,lambda d:d['original_cycle'].__setitem__('actual_numerical_failed',1)),
        ('owner_epoch',1,lambda d:d['original_cycle']['actual_prefix_epochs'].__setitem__('0',999)),
        ('target_hidden',1,lambda d:d['target_numeric'].__setitem__('all197_hidden_original_matcher',False)),
        ('restored_flag',1,lambda d:d['target_numeric']['original_native']['restored_entry'].__setitem__('runtime_validated',False)),
        ('MC2_counter',1,lambda d:d['actual_MC2']['candidate']['dispatch']['delta'].__setitem__('fallback',1))]
    passed=[]
    for name,rank,mutate in cases:
        value=copy.deepcopy(draft if rank==0 else target);mutate(value)
        try:validator.phase_record(value,scope,0,'draft' if rank==0 else 'target',rank=rank)
        except (ValueError,TypeError,KeyError,RuntimeError,AssertionError):passed.append(name)
        else:raise ValueError('Tampered phase was accepted: '+name)
    return passed


def run(tag):
    h=load_helper();h.TAG=tag
    P=h.P;root=P/'local-output'/tag
    receipt_path=P/f'local-receipt-{tag}.json';receipt=h.read(receipt_path)
    h.need(receipt['status']=='failed' and receipt['child']['exit_code']==1 and 'devices_after' in receipt,
        'This auditor only accepts the ROOT-notified terminal failure')
    h.need(receipt['action']=='combined-full463' and receipt['formal_Goodput'] is receipt['online_qualified'] is False,
        'terminal action/qualification scope')
    with h.modules() as mods:
        source=h.audit_sources(receipt,mods);cleanup=h.cleanup(receipt,root)
        scopes=mods['local_qualification_scope'].SCOPES
        validator=mods['bounded_report_audit_v78']
        expected={root/f'rank{rank}-qualification-{scope.name}-phase{index}.json.gz'
            for rank in range(4) for scope in scopes for index in range(len(scope.phases()))}
        h.need(len(scopes)==21 and sum(len(s.phases()) for s in scopes)==26 and len(expected)==104
            and set(root.glob('rank*-qualification-*-phase*.json.gz'))==expected,'Exact104 retained phase set')
        signatures={};all_phases=[];first=[];target_layers=0;draft_layers=0
        for rank in range(4):
            role='draft' if rank==0 else 'target';rank_signatures=[]
            for scope in scopes:
                scope_signatures=[]
                for index,geometry in enumerate(scope.phases()):
                    path=root/f'rank{rank}-qualification-{scope.name}-phase{index}.json.gz'
                    phase,artifact=read_phase(h,path,rank,scope,index)
                    signature=validator.phase_record(phase,scope,index,role,rank=rank)
                    if rank<2 and scope is scopes[0] and index==0:first.append(phase)
                    if index:
                        previous=scope_signatures[-1]
                        h.need(signature['target_home']==1-previous['target_home'],'Actual home rotation')
                        h.need(all(new==old+emitted for old,new,emitted in zip(previous['actual_logical_tokens'],
                            signature['actual_logical_tokens'],signature['committed_tokens'],strict=True)),
                            'Actual retained logical prefix recurrence')
                    scope_signatures.append(signature)
                    numeric=phase['draft_numeric'] if rank==0 else phase['target_numeric']
                    all_phases.append(dict(rank=rank,role=role,scope=scope.name,phase_index=index,
                        N=geometry['N'],M=geometry['physical_M'],S=geometry['physical_S'],
                        prompt_start=geometry['prompt_start'],prompt_end=geometry['prompt_end'],final=geometry['final'],
                        actual_prefix_epochs=phase['original_cycle']['actual_prefix_epochs'],
                        original_native_changed_input_gate=(None if rank==0 else numeric['original_native']['cold_runtime_gate_closed']),
                        original_validator_phase_record_passed=True,artifact=artifact))
                    if rank==0:draft_layers+=28
                    else:target_layers+=64
                rank_signatures.append(scope_signatures)
            signatures[rank]=rank_signatures
        h.need(all(signatures[r]==signatures[0] for r in range(1,4)),
            'All4 retained logical token/path/publication/geometry signatures')
        negatives=negative_checks(h,validator,*first,scopes[0])
        errors=[];error_texts=[]
        for rank in range(4):
            path=root/f'rank{rank}-last-error.json';error=h.read(path)
            h.need(error['rank']==rank and error['action']=='combined-full463'
                and error['online_qualified'] is error['formal_Goodput'] is False,'Exact4 terminal worker errors')
            text=error['error'];error_texts.append(text)
            h.need('shape_scope_v101.py' in text and 'qualify_structures' in text
                and 'V113/V101-complete-actual-structure-and-original-protocol-gate' in text
                and all("'rank': "+str(r) in text for r in range(4))
                and text.count('BoundaryRejected: Original reservation source/capacity/epoch packet differs')==4,
                'Actual common all4 failure packet')
            errors.append(h.desc(path))
        absent=['completed.json']+[f'rank{r}-{name}' for r in range(4) for name in (
            'local-diagnostic.json','same-M-qualification.json.gz','paired-checkpoint.json.gz',
            'original-full463-actual-trace.json.gz','combined-full463-actual-trace.json.gz')]
        h.need(all(not (root/name).exists() for name in absent),'Unexpected aggregate/full evidence requires a different audit')
        h.need(not any('actual-trace' in p.name for p in root.iterdir()),'No warm/zero/full trace expected')
        log=P/f'local-diagnostic-{tag}.log';logtext=log.read_text(errors='replace')
        h.need('Original reservation source/capacity/epoch packet differs' in logtext,'Original child traceback log')
        return dict(status='failed_attempt_evidence_audited',original_run_status='failed',
            original_run_errors=receipt['errors'],receipt=h.desc(receipt_path),
            elapsed_seconds=receipt['elapsed_seconds'],source_provenance=source,cleanup=cleanup,
            original_child_log=h.desc(log),worker_error_artifacts=errors,
            actual_failure_stage='V113/V101-complete-actual-structure-and-original-protocol-gate',
            actual_failure='BoundaryRejected: Original reservation source/capacity/epoch packet differs',
            failure_synchronized_across_all4=True,all104_phase_records_passed_frozen_phase_record=True,
            scope_count_with_all_phase_files=21,retained_completed_phase_count=104,
            distinct_actual_phases_per_rank=26,retained_phases=all_phases,
            all4_logical_token_path_publication_geometry_signatures_equal=True,
            per_phase_original_prefix_epoch_checks_passed=True,
            phase_marker_layer_count_sum=dict(draft=draft_layers,target=target_layers),
            meaningful_phase_negative_checks=negatives,
            numerical_scope='Actual actors recorded original matcher and exact head/top2/commit gates for each retained phase; HOST re-evaluated all original phase_record checks and raw counters. No hidden/KV tensor bytes were retained for independent offline numerical recomputation.',
            startup=dict(original_source_orders_audit_actual_startup_before_qualification=True,
                later_actual_phase_files_observed=True,complete_startup_report_retained=False,
                original_startup_inventory_and_MC2_report_independently_reaudited=False),
            reservation_epoch_packet=dict(actual_packet_bytes_retained=False,full_boundary_event_lists_retained=False,
                all_phase_original_prefix_epochs_retained=True,
                allocation_epoch_nonce_reserved18pages_livepages_recomputable=False,
                source_contract_mismatch='Producer observe returns tuple active/pending/reserved_pages/live_pages while aggregate JSON-domain validator requires list; actual failed in-memory packets are unavailable.',
                actual_capacity_or_epoch_corruption_proven=False),
            missing=['complete actor chain','reservation allocator epoch/nonce/page packets','complete ordered common boundary events',
                'terminal manager and publication duplicate guards','next-normal/published draft probes','8 lifecycle case reports',
                'full startup report','actual before/after whole Native inventory','per-generation Native21 counters',
                'warm65 and zero65 traces','paired463 token streams and timers'],
            full_chain_or_lifecycle_qualification_granted=False,complete_structure_qualification=False,
            actual_production_generations_completed=0,actual_full463_runs_completed=0,
            performance_result=None,Goodput=None,formal_Goodput=False,online_qualified=False,
            independent_own_AR_qualified=False,NPU_initialized=False,model_initialized=False,
            torch_imported='torch' in sys.modules,source_helper=h.desc(HELPER))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='s35-qualified-counts-paired-full463-v119-1')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    start=datetime.now(timezone.utc).isoformat()
    try:result=run(args.tag);result['audit_errors']=[]
    except BaseException:result=dict(status='audit_failed',audit_errors=[traceback.format_exc()],formal_Goodput=False,NPU_initialized=False)
    result.update(schema='V119.external-independent-terminal-FAILURE-audit.v1',actual_audit_argv=sys.argv,
        actual_audit_python=sys.executable,actual_python_version=sys.version,runtime_wrapper_used=False,
        audit_started_UTC=start,audit_finished_UTC=datetime.now(timezone.utc).isoformat(),
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],output=args.output)))
    return int(result['status']=='audit_failed')

if __name__=='__main__':raise SystemExit(main())
