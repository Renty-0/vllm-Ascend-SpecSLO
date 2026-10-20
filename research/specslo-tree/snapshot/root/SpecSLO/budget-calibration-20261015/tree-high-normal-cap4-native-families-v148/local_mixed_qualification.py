"""Source-bound S19 atomic mixed-prefill local diagnostic, never Goodput."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import time
import traceback


P = Path(__file__).resolve().parent
NATIVE_COUNTERS = (
    'aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays',
    'aclgraph_failed_captures', 'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks',
    *(f'aclgraph_{role}_{field}' for role in ('generic', 'draft', 'target')
      for field in ('eager_fallback_calls', 'disabled_entry_calls', 'unclassified_calls',
                    'runtime_validation_calls', 'runtime_validation_failures')),
)


class HarnessIncomplete(RuntimeError):
    pass


def require_complete_harness():
    # Source availability is permission to run a diagnostic, never its result.
    import atomic_bounded_harness_v75 as harness
    import lifecycle_actor_v75 as lifecycle
    import mixed_target_reference_v75 as reference
    import independent_history_v75 as history
    import flexible_target_v78 as target
    required = ((harness, 'run_two_round_scope'), (lifecycle, 'run_abort_and_fallback_cases'),
        (reference, 'qualify_mixed_round'), (history, 'allocate_independent_epoch'),
        (target, 'prepare_mixed_call'))
    for module, name in required:
        if Path(module.__file__).resolve().parent != P or not callable(getattr(module, name, None)):
            raise HarnessIncomplete('V78 missing local source actor: '+name)
    from local_qualification_scope import scopes
    geometry = [scope.phases() for scope in scopes()]
    if ([len(rows) for rows in geometry] != [2, 3]
            or any(row['physical_M'] != 197 or row['physical_S'] != 19
                   for rows in geometry for row in rows)):
        raise HarnessIncomplete('V78 requires its actual five S19 progressive phases')
    return {'source_actors_present': True, 'numerical_qualification_inferred': False}


def run_atomic_bounded_qualification(engine):
    require_complete_harness()
    from startup_receipts_v75 import stage4
    from inventory_memory import checkpoint
    stage = lambda label, callback: stage4(engine, label, callback)
    def memory(label):
        return checkpoint(engine, 'V78-'+label, 0, 0)
    report = {'schema': 'V78.S19-local-atomic-qualification.v1',
        'qualification': run_prepared_actor_chain(engine), 'fixed_point': [],
        'online_qualified': False, 'cost_or_Goodput': False}
    stage('V78-qualification-progress-write', lambda: save(P/'local-progress', f'rank{engine.rank}.json', report))
    report['after_qualification_memory'] = memory('after-qualification')
    for repeat in range(2):
        before = stage('V78-zero-trace-before', lambda: native_snapshot(engine))
        trace = run_prepared_actor_chain(engine)
        after = stage('V78-zero-trace-after', lambda: native_snapshot(engine))
        def verify():
            delta = native_delta(before, after)
            if any(delta.values()):
                raise RuntimeError('V78 complete local trace discovered/validated/fell back: '+repr(delta))
            return delta
        delta = stage('V78-actual-complete-zero-trace', verify)
        report['fixed_point'].append({'repeat': repeat, 'before': before, 'after': after,
            'delta': delta, 'actor_chain': trace, 'memory': memory('zero-trace-'+str(repeat))})
        stage('V78-zero-trace-progress-write', lambda: save(P/'local-progress', f'rank{engine.rank}.json', report))
    stage('V78-original-seal', engine.seal_graph_cache)
    report['sealed'] = stage('V78-inspect-original-seal', lambda: bool(engine.graph_runner.graph_cache_sealed))
    if report['sealed'] is not True:
        raise RuntimeError('Original Native graph cache did not seal')
    report['after_seal_memory'] = memory('after-seal')
    return report


def native_snapshot(engine):
    metrics = engine.graph_metrics()
    if any(key not in metrics or type(metrics[key]) is not int or metrics[key] < 0
           for key in NATIVE_COUNTERS):
        raise RuntimeError('V78 needs every actual nonnegative integer Native counter')
    return {key: metrics[key] for key in NATIVE_COUNTERS}


def native_delta(before, after):
    if set(before) != set(NATIVE_COUNTERS) or set(after) != set(NATIVE_COUNTERS):
        raise RuntimeError('V78 Native discovery counter missing; implicit zero forbidden')
    if any(type(value) is not int or value < 0 for row in (before, after) for value in row.values()):
        raise RuntimeError('V78 Native raw snapshots require exact nonnegative integers')
    delta = {key: after[key]-before[key] for key in NATIVE_COUNTERS}
    if any(type(value) is not int or value < 0 for value in delta.values()):
        raise RuntimeError('V78 Native counter decreased')
    return delta


def run_prepared_actor_chain(engine):
    """Concrete future all4 path; unavailable without the private local permit."""
    from diagnostic_authorization import require_bounded_scope
    require_bounded_scope()
    from local_qualification_scope import scopes
    from atomic_bounded_harness_v75 import run_two_round_scope
    from lifecycle_actor_v75 import run_abort_and_fallback_cases
    from startup_receipts_v75 import stage4
    from vllm_ascend.spec_decode.pearl import native_engine as native
    stage = lambda label, callback: stage4(engine, label, callback)
    rows = [run_two_round_scope(engine, native, scope, startup_stage=stage) for scope in scopes()]
    lifecycle = run_abort_and_fallback_cases(engine, native, stage_all4=stage)
    return {'schema': 'V101.executed-local-S19-S35-actor-chain.v1',
            'scopes': rows, 'lifecycle': lifecycle,
            'online_qualified': False, 'cost_or_Goodput': False,
            'initial_fixture_adoption_is_online_reachable_proof': False}


def audit_completed_bounded_reports(reports):
    """Validate actual four-rank files; no CPU/old receipt substitutes for them."""
    require_complete_harness()
    from bounded_report_audit_v78 import audit_reports
    return audit_reports(reports)


def audit_completed_poisson_worker_reports(reports):
    """Check the raw worker envelope before auditing its actual coverage record."""
    from poisson_coverage_probe import audit_poisson_coverage_reports
    return _audit_completed_poisson_worker_payload(reports, 'poisson-coverage-probe',
        'poisson_coverage_probe', audit_poisson_coverage_reports)


def audit_completed_live_poisson_worker_reports(reports):
    from live_poisson_probe import audit_live_poisson_reports
    return _audit_completed_poisson_worker_payload(reports, 'live-poisson-probe',
        'live_poisson_probe', audit_live_poisson_reports)


def _audit_completed_poisson_worker_payload(reports, action, field, audit):
    if (type(reports) is not list or len(reports) != 4
            or any(type(row) is not dict or type(row.get('rank')) is not int for row in reports)
            or sorted(row['rank'] for row in reports) != list(range(4))):
        raise RuntimeError('Coverage needs four distinct actual integer worker ranks')
    inner = []
    for row in sorted(reports, key=lambda value: value['rank']):
        record = row.get(field)
        if (row.get('action') != action
                or any(type(row.get(key)) is not list or row[key] != []
                       for key in ('errors', 'gain', 'timing'))
                or any(row.get(key) is not False for key in
                       ('new_mixed197_qualified', 'online_qualified', 'cost_or_Goodput'))
                or type(record) is not dict or type(record.get('rank')) is not int
                or record['rank'] != row['rank']):
            raise RuntimeError('Coverage raw worker envelope differs: '+str(row['rank']))
        inner.append(record)
    return audit(inner)


def save(directory, name, data):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def worker(cfg, rank, port, connection, admission_connection=None,
           compact_worker_result_metrics=False, *, diagnostic_spec, output_directory):
    from diagnostic_authorization import activate_local_diagnostic
    engine = None
    try:
        with activate_local_diagnostic(diagnostic_spec):
            import torch
            import torch.distributed as dist
            os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank),
                              WORLD_SIZE='4', LOCAL_RANK=str(rank))
            if os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE') == 'auto':
                os.environ.pop('HCCL_NPU_SOCKET_PORT_RANGE')
            # Local P and untouched original startup qualifiers; no old source
            # directory is installed and no model exists before this gate.
            from vllm_ascend.spec_decode.pearl import native_engine as native
            from original_tree_cycle_v75 import capture_original_protocol_before_startup
            compiled_protocol = capture_original_protocol_before_startup(native)
            import tree_entry
            tree_entry.install(fast=True, host=True, transport=True, replay=True, repair=True, prefill=True)
            # Diagnostic-only captured hidden lifetimes must be installed
            # before the original draft inventory captures its real graphs.
            if diagnostic_spec['action'] not in ('live-generation-probe', 'poisson-coverage-probe',
                                                 'live-poisson-probe'):
                from vllm_ascend.spec_decode.pearl import native_graph
                import tail_hidden_observer
                tail_hidden_observer.install(native_graph)
            from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
            engine = NativePearlEngine(cfg)
            engine._v75_original_protocol = compiled_protocol
            connection.send(('ready', rank, engine.graph_metrics()))
            while True:
                mode, action, _, _ = connection.recv()
                if mode == 'exit':
                    break
                if mode != 'calibrate' or action != diagnostic_spec['action']:
                    raise RuntimeError('V78 worker rejects nonlocal or changed diagnostic action')
                from startup_receipts_v75 import audit_actual_startup, prepare_original_draft_family, stage4
                with torch.inference_mode():
                    prepare_original_draft_family(engine)
                    startup = audit_actual_startup(engine)
                    report = {'rank': rank, 'action': action, 'gain': [], 'timing': [], 'errors': [],
                              'startup': startup, 'new_mixed197_qualified': False,
                              'online_qualified': False, 'cost_or_Goodput': False}
                    if action == 'bounded-qualification':
                        report['bounded_qualification'] = run_atomic_bounded_qualification(engine)
                    elif action == 'live-generation-probe':
                        from live_generation_probe import run_live_generation_probe
                        from live_runtime_factory import make_live_runtime_hooks
                        report['live_generation_probe'] = run_live_generation_probe(engine,
                            make_hooks=make_live_runtime_hooks,
                            allowed_partitions=frozenset((1, (b,)) for b in range(1, 6)))
                    elif action == 'poisson-coverage-probe':
                        from poisson_coverage_probe import run_poisson_coverage_probe
                        report['poisson_coverage_probe'] = run_poisson_coverage_probe(engine)
                    elif action == 'live-poisson-probe':
                        from live_poisson_probe import run_live_poisson_probe, live_poisson_partitions
                        from live_runtime_factory import make_live_runtime_hooks
                        report['live_poisson_probe'] = run_live_poisson_probe(engine,
                            make_hooks=make_live_runtime_hooks,
                            allowed_partitions=live_poisson_partitions())
                    stage4(engine, 'original-startup-cache-release', engine._release_cache)
                save(Path(output_directory), f'rank{rank}-local-diagnostic.json', report)
                connection.send(('calibrated', rank, {'rows': len(report['gain']), 'timings': len(report['timing'])}))
    except BaseException:
        error = traceback.format_exc()
        try:
            save(Path(output_directory), f'rank{rank}-last-error.json',
                 {'rank': rank, 'action': diagnostic_spec['action'], 'error': error,
                  'online_qualified': False, 'cost_or_Goodput': False})
        except BaseException:
            error += '\nError snapshot failed:\n'+traceback.format_exc()
        connection.send(('error', error))
    finally:
        connection.close()
        if admission_connection is not None:
            admission_connection.close()
        if engine is not None:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('startup-only', 'bounded-qualification',
        'live-generation-probe', 'poisson-coverage-probe', 'live-poisson-probe'), required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    from check_v75 import require_local_launch
    evidence = require_local_launch(args.action)  # Before api/model import.
    diagnostic_spec = {'action': args.action, 'identity': evidence['identity']}
    output = P/'local-output'/args.tag
    output.mkdir(parents=True, exist_ok=False)
    from diagnostic_authorization import activate_local_diagnostic
    with activate_local_diagnostic(diagnostic_spec):
        from startup_config_v75 import make_config
        from vllm_ascend.spec_decode.pearl import api
        api._pearl_worker = partial(worker, diagnostic_spec=diagnostic_spec, output_directory=str(output))
        engine = None
        try:
            engine = api.PEARLEngine(make_config())
            engine._send_all(('calibrate', args.action, None, None))
            replies = engine._receive_all('V78 local '+args.action)
            if sorted(replies, key=lambda row: row[1]) != [
                    ('calibrated', rank, {'rows': 0, 'timings': 0}) for rank in range(4)]:
                raise RuntimeError('V78 exact four-rank worker finalreply differs')
            actual_audit = None
            if args.action == 'bounded-qualification':
                actual_audit = audit_completed_bounded_reports([
                    json.loads((output/f'rank{rank}-local-diagnostic.json').read_text())
                    for rank in range(4)])
            elif args.action == 'live-generation-probe':
                from live_worker_report_audit import audit_completed_live_worker_reports
                actual_audit = audit_completed_live_worker_reports([
                    json.loads((output/f'rank{rank}-local-diagnostic.json').read_text())
                    for rank in range(4)])
            elif args.action == 'poisson-coverage-probe':
                actual_audit = audit_completed_poisson_worker_reports([
                    json.loads((output/f'rank{rank}-local-diagnostic.json').read_text())
                    for rank in range(4)])
            elif args.action == 'live-poisson-probe':
                actual_audit = audit_completed_live_poisson_worker_reports([
                    json.loads((output/f'rank{rank}-local-diagnostic.json').read_text())
                    for rank in range(4)])
            save(output, 'completed.json', {'replies': replies, 'action': args.action,
                 'actual_bounded_raw_report_audit': actual_audit, 'online_qualified': False})
        finally:
            if engine is not None:
                engine.exit()
            save(output, 'worker-exit.json', {'closed': bool(engine and engine._closed),
                'all_exited': bool(engine and all(not proc.is_alive() for proc in engine._processes))})


if __name__ == '__main__':
    main()
