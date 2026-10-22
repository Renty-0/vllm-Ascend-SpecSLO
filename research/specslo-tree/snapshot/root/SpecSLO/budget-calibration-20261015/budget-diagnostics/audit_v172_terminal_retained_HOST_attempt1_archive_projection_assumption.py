"""Read only terminal P172 evidence. No backend import, model or device call."""
import argparse, ast, copy, gzip, hashlib, importlib.abc, json, math, statistics
import sys, tarfile, traceback, types
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-live-large-mixed-mask-int8-full-v172'
A = P/'local-output/live645-int8-v172-full463-v2'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def desc(p):
    p = Path(p)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

def read(p):
    p = Path(p)
    data = p.read_bytes()
    if p.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

class Firewall(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'safetensors', 'acl'):
            self.attempts.append(fullname)
            raise ImportError('Independent terminal HOST firewall: '+fullname)

def extracted(path, name, namespace):
    node = copy.deepcopy(next(n for n in ast.walk(ast.parse(Path(path).read_text()))
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path)+'::independent-HOST-only', 'exec'), namespace)
    return namespace[name]

class Value:
    def __init__(self, values): self.values = values
    @property
    def ndim(self): return 1
    def numel(self): return len(self.values)
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return copy.deepcopy(self.values)
    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            return Value([row[cols] for row in self.values[rows]])
        return Value(self.values[key])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(args.output.resolve().is_relative_to(D) and not args.output.exists(), 'Fresh private audit output')
    receipt_path = P/'receipt-live645-int8-v172-full463-v2.json'
    need(sha(receipt_path) == args.expected_receipt_sha256, 'Exact terminal receipt')
    receipt = read(receipt_path)
    result = dict(schema='V172.independent-terminal-component-and-retained-full463-audit.v1',
                  status='incomplete', job_status=receipt['status'], receipt=desc(receipt_path),
                  audit_script=desc(__file__), errors=[], frozen_sources_modified=False,
                  NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False)
    try:
        need(receipt['status'] == 'failed' and receipt['child']['exit_code'] == 1
             and receipt['sources_unchanged'] is True and receipt['owned_remaining'] == [], 'Preserve terminal failed job and clean source/ownership')
        need(receipt['elapsed_seconds'] < receipt['limits']['child_execution_seconds'], 'Not a deadline expiry')
        need(read(A/'worker-exit.json') == dict(closed=True, all_exited=True), 'Actual worker pool closed')
        for name in ('pre_verifier', 'post_verifier'):
            row = receipt[name]; proof = row['actual_result']
            need(row['exit_code'] == 0 and sha(row['path']) == row['sha256']
                 and proof['status'] == 'pass' and proof['errors'] == []
                 and (proof['checked_frozen_files'], proof['checked_external_files'], proof['checked_packages']) == (10496, 15, 212), name)
        need(receipt['devices_before']['all_idle'] is receipt['devices_after']['all_idle'] is True, 'Retained guardian idle checks')
        log = Path(receipt['child']['log']).read_text()
        need("KeyError: 'used_aclgraph'" in log and 'audit172.py' in log
             and "execution['used_aclgraph']" in log, 'Exact post-worker traceback')
        result['terminal'] = dict(child_exit_code=1, supervisor_elapsed_seconds=receipt['elapsed_seconds'],
            deadline_expired=False, original_baseline_prepost_pass=True, owned_remaining=[],
            retained_four_idle=True, worker_pool_closed=True, log=desc(receipt['child']['log']),
            completed_json_present=(A/'completed.json').exists(),
            classification='post-worker HOST audit serialization-schema KeyError; not a model exception')

        frozen = read(P/'FROZEN_SOURCE172.json')
        need(sha(P/'FROZEN_SOURCE172.json') == receipt['readiness']['freeze_sha256'], 'Exact prepared freeze')
        files = {}
        maps = [frozen[k] for k in ('files', 'external_files', 'retained_HOST_evidence', 'execution_guard_inputs', 'peer_reviews')]
        maps += [{str(P/'CPU_RESULT172.json'): frozen['CPU_sha256'], str(P/'FROZEN_SOURCE172.json'): sha(P/'FROZEN_SOURCE172.json'),
                  frozen['root_review']['path']: frozen['root_review']['sha256']}]
        for mapping in maps:
            for path, wanted in mapping.items():
                need(path not in files or files[path] == wanted, 'No conflicting source pin')
                files[path] = wanted
        need(all(sha(path) == wanted for path, wanted in files.items()), 'All current frozen selected source bytes')
        archive = P/'PREPARED_SOURCE_ARCHIVE172.tar.gz'
        need(sha(archive) == receipt['readiness']['archive']['sha256'], 'Current source archive SHA')
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            need(len(members) == len(files) == len({m.name for m in members}), 'Exact archive membership')
            for member in members:
                need(member.isfile() and '/'+member.name in files and hashlib.sha256(tar.extractfile(member).read()).hexdigest() == files['/'+member.name], 'Regular archive member exact bytes')
        result['source_provenance'] = dict(freeze=desc(P/'FROZEN_SOURCE172.json'), archive=desc(archive),
            source_members=len(files), all_selected_bytes_current=True, all_regular_archive_members_exact=True,
            old_dependency_archives_not_reexpanded_by_this_audit=True)

        raw_files = sorted(p for p in A.rglob('*') if p.is_file())
        manifest = {str(p): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in raw_files}
        manifest_path = args.output.with_name(args.output.stem+'_RAW_MANIFEST.json')
        need(not manifest_path.exists(), 'Fresh raw manifest')
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        result['raw_manifest'] = dict(**desc(manifest_path), files=len(manifest), bytes_total=sum(v['bytes'] for v in manifest.values()))

        firewall = Firewall(); sys.meta_path.insert(0, firewall); sys.path.insert(0, str(P))
        import audit172
        reports = [read(A/f'rank{i}-worker.json') for i in range(4)]
        need(all(r['errors'] == [] and r['startup_completed'] is r['sealed'] is r['full463_completed'] is r['cache_released'] is True for r in reports), 'All actual workers complete')
        try:
            audit172.audit_component(A/'large645-component')
        except KeyError as exc:
            need(exc.args == ('used_aclgraph',), 'Only expected original failure')
            result['original_frozen_auditor'] = dict(status='failed_as_retained', error="KeyError: 'used_aclgraph'", source=desc(P/'audit172.py'))
        else:
            raise ValueError('Expected frozen audit failure did not reproduce')

        # asdict serializes actual dataclass fields; used_aclgraph is a property.
        graph_tree = ast.parse((NATIVE/'native_graph.py').read_text())
        cls = next(n for n in graph_tree.body if isinstance(n, ast.ClassDef) and n.name == 'NativeGraphExecution')
        field_names = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
        getter = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'used_aclgraph'))
        getter.decorator_list = []; property_ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[getter], type_ignores=[])), str(NATIVE/'native_graph.py')+'::original-property-only', 'exec'), property_ns)
        property_counts = collections_counter = {'derived_missing_property': 0}
        def execution_property(row):
            need(set(row) == field_names and type(row['mode']) is str
                 and (row['fallback_reason'] is None or type(row['fallback_reason']) is str)
                 and type(row['capture_attempted']) is type(row['replay_executed']) is bool, 'Exact original serialized dataclass schema')
            need('used_aclgraph' not in row, 'Property omission explicitly distinguished from False')
            property_counts['derived_missing_property'] += 1
            return property_ns['used_aclgraph'](types.SimpleNamespace(**row))
        tree = ast.parse((P/'audit172.py').read_text()); nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)]
        changed = []
        for fn in nodes:
            if fn.name != 'audit_component': continue
            class Adapt(ast.NodeTransformer):
                def visit_Subscript(self, n):
                    if ast.unparse(n) == "execution['used_aclgraph']":
                        changed.append(1)
                        return ast.Call(func=ast.Name(id='_execution_property', ctx=ast.Load()), args=[ast.Name(id='execution', ctx=ast.Load())], keywords=[])
                    return self.generic_visit(n)
            Adapt().visit(fn)
        need(len(changed) == 1, 'Single external schema interpretation')
        namespace = dict(vars(audit172), _execution_property=execution_property)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(P/'audit172.py')+'::independent-property-schema-only', 'exec'), namespace)
        strict = namespace['audit_worker_reports'](reports, A)
        need(strict['status'] == 'passed', 'All original remaining HOST gates')
        result['strict_retained_audit'] = strict
        result['schema_interpretation'] = dict(original_dataclass=desc(NATIVE/'native_graph.py'),
            fields=sorted(field_names), missing_property_count=property_counts['derived_missing_property'],
            interpretation='Execute exact original @property on the four retained dataclass fields; never default missing to False or insert values into raw',
            changed_external_predicates=1, all_other_original_auditor_gates_executed=True)

        # Re-execute the actual original topology packing and host tree verifier.
        tns = {}; topology = extracted(NATIVE/'tree.py', '_spine_first_topology', tns)
        parents = topology(2, 4)[0]
        pnode = copy.deepcopy(next(n for n in ast.parse((NATIVE/'tree.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'pack_selected_tree_plan'))
        pnode.decorator_list = []; stop = next(i for i,n in enumerate(pnode.body) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'device' for t in n.targets))
        pnode.body = pnode.body[:stop] + [ast.Return(value=ast.Name(id='new_parents',ctx=ast.Load()))]
        host_torch = types.SimpleNamespace(Tensor=Value, tensor=lambda x, **kw: Value(x), long='long')
        pns = dict(torch=host_torch)
        unit = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),pnode],type_ignores=[])
        exec(compile(ast.fix_missing_locations(unit),str(NATIVE/'tree.py')+'::original-packing-prefix-only','exec'),pns)
        vns = dict(torch=host_torch, TreeVerificationOutput=lambda a,b: types.SimpleNamespace(token_ids=a,accepted_node_indices=b))
        verifier = extracted(NATIVE/'native_engine.py', '_verify_tree_outputs_host', vns)
        components = [read(A/f'large645-component/rank{i}-component.json.gz') for i in range(4)]
        checkpoint_count = verdict_count = owner_verdicts = zero_subtractions = 0
        sequences = []
        for rank, report in enumerate(components):
            bounds = report['boundaries']; seq = [(x['sequence'],x['label']) for x in bounds]
            need([x[0] for x in seq] == list(range(len(seq))), 'Complete component stage sequence')
            need(all(len(e['ranks']) == 4 and sorted(x['rank'] for x in e['ranks']) == [0,1,2,3]
                     and all(x['sequence'] == e['sequence'] and x['label'] == e['label'] and x['error'] is None for x in e['ranks']) for e in bounds), 'All component TP-role shared stages')
            sequences.append(seq); fs = [x['fingerprint'] for x in report['traces']]
            need(fs[:17] == fs[17:34] == fs[34:51], 'Two complete fresh-history fingerprint repeats')
            inventory = report['inventory_delta']; wanted = set(inventory['new_keys'])
            for family in ('entries','draft_entries','target_entries'):
                need(set(inventory['before'][family]) <= set(inventory['after'][family])
                     and set(inventory['after'][family])-set(inventory['before'][family]) == (wanted if family == 'target_entries' else set()), 'Exact actual family inventory differences')
            for row in report['traces']:
                if row['pass_index']:
                    zero_subtractions += len(row['Native']['delta'])
                need(all(row['duplicate_publication_guard'][k] is True for k in ('original_terminal_observed','repeat_claim_rejected','repeated_runtime_publish_noop','actual_delivery_counter_unchanged')), 'Actual once-publication negatives')
                need(row['postpublication_normal_append_is_not_reserved_lease_growth'] is True, 'Lease closes before normal append')
                nxt = row['published_prompt_next_draft']
                if rank == 0:
                    need(nxt['draft_model_executed'] is True and nxt['all28_query_KV_original_matcher'] is True
                         and nxt['exact_materialized_ledger'] is True and nxt['candidate_KV_used_to_initialize_reference'] is False, 'Actual published-prompt nextdraft own oracle')
                for cycle, record in enumerate(row['records']):
                    path = A/'large645-component'/f"rank{rank}-pass{row['pass_index']}-{row['scope']}-cycle{cycle}.json.gz"
                    need(read(path) == record, 'Actual raw cycle checkpoint equals completed report transcript')
                    checkpoint_count += 1
                    if not rank: continue
                    target = record['target']; selected = record['selected']
                    plans = [types.SimpleNamespace(candidate_budget=len(ids), parent_indices=Value(pns['pack_selected_tree_plan'](types.SimpleNamespace(parent_indices=Value(parents), candidate_budget=8),ids))) for ids in selected]
                    verdict = verifier(record['rows'], plans, Value(target['normal_query_tokens']), 4, 'explicit-HOST-carrier')
                    emitted = [[v for v in row if v >= 0] for row in verdict.token_ids.values]
                    commit = record['compiled_commit']
                    need(emitted == commit['committed_tokens'] and verdict.accepted_node_indices.values == commit['accepted_paths'], 'Original packed-ancestry verdict recomputed from actual selected target heads')
                    need(commit['actual_logical_tokens'] == [old['token_ids']+tokens for old,tokens in zip(record['source']['contexts'],emitted,strict=True)], 'Own actual logical commit histories')
                    need(target['final_prompt_tokens'] == ([record['publication']['first_token']] if record['publication']['published'] else []), 'Actual final prompt head publishes once')
                    verdict_count += 1; owner_verdicts += len(plans)
        need(all(s == sequences[0] for s in sequences), 'Exact all-rank component stage labels')
        need(checkpoint_count == 444 and zero_subtractions == 4*17*2*21, 'Complete component raw/zero domain')
        result['additional_independent_component_raw_checks'] = dict(cycle_checkpoints_exact=checkpoint_count,
            actual_original_verdict_reexecutions=verdict_count, actual_owner_verdicts_reexecuted=owner_verdicts,
            all_stage_sequences_same=True, stage_count_per_rank=len(sequences[0]), complete_component21zero_subtractions=zero_subtractions,
            once_publication_and_nextdraft_markers_rechecked=True, hidden_or_KV_tensors_recomputed=False)

        full = read(A/'candidate/rank1-full463-actual-trace.json.gz')
        clock_source = D/'analyze_completed_tree_full463_cost.py'
        cns = dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'))
        extracted(clock_source,'need',cns); metrics = extracted(clock_source,'clocks',cns)(full)
        metrics['TTFT172'] = namespace['fairness'](full)['TTFT']
        result['full463_metrics'] = metrics
        result['per_request_metrics'] = [dict(request_index=i, request_id=r['request_id'], slo_class=r['slo_class'],
            output_tokens=len(r['completion_token_ids']), paper_TPOT_ms=r['paper_tpot_ms'], observed_TPOT_ms=r['observed_tpot_ms'],
            SLO_ms=r['slo_tpot_ms'], paper_attained=r['paper_slo_attained']) for i,r in enumerate(full['generate_results'])]
        need(not firewall.attempts and not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules), 'No backend import')
        result['HOST_imports'] = dict(forbidden_attempts=firewall.attempts, backend_imported=False)
        result['status'] = 'passed_retained_component_and_full_trace_audit_job_failed'
        result['limitations'] = ['Frozen audit remains failed; only external property-schema interpretation allowed its remaining original gates to execute.',
            'Retained original matcher/finite markers establish actual source gates, not tensor mathematics recomputed on HOST.',
            'All463 stream/source/budget/native/publication arithmetic passed; whole-workload independent AR equality and MC2 at645 remain unqualified.',
            'Metrics are descriptive retained-run results. Jobfailed is preserved; not a formal winner or adoption.']
    except BaseException:
        result['errors'].append(traceback.format_exc())
        result['status'] = 'failed_external_audit_or_incomplete_evidence'
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),status=result['status'],errors=result['errors'])))
    return int(bool(result['errors']))

if __name__ == '__main__':
    raise SystemExit(main())
