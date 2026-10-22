"""Static SOURCE review only. Never import an auditor or traverse actual/receipt."""
from pathlib import Path
import ast, copy, datetime, hashlib, json, re, sys

assert sys.version_info[:2] == (3, 11)
B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-normal-b4-final-kv5-single-writer-full-v204'
OLD = D / 'audit_v201_terminal_HOST_revision2.py'
NEW = D / 'audit_v204_terminal_HOST.py'
PROOF = D / 'V204_TERMINAL_HOST_AUDITOR_STATIC_SOURCE_INVERSE.json'
READY = D / 'V204_TERMINAL_HOST_AUDITOR_SOURCE_READY.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def desc(p): return dict(path=str(p), sha256=sha(p), size=Path(p).stat().st_size)
def dump(p, v):
    with p.open('x') as f:
        json.dump(v, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
assert sha(OLD) == '1c3eb7aafb01e346c43d9d76e750e7448ce910a69a52ef3595a479ff7da56729'
assert sha(NEW) == '6ed13db379ae1cd1e49c9712418b60023b49c608881915f34478ce9bcb767cd9'
old, new = OLD.read_text(), NEW.read_text()
opaque = {}
def protect(m):
    key = '__OPAQUE_SHA_' + str(len(opaque)) + '__'
    opaque[key] = m[0]
    return key
reference = re.sub(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', protect, old).replace('201', '204')
for key, value in opaque.items(): reference = reference.replace(key, value)
reference = reference.replace('Original204/44/114', 'Original201/44/114')
def defs(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
def same(a, b): return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
x, y = defs(reference), defs(new)
assert x.keys() == y.keys()
equal = [n for n in x if same(x[n], y[n])]
changed = [n for n in x if not same(x[n], y[n])]
assert changed == ['provenance', 'source_import_origins']
assert len(equal) == 13
origin = copy.deepcopy(y['source_import_origins'])
loops = [n for n in ast.walk(origin) if isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple)]
assert len(loops) == 1
added = loops[0].iter.elts.pop(0)
assert ast.unparse(added) == "SOURCE_PARENT / 'FROZEN_SOURCE201.json'"
assert same(origin, x['source_import_origins'])
assert "json.loads(json.dumps(parent,sort_keys=True,allow_nan=False))==readiness['parent_ready']" in new
assert "if sys.version_info[:2]!=(3,11):raise RuntimeError(" in new
assert 'SOURCE_PARENT_FREEZE_SHA=' in new and 'BASELINE_AUDITOR_SHA=' in new
compile(new, str(NEW), 'exec')  # No exec/import of the compiled source.

freeze = P / 'FROZEN_SOURCE204.json'
cpu = P / 'CPU_RESULT204.json'
prepared = P / 'PREPARED_HOST204.json'
root = D / 'ROOT_V204_PREPARED_SOURCE_HOST_AUDIT.json'
source_ready = D / 'V204_SINGLE_WRITER_FINAL_KV5_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer = D / 'V204_INDEPENDENT_SOURCE_PEER_REVIEW.json'
expected = {
    freeze: 'e3851927bdaf04596df9cdce579eb7a20b41e2a132413b55768a5820471fd564',
    cpu: 'e4f53a661bc2ed8bdd0b54e8891549a2183a64c855e70861982b4363f75fe7ef',
    prepared: '6bc5cf6a708bcf5f75ec4bb476c524b872d13fcc1c3995f01b2389b3ddb0fe49',
    root: '7d5a6109e622550927aa5ecd4fa3b7914b03fd0579a475e7a5ae57297caa2b02',
    source_ready: '7661a8fcc3ca6e164464655f80d32f76c7087db32b0fa90bf1fabb6819e084a4',
    peer: '8c4c38efd76e6006a843ae2a48cc08821508661014e9b168f6fbfca6ed68ecef',
}
assert all(sha(p) == h for p, h in expected.items())
f, c, ph = read(freeze), read(cpu), read(prepared)
rr = read(f['root_review']['path'])
assert sha(f['root_review']['path']) == f['root_review']['sha256']
assert f['schema'] == 'V204.ROOT-source-freeze.v1' and f['ROOT_source_reviewed'] is True
assert {k: len(f[k]) for k in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews')} == dict(files=24,external_files=251,retained_HOST_evidence=32,execution_guard_inputs=4,peer_reviews=1)
assert len(rr['reviewed_inputs']) == 313
assert c['schema'] == 'V204.full-source-HOST.v1' and c['status'] == 'passed' and c['tests_run'] == 6 and c['errors'] == c['failures'] == 0
assert c['local_before'] == c['local_after'] == ph['files'] == f['files']
assert c['external_before'] == c['external_after'] == ph['external_files'] == f['external_files']
assert ph['current_actual_tests_run'] == 6 and len(ph['actual_HOST_attempts']) == 1 and len(ph['retained_HOST_files']) == 30
writer = c['details']['single_writer_directory']
assert all(writer[k] is True for k in ('actual_actor_lambda','actual_original_stage4','real_four_process_shared_FS','HOST_collective_and_sync_carriers'))
assert [(r['case'],r['all4_passed'],r['local_FileExists_errors']) for r in writer['cases']] == [('old-race',False,3),('new-single-writer',True,0),('new-preexisting-refused',False,1)]
assert len(read(P/'PROJECTIONS204.json')['projection']) == 21
assert read(P/'INPUTS204.json')['source_parent201']['freeze_sha256'] == '035cf69b61bcf6871baa472d5f9eb12001454e2ba076de07baf025d703ee460b'
helper_names = ('HELPERS','REUSE','CLOCKS','STAT_SOURCE')
constants = {}
for n in ast.parse(new).body:
    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
        name = n.targets[0].id
        if name in ('HELPER_SHA','REUSE_SHA','CLOCK_SHA','STAT_SHA'):
            constants[name] = ast.literal_eval(n.value)
helpers = {
    D/'audit_v173_terminal_full463_HOST.py': constants['HELPER_SHA'],
    D/'audit_v197_terminal_HOST.py': constants['REUSE_SHA'],
    D/'analyze_completed_tree_full463_cost.py': constants['CLOCK_SHA'],
    D/'audit_v189_terminal_HOST.py': constants['STAT_SHA'],
}
assert all(sha(p) == h for p, h in helpers.items())
producer_files = [P/(n+'.py') for n in ('base204','check204','guard204','audit204','actor204','component204','runtime204','trace204','metadata204') if (P/(n+'.py')).is_file()]
# guard is intentionally in D, rather than guessed to be a candidate module.
producer_files.append(D/'v204-root-execution/guard204.py')
for path in producer_files: ast.parse(path.read_text())
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
proof = dict(schema='V204.future-terminal-auditor-static-SOURCE-inverse.v1',UTC=now,status='passed_static_source_review',
    baseline=desc(OLD),script=desc(NEW),namespace_rule='201->204 labels/module identity with 64hex protected; Original201/44/114 count retained',
    exact_namespace_AST_functions=equal,changed_functions=changed,
    source_import_origins_only_delta="prepend SOURCE_PARENT/'FROZEN_SOURCE201.json' to checked source manifests; exact inverse verified",
    provenance_delta='Own24/251/32/4/1/315, six HOST/current single-writer cases, exact531 immutable source-donor archive and21whole projections; no device proof inherited',
    JSON_canonical_parent_ready_tuple_fix_retained=True,compile_only=True,auditor_imported=False,auditor_executed=False,
    actual_or_receipt_read=False,Torch_imported=False,NPU_used=False,old_suites_rerun=False,
    current_source_only_CPU_schema_checked=True,producer_source_files={str(p):sha(p) for p in producer_files},
    fixed_source_pins={str(p):h for p,h in expected.items()},pure_helper_source_pins={str(p):h for p,h in helpers.items()},
    archive_bytes_not_reverified_here='ROOT already independently sealed315; future runtime provenance performs own member check and donor531 check',
    archive=dict(path=str(P/'PREPARED_SOURCE_ARCHIVE204.tar.gz'),sha256='8a229b36ebe075ac5e667708ec44d1da1462d30ce90b014702df25e33ffee604',members=315))
dump(PROOF, proof)
ready = dict(schema='V204.future-terminal-HOST-auditor-SOURCE-ready.v1',UTC=now,status='SOURCE_ONLY_READY_NOT_EXECUTED',candidate=str(P),script=desc(NEW),
    static_review=desc(PROOF),static_review_script=desc(Path(__file__)),baseline=desc(OLD),
    fixed_source_pins={str(p):h for p,h in expected.items()},pure_helper_source_pins={str(p):h for p,h in helpers.items()},
    preserved_pre_pin_source=desc(D/'V204_AUDITOR_NAMESPACE_REFERENCE_SOURCE.txt'),
    preserved_pre_pin_source_description='Initial generated204 source before ROOT freeze pins/interpreter and exact201baseline gates; not claimed to be a pure namespace-only reference',
    preparation_script=desc(D/'_prepare_audit_v204_source.py'),
    retained_static_review_failure=[desc(D/'V204_AUDITOR_STATIC_REVIEW_ATTEMPT1_SOURCE.py'),desc(D/'V204_AUDITOR_STATIC_REVIEW_ATTEMPT1_RESULT.json')],
    future_execution_requires=['ROOT explicit terminal authorization','exact tag','exact terminal receipt SHA256','fresh private D output','original Python3.11'],
    future_command='PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/vllm-hust-dev/bin/python3.11 -B '+str(NEW)+' --terminal-authorized --tag ROOT_EXACT_TAG --expected-receipt-sha256 ROOT_EXACT_TERMINAL_SHA --output '+str(D/'ROOT_CHOSEN_FRESH_V204_AUDIT.json'),
    preserved_math='13 functions namespace-AST-identical including integerBONUS, all5KV/component/usage checks, terminal cleanup, failure evidence and complete metrics/TTFT main; four exact pure source helpers retained',
    rechecked_control_evidence='current six HOST only; previous201 five HOST and earlier218-byte preservation/four HOST remain separate historical evidence',
    limitations=['No terminal or running actual/receipt read, auditor import/execution or new HOST suite performed.',
      'HOST can recompute integer verdict/counter/query/clock arithmetic and audit retained source-gated tensor matches; it cannot reexecute hidden/logit/KV/finite device mathematics.',
      'Failed/incomplete job remains failed/incomplete; no fabricated GP0, full qualification, adoption, formal GP or whole-workload AR equivalence.',
      'Original ROOT first Py3.12 prepare failure is retained before freeze; future auditor explicitly requires originalPy3.11 and source-only review does not relabel it as component failure.'],
    Torch_imported=False,NPU_used=False,model_constructed=False,actual_or_receipt_read=False,auditor_imported=False,auditor_executed=False,old_suites_rerun=False,freeze_created=False,formal_Goodput=False,whole_workload_AR_qualified=False,adopted=False)
dump(READY, ready)
print(json.dumps(dict(script=desc(NEW),proof=desc(PROOF),ready=desc(READY)),indent=2))
