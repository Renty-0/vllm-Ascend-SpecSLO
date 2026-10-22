"""Actual prototype writer/auditor on explicit HOST carriers; no model math."""
import ast
import copy
import datetime
import gzip
import hashlib
import importlib.abc
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = D/'v210-source-design-only'
DONOR = B/'tree-normal-cap3-original-geometries-full-v207'
FORBIDDEN = ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'acl', 'transformers', 'safetensors')


class Firewall(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('HOST prototype forbids backend import: '+fullname)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.meta_path.insert(0, Firewall())
    sys.path[:0] = [str(P), str(DONOR), str(B/'tree-high-normal-cap4-native-families-v148')]
    folder = P/'HOST-attempts'/str(time.time_ns())
    folder.mkdir(parents=True)
    watched = [p for p in P.glob('*.py')] + [DONOR/'audit_host207.py', Path(__file__)]
    before = {str(p): sha(p) for p in watched}
    for p in watched:
        (folder/p.name).write_bytes(p.read_bytes())
    result = dict(schema='ROOT.V210.prototype-writer-auditor-HOST.v1',
                  UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='failed', NPU_used=False, model_constructed=False,
                  tensor_math_qualified=False, production_issuer=False, errors=[])
    started = time.monotonic()
    try:
        from scope_v146 import NATIVE_COUNTERS
        source = (DONOR/'audit_host207.py').read_text()
        replacements = [
            ('from capability207 import GREEDY,HIDDEN', 'from geometry210 import GREEDY,HIDDEN'),
            ('from capability207 import HIDDEN', 'from geometry210 import GREEDY,HIDDEN'),
            ('from audit3207 import audit', 'from audit3210 import audit'),
            ("original_statement('actor207.py'", "original_statement('actor210.py'"),
            ("schema='V207.original96-B3-current-component.v1'", "schema='V210.bounded64-B3-current-component.v1'"),
            ('heads=[11,12,13,14]*16+[0]*32', 'heads=[11,12,13,14]*16'),
            ('cold=family==\'hidden\'and pass_index==0 and cycle==0 and index==0',
             'cold=pass_index==0 and cycle==0 and index==0'),
            ('runtime_validated_before=not cold,runtime_validated_after=not cold',
             "runtime_validated_before=not cold,runtime_validated_after=(not cold or family=='greedy')"),
            ('inputs=[0]*96,positions=[0]*96,slots=list(range(96))',
             'inputs=[0]*64,positions=[0]*64,slots=list(range(64))'),
            ('actual_seq_lengths_q=list(range(4,65,4))+list(range(66,97,2))',
             'actual_seq_lengths_q=list(range(4,65,4))'),
            ('production_greedy_M=96', 'production_greedy_M=64'),
            ('separate_hidden_counterpart_M=96', 'separate_hidden_counterpart_M=64'),
            ('same_M=96,same_S=32', 'same_M=64,same_S=16'),
            ('owners=list(range(16))+list(range(463,479))', 'owners=list(range(16))'),
            ('owners=list(range(16,32))+list(range(479,495))', 'owners=list(range(16,32))'),
            ('production_new_keys=0,diagnostic_new_hidden_keys=1,',
             'production_new_keys=0,diagnostic_new_hidden_keys=1,diagnostic_new_greedy_keys=1,'),
            ('diagnostic_new_keys=[]if rank==0 else[repr(HIDDEN)]',
             'diagnostic_new_keys=[]if rank==0 else sorted(map(repr,(GREEDY,HIDDEN)))'),
            ('case=dict(M=96,S=32,Q=64,dummy_lengths=[2]*16)',
             'case=dict(M=64,S=16,Q=64,dummy_lengths=[])'),
        ]
        for old, new in replacements:
            if old == "original_statement('actor207.py'":
                require(source.count(old) == 2, 'two exact original writer statement selectors')
            else:
                require(source.count(old) == 1, 'unique HOST schema adaptation '+old)
            source = source.replace(old, new)
        names = {'save', 'original_statement', 'evaluate', 'zero', 'semantic', 'committed',
                 'draft_proof', 'target_proof', 'run_normal'}
        nodes = [copy.deepcopy(n) for n in ast.parse(source).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        require({n.name for n in nodes} == names, 'complete actual role-carrier functions')
        namespace = dict(P=P, require=require, Path=Path, gzip=gzip, json=json,
                         ast=ast, copy=copy, NS=NS, NATIVE_COUNTERS=NATIVE_COUNTERS)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(DONOR/'audit_host207.py')+'::bounded64-HOST-only', 'exec'), namespace)
        raw = folder/'raw';raw.mkdir()
        result['writer_and_audit'] = namespace['run_normal'](raw)
        result['explicit_carrier_raw_files'] = len(list(raw.glob('*.json.gz')))
        require(result['explicit_carrier_raw_files'] == 64, 'all four role files retained')
        from audit3210 import audit,load
        from geometry210 import diagnostic_geometry
        # The donor's final negative test intentionally leaves one target None.
        # Retain that exact carrier, then restore its identical synthetic target
        # replica so each further mutation is tested against a passing baseline.
        broken=raw/'rank1-pass0-cycle0.json.gz'
        (folder/'negative-target-None.json.gz').write_bytes(broken.read_bytes())
        broken.write_bytes((raw/'rank2-pass0-cycle0.json.gz').read_bytes())
        audit(raw)
        mutations=[
            ('old96-case', 'rank2-pass0-cycle0.json.gz', lambda v:v['case'].update(M=96)),
            ('old96-extra-heads', 'rank2-pass0-cycle0.json.gz', lambda v:v['target']['greedy'][2]['full_physical_tokens'].extend([0]*32)),
            ('old96-dummy-owner', 'rank2-pass0-cycle0.json.gz', lambda v:v['target']['metadata']['owners'].append(463)),
            ('forged-greedy-cold-before', 'rank2-pass0-cycle0.json.gz', lambda v:v['target']['greedy'][0].update(runtime_validated_before=True)),
            ('target-has-draft-proof', 'rank2-pass0-cycle0.json.gz', lambda v:v.update(draft={})),
            ('draft-missing-proof', 'rank0-pass0-cycle0.json.gz', lambda v:v.update(draft=None)),
            ('target-next-forged-proof', 'rank2-pass2-next.json.gz', lambda v:v.update(draft_proof={})),
            ('nonzero-zero-trace', 'rank2-component.json.gz', lambda v:v['traces'][2]['Native']['delta'].update(aclgraph_captures=1)),
        ]
        rejected=[]
        for label,name,mutate in mutations:
            path=raw/name;original=path.read_bytes();value=load(path);mutate(value)
            namespace['save'](raw,name,value)
            try:
                try:audit(raw)
                except (RuntimeError,ValueError,TypeError,KeyError) as error:
                    rejected.append(dict(case=label,rejected=True,error_type=type(error).__name__))
                else:raise AssertionError('mutation incorrectly accepted: '+label)
            finally:path.write_bytes(original)
        result['schema_negatives']=rejected
        # Exercise the actual scoped mapper, including exceptions/finally.
        parent=object();sentinel=object()
        def old(n,q,f):return sentinel
        engine=NS(cache_allocation=None,cache_block_tables=None,graph_runner=NS(graph_cache_sealed=False))
        control=NS(engine=engine,mode='candidate',mapper=old,layout=NS(envelope=old))
        current=lambda e:parent
        with diagnostic_geometry(engine,control,current_parent=current) as dispatch:
            require(dispatch(2,10,'causal')is sentinel,'out-of-domain original return identity')
            try:dispatch(16,64,'causal')
            except RuntimeError:pass
            else:raise AssertionError('missing current allocation accepted')
            engine.cache_allocation=object()
            require(dispatch(16,64,'causal')==(64,16,[]),'bounded exact real-only geometry')
            engine.graph_runner.graph_cache_sealed=True
            try:dispatch(16,64,'causal')
            except RuntimeError:pass
            else:raise AssertionError('sealed graph accepted bounded extension')
            engine.graph_runner.graph_cache_sealed=False
        require(control.mapper is control.layout.envelope is old
                and not hasattr(engine,'_v210_bounded_normal64_geometry'),'original owner restored')
        engine.cache_allocation=None
        try:
            with diagnostic_geometry(engine,control,current_parent=current):raise ValueError('original-model-error-carrier')
        except ValueError as error:require(str(error)=='original-model-error-carrier','original exception propagated')
        require(control.mapper is control.layout.envelope is old
                and not hasattr(engine,'_v210_bounded_normal64_geometry'),'finally restores owner on model error')
        result['actual_bounded_mapper_HOST']=dict(exact_domain=True,fallback_identity=True,
             missing_allocation_rejected=True,sealed_rejected=True,finally_owner_restored=True,
             original_exception_preserved=True,production_issuer=False)
        require(before == {p: sha(p) for p in before}, 'no source changes during HOST')
        require(not any(n.split('.')[0] in FORBIDDEN for n in sys.modules), 'no backend imported')
        result.update(status='passed_explicit_HOST_schema_only', source_before=before, source_after=before,
                      source_schema_adaptations=replacements)
    except BaseException:
        result['errors'].append(traceback.format_exc())
    result['elapsed_seconds'] = time.monotonic()-started
    result['Torch_library_imported'] = 'torch' in sys.modules
    output = folder/'RESULT.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(output), sha256=sha(output), status=result['status'], errors=result['errors'])))
    return 0 if result['status'].startswith('passed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
