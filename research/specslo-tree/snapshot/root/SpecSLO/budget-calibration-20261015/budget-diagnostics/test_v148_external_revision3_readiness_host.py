"""One real-producer JSON readiness regression; no device or parent suites."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

D=Path(__file__).resolve().parent
OLD=D/'analyze_v148_actual_full_cap4_revision2.py'
NEW=D/'analyze_v148_actual_full_cap4_revision3.py'
FAILED=D/'ROOT_V148_ACTUAL_FULL_CAP4_REVISION2_AUDIT.json'
TAG='high-normal-cap4-v148-restored-paths-full463-v2'
RECEIPT_SHA='5d8c272410120e25fe311328edcfdd6202c6db516cc0f22a1736458b064342d3'
OLD_SHA='e3e1147a478a266d94d4febbe1f16bb0a88cc04ed58fa0bcc9c8f8326b5958a9'
FAILED_SHA='b29779074f937c260397af4a801d121e45dc299e0dd831089152f30df7de31c2'
spec=importlib.util.spec_from_file_location('_v148_external_revision3_readiness',NEW)
subject=importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
observations={}


class ReadinessSerialization(unittest.TestCase):
    def test_actual_frozen_producer_roundtrip_preserves_all_fields_and_rejects_changes(self):
        self.assertEqual(subject.sha(OLD),OLD_SHA)
        self.assertEqual(subject.sha(FAILED),FAILED_SHA)
        previous=subject.read(FAILED)
        self.assertEqual(previous['status'],'failed')
        self.assertIn('terminal readiness exactly equals',previous['error'])
        old_line="    need(receipt['readiness']==mods['check_v132'].require_local_launch('split192-candidate-full463'),\n"
        new_line="    need(receipt['readiness']==json.loads(json.dumps(mods['check_v132'].require_local_launch('split192-candidate-full463'),allow_nan=False)),\n"
        self.assertEqual(NEW.read_text().replace(new_line,old_line),OLD.read_text())
        tree=ast.parse(NEW.read_text())
        source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_sources')
        message='terminal readiness exactly equals current full original source/prior-component binding'
        statements=[n for n in ast.walk(source) if isinstance(n,ast.Expr)
            and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
            and n.value.func.id=='need' and len(n.value.args)==2
            and isinstance(n.value.args[1],ast.Constant) and n.value.args[1].value==message]
        self.assertEqual(len(statements),1)
        unit=ast.parse('def guard(receipt,mods):\n    pass\n')
        unit.body[0].body=[copy.deepcopy(statements[0]),ast.Return(ast.Constant(True))]
        namespace=dict(json=json,need=subject.need)
        exec(compile(ast.fix_missing_locations(unit),'V148-exact-revision3-readiness-statement','exec'),namespace)
        guard=namespace['guard']
        receipt_path=subject.P/f'local-receipt-{TAG}.json'
        self.assertEqual(subject.sha(receipt_path),RECEIPT_SHA)
        receipt=subject.read(receipt_path)
        with subject.host_modules() as (mods,helpers):
            producer=mods['check_v132'].require_local_launch('split192-candidate-full463')
            frozen=subject.read(subject.P/'PREPARED_SOURCE_HASHES_V132.json')
            paths=[subject.P/n for n in ('check_v132.py','component_gates_v132.py',
                'candidate_controls_v132.py','supervise_candidate_v132.py')]
            self.assertTrue(all(subject.sha(p)==frozen['files'][str(p)] for p in paths))
            self.assertNotEqual(receipt['readiness'],producer)
            self.assertEqual(receipt['readiness'],json.loads(json.dumps(producer,allow_nan=False)))
            original_parts=producer['component_actual_evidence']['mapper_partitions']
            stored_parts=receipt['readiness']['component_actual_evidence']['mapper_partitions']
            self.assertEqual(len(original_parts),6)
            self.assertTrue(all(type(p) is tuple for p in original_parts))
            self.assertTrue(all(type(p) is list for p in stored_parts))
            self.assertEqual(original_parts,[(16,n,'causal') for n in range(59,65)])
            fixed=copy.deepcopy(producer)
            fixed['component_actual_evidence']['mapper_partitions']=[list(p) for p in original_parts]
            self.assertEqual(receipt['readiness'],fixed)  # Only these six container types differ.

            # Reuse the captured REAL producer output below; these lambdas are
            # explicit HOST test wrappers, not substituted qualification proof.
            def wrapped(value):
                return {'check_v132':SimpleNamespace(require_local_launch=lambda action:value)}
            self.assertTrue(guard(receipt,wrapped(producer)))
            mutations=[]
            altered=copy.deepcopy(receipt);altered['readiness']['identity']['CPU_sha256']='0'*64
            mutations.append(('changed_CPU_SHA',altered))
            altered=copy.deepcopy(receipt);altered['readiness']['component_actual_evidence']['mapper_partitions'][0][1]=58
            mutations.append(('changed_partition_value',altered))
            altered=copy.deepcopy(receipt);altered['readiness']['unexpected_sentinel']=False
            mutations.append(('added_unrelated_field',altered))
            altered=copy.deepcopy(receipt);del altered['readiness']['formal_Goodput']
            mutations.append(('removed_field',altered))
            for name,altered in mutations:
                with self.subTest(name=name),self.assertRaises(ValueError):guard(altered,wrapped(producer))
            nonfinite=copy.deepcopy(producer);nonfinite['unexpected_nonfinite']=float('nan')
            with self.assertRaises(ValueError):guard(receipt,wrapped(nonfinite))
            observations.update(actual_receipt=subject.desc(receipt_path),
                retained_failed_revision2=subject.desc(FAILED),
                actual_frozen_producer_source=[subject.desc(p) for p in paths],
                sole_difference_paths=[f'component_actual_evidence.mapper_partitions[{i}]' for i in range(6)],
                sole_difference='receipt JSON list vs actual producer tuple, identical16/59..64/causal values',
                producer_JSON_roundtrip_exactly_equals_retained_readiness=True,
                counterexamples_rejected=[name for name,_ in mutations]+['nonfinite_producer'],
                original_actual_readiness_producer_called=True,full_terminal_audit_executed=False,
                exact_inverse_restores_revision2_bytes=True,source_or_numerical_gates_removed=False)
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('torch_npu',sys.modules)


if __name__=='__main__':
    started=time.perf_counter()
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ReadinessSerialization))
    stamp=time.time_ns()
    log=D/f'V148_EXTERNAL_AUDITOR_REVISION3_READINESS_HOST_{stamp}.log'
    with log.open('x') as output:output.write(stream.getvalue())
    proof=dict(schema='HOST.V148.external-revision3-real-readiness-JSON-regression.v1',
        status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-started,
        source=subject.desc(NEW),test_source=subject.desc(__file__),log=subject.desc(log),
        original_revision2=subject.desc(OLD),observations=observations,
        actual_HOST_argv=sys.argv,Torch_imported='torch' in sys.modules,
        NPU_queried=False,model_or_constructor_called=False,parent_suites_rerun=False,
        frozen_candidate_modified=False,old_auditor_or_failed_record_modified=False)
    output=D/f'V148_EXTERNAL_AUDITOR_REVISION3_READINESS_HOST_{stamp}.json'
    with output.open('x') as handle:json.dump(proof,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status=proof['status'],tests_run=proof['tests_run'],
        output=subject.desc(output),source=subject.desc(NEW))))
    if not result.wasSuccessful():raise SystemExit(1)
