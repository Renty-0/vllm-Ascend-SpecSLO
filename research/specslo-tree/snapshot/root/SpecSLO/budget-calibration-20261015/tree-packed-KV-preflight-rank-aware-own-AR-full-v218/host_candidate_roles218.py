"""Regression against actual failed216 role traces; no Torch or NPU imports."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import candidate_comparison218 as subject

P = Path(__file__).resolve().parent
B = P.parent
ACTUAL = B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/cost-aware-N9to16-root-v216-full463-AR-v1'
FOLDER = ACTUAL/'windows/window32/full463'
INPUTS = [FOLDER/f'rank{rank}-full463-actual-trace.json.gz' for rank in range(4)]


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual = [read(p) for p in INPUTS]

    def test_actual_native_only_leader_materializes_results(self):
        self.assertEqual([type(t['generate_results']).__name__ for t in self.actual],
                         ['NoneType', 'list', 'NoneType', 'NoneType'])
        native = subject.Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
        values = [ast.unparse(n.value) for n in ast.walk(ast.parse(native.read_text())) if isinstance(n, ast.Return) and n.value is not None]
        self.assertIn('results if self.rank == self.topology.target_leader_rank else None', values)

    def test_all_four_actual_roles_load_same_comparison_only_artifact(self):
        rows = [subject.load_candidate(p, t, 1) for p, t in zip(INPUTS, self.actual, strict=True)]
        self.assertTrue(all(row == rows[0] for row in rows))
        self.assertEqual(len(rows[0][0]), 463)
        self.assertEqual(rows[0][1]['sha256'], hashlib.sha256(INPUTS[1].read_bytes()).hexdigest())
        self.assertEqual([t['rank'] for t in self.actual], list(range(4)))
        self.assertTrue(all(t['generate_results'] is None for i, t in enumerate(self.actual) if i != 1))

    def test_original_role_ownership_is_not_silently_rewritten(self):
        own = dict(self.actual[0], generate_results=self.actual[1]['generate_results'])
        with self.assertRaisesRegex(RuntimeError, 'None result ownership'):
            subject.load_candidate(INPUTS[0], own, 1)
        with self.assertRaisesRegex(RuntimeError, 'topology'):
            subject.load_candidate(INPUTS[0], self.actual[0], 0)

    def test_cross_run_and_cross_cohort_inputs_rejected(self):
        for key in ('manifest_sha256', 'prefix_sha256', 'source_generation',
                    'tokenized_inputs', 'requests', 'standalone_prefill_calls'):
            with self.subTest(key=key):
                own = dict(self.actual[2])
                own[key] = 'incorrect identity'
                with self.assertRaisesRegex(RuntimeError, 'input and cohort identity'):
                    subject.load_candidate(INPUTS[2], own, 1)

    def test_incomplete_duplicate_short_or_invalid_leader_outputs_rejected(self):
        for mode in ('missing', 'duplicate', 'short', 'invalid_token'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                leader = dict(self.actual[1])
                results = copy.deepcopy(leader['generate_results'])
                if mode == 'missing':
                    results.pop()
                elif mode == 'duplicate':
                    results[-1]['request_id'] = results[0]['request_id']
                elif mode == 'short':
                    results[0]['completion_token_ids'].pop()
                else:
                    results[0]['completion_token_ids'][0] = -1
                leader['generate_results'] = results
                path = Path(folder)/INPUTS[1].name
                path.write_bytes(gzip.compress(json.dumps(leader).encode(), mtime=0))
                with self.assertRaises(RuntimeError):
                    subject.load_candidate(Path(folder)/INPUTS[0].name, self.actual[0], 1)

    def test_actual_load_is_one_synchronized_preallocation_stage(self):
        source = (P/'reference218.py').read_text()
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        stages = [n for n in fn.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                  and isinstance(n.value.func, ast.Name) and n.value.func.id == 'stage']
        loads = [n for n in stages if n.value.args[0].value == 'AR218/retained-output-leader-comparison-only']
        self.assertEqual(len(loads), 1)
        self.assertIsInstance(loads[0].value.args[1], ast.Lambda)
        self.assertIn('load_candidate(artifact, trace, engine.topology.target_leader_rank)', ast.unparse(loads[0]))
        self.assertLess(fn.body.index(loads[0]), next(i for i,n in enumerate(fn.body) if isinstance(n, ast.For)))
        self.assertNotIn("for r in trace['generate_results']", source)

    def test_independent_auditor_never_iterates_nonleader_None(self):
        source = (P/'audit_reference218.py').read_text()
        tree = ast.parse(source)
        candidate = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'candidate' for t in n.targets)]
        self.assertEqual(len(candidate), 1)
        self.assertEqual(ast.unparse(candidate[0].value.generators[0].iter), "leader['generate_results']")
        self.assertIn("summary['candidate_result_artifact'] == leader_info", ast.unparse(tree))
        self.assertIn("trace['generate_results'] is None", ast.unparse(tree))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), Torch_imported='torch' in sys.modules,
        NPU_used=False, actual_failed216_role_inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS})))
    raise SystemExit(0 if result.wasSuccessful() else 1)
