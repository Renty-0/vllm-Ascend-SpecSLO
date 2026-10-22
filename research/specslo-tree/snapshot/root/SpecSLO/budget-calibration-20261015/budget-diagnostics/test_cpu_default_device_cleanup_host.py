"""Real retained-PyTorch CPU state tests; no Native/model/NPU imports."""
import ast
import dataclasses
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys
import threading
import unittest
import warnings
from unittest.mock import patch

import torch
from torch.overrides import TorchFunctionMode, _get_current_function_mode_stack

from cpu_default_device_cleanup_host import (
    capture_before_native_constructor, reinstall_redundant_cpu_device_context,
    restore_redundant_cpu_device_context, observe_cpu_device_context,
)

TREE_SOURCE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/tree.py')
TREE_SHA = '692b2d225d1fedffb84eee7afca7030609a3b4f29d1796cddde448780b5b911a'
MODEL_SOURCE = TREE_SOURCE.with_name('native_model.py')
MODEL_SHA = 'e0d9efe4572c7e87ad860b014ef146394ca44f53ac8d87d12edfed2b572422a1'
LAST_TREE_EQUIVALENCE = {}


class RecordingMode(TorchFunctionMode):
    def __init__(self): self.calls = []
    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append(func.__name__)
        return func(*args, **(kwargs or {}))


def load_pinned_tree(name):
    if hashlib.sha256(TREE_SOURCE.read_bytes()).hexdigest() != TREE_SHA:
        raise AssertionError('original Native tree source changed')
    spec = importlib.util.spec_from_file_location(name, TREE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plan_domain_digest(name):
    """New module and caches for each arm; enumerate all closed B<=5 subsets.

    This proves byte equality only for the explicit CPU planning domain below,
    not arbitrary Python functions, NPU operations, or model numerical results.
    """
    tree = load_pinned_tree(name)
    digest = hashlib.sha256()
    count = 0
    try:
        for width, depth in [(1, depth) for depth in range(1, 6)] + [(2, 4)]:
            parents = tree._spine_first_topology(width, depth)[0]
            subsets = [subset for n in range(1, min(5, len(parents)) + 1)
                       for subset in itertools.combinations(range(len(parents)), n)
                       if all(parents[i] == -1 or parents[i] in subset for i in subset)]
            for prefix in (0, 1, 127, 128, 170, 298, 4087):
                for budget in range(1, min(5, len(parents)) + 1):
                    for build in (tree.build_tree_speculation_plan, tree.cached_cpu_tree_speculation_plan):
                        plan = build(width, depth, prefix, 4096, candidate_budget=budget)
                        for selected in (None, *subsets):
                            actual = plan if selected is None else tree.pack_selected_tree_plan(plan, selected)
                            for field in dataclasses.fields(actual):
                                value = getattr(actual, field.name)
                                digest.update(field.name.encode())
                                if isinstance(value, torch.Tensor):
                                    if value.device.type != 'cpu':
                                        raise AssertionError('CPU planning allocated a non-CPU tensor')
                                    descriptor = (str(value.dtype), tuple(value.shape), tuple(value.stride()), value.storage_offset())
                                    digest.update(repr(descriptor).encode())
                                    digest.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())
                                else:
                                    digest.update(repr(value).encode())
                            count += 1
        return {'sha256': digest.hexdigest(), 'plan_records': count,
                'domain': 'width1 depth1..5 + width2 depth4; B1..5; every nonempty ancestor-closed selected subset <=5; prefixes0,1,127,128,170,298,4087; maxlen4096; fresh/cached builders'}
    finally:
        sys.modules.pop(name, None)


class DefaultCPUContextTests(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.get_default_dtype()
        self.state = capture_before_native_constructor(torch)
        if not self.state.eligible:
            self.fail('test process must begin with unmodified default CPU/no DeviceContext')

    def tearDown(self):
        torch.set_default_device(None)
        torch.set_default_dtype(self.dtype)
        self.assertEqual(tuple(_get_current_function_mode_stack()), ())
        self.assertEqual(str(torch.get_default_device()), 'cpu')
        if hasattr(torch, 'npu'):
            self.assertFalse(torch.npu.is_initialized())

    def test_native_cpu_restore_pattern_leaves_mode_and_strict_cleanup(self):
        original_device = torch.get_default_device()
        # CPU stand-in for Native's temporary NPU device; no NPU is touched.
        torch.set_default_device('cpu')
        torch.set_default_device(original_device)
        after = tuple(_get_current_function_mode_stack())
        self.assertEqual(len(after), 1)
        self.assertEqual(str(after[0].device), 'cpu')
        result = restore_redundant_cpu_device_context(torch, self.state)
        self.assertEqual(result['status'], 'cleaned')
        self.assertEqual(tuple(_get_current_function_mode_stack()), self.state.mode_stack)
        self.assertEqual(torch.get_default_dtype(), self.dtype)
        self.assertEqual(torch.get_default_device(), self.state.default_device)

    def test_repeat_diagnostic_control_restores_original_other_mode_identity(self):
        with RecordingMode() as mode:
            state = capture_before_native_constructor(torch)
            for _ in range(3):
                self.assertEqual(reinstall_redundant_cpu_device_context(torch, state)['status'], 'reinstalled')
                torch.ones(2, device='cpu')
                self.assertEqual(restore_redundant_cpu_device_context(torch, state)['status'], 'cleaned')
                self.assertIs(tuple(_get_current_function_mode_stack())[0], mode)
                self.assertEqual(len(tuple(_get_current_function_mode_stack())), 1)
            self.assertEqual(mode.calls.count('ones'), 3)

    def test_existing_user_cpu_context_preserved(self):
        torch.set_default_device('cpu')
        state = capture_before_native_constructor(torch)
        before = tuple(_get_current_function_mode_stack())
        self.assertFalse(state.eligible)
        self.assertEqual(restore_redundant_cpu_device_context(torch, state)['status'], 'preserved')
        self.assertEqual(reinstall_redundant_cpu_device_context(torch, state)['status'], 'preserved')
        self.assertTrue(all(a is b for a, b in zip(before, _get_current_function_mode_stack())))

    def test_non_cpu_default_is_not_read_or_mutated(self):
        torch.set_default_device('meta')  # Creates a mode only; no tensor or accelerator.
        mode = tuple(_get_current_function_mode_stack())[0]
        with patch.object(torch, 'get_default_device', side_effect=AssertionError('must not resolve non-CPU default')):
            state = capture_before_native_constructor(torch)
            self.assertFalse(state.eligible)
            self.assertEqual(restore_redundant_cpu_device_context(torch, state)['status'], 'preserved')
            self.assertEqual(restore_redundant_cpu_device_context(torch, self.state)['status'], 'preserved')
        self.assertIs(tuple(_get_current_function_mode_stack())[0], mode)

    def test_legacy_non_cpu_device_is_rejected_without_tensor_probe(self):
        with patch.object(torch._C, '_get_default_device', return_value='meta'):
            with patch.object(torch, 'get_default_device', side_effect=AssertionError('must not probe legacy non-CPU default')):
                state = capture_before_native_constructor(torch)
                self.assertFalse(state.eligible)
                self.assertEqual(state.reason, 'before_legacy_default_device_is_not_cpu')
                self.assertEqual(restore_redundant_cpu_device_context(torch, state)['status'], 'preserved')
                self.assertEqual(reinstall_redundant_cpu_device_context(torch, state)['status'], 'preserved')
        self.assertEqual(tuple(_get_current_function_mode_stack()), ())

    def test_new_other_mode_and_non_global_context_preserved(self):
        with RecordingMode():
            torch.set_default_device('cpu')
            before = tuple(_get_current_function_mode_stack())
            result = restore_redundant_cpu_device_context(torch, self.state)
            self.assertEqual(result['status'], 'preserved')
            self.assertTrue(all(a is b for a, b in zip(before, _get_current_function_mode_stack())))
            torch.set_default_device(None)
        with torch.device('cpu'):
            before = tuple(_get_current_function_mode_stack())
            result = restore_redundant_cpu_device_context(torch, self.state)
            self.assertEqual(result['status'], 'preserved')
            self.assertEqual(result['reason'], 'extra_mode_is_not_global_context_owned')
            self.assertIs(tuple(_get_current_function_mode_stack())[0], before[0])

    def test_dtype_api_and_thread_changes_preserved(self):
        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float64)
        self.assertEqual(restore_redundant_cpu_device_context(torch, self.state)['reason'], 'default_dtype_changed')
        self.assertEqual(torch.get_default_dtype(), torch.float64)
        torch.set_default_dtype(self.dtype)
        with patch.object(torch, 'get_default_device', side_effect=AssertionError('changed API must not run')):
            self.assertEqual(restore_redundant_cpu_device_context(torch, self.state)['reason'], 'torch_api_changed')
        outcomes = []
        thread = threading.Thread(target=lambda: outcomes.append(restore_redundant_cpu_device_context(torch, self.state)))
        thread.start(); thread.join()
        self.assertEqual(outcomes[0]['reason'], 'different_torch_or_thread')
        self.assertEqual(len(tuple(_get_current_function_mode_stack())), 1)

    def test_no_extra_mode_and_invalid_reinstall_preserved(self):
        self.assertTrue(observe_cpu_device_context(torch, self.state, redundant_cpu=False)['passed'])
        self.assertFalse(observe_cpu_device_context(torch, self.state, redundant_cpu=True)['passed'])
        self.assertEqual(restore_redundant_cpu_device_context(torch, self.state)['status'], 'preserved')
        torch.set_default_device('cpu')
        before = tuple(_get_current_function_mode_stack())
        self.assertTrue(observe_cpu_device_context(torch, self.state, redundant_cpu=True)['passed'])
        self.assertFalse(observe_cpu_device_context(torch, self.state, redundant_cpu=False)['passed'])
        self.assertEqual(reinstall_redundant_cpu_device_context(torch, self.state)['status'], 'preserved')
        self.assertIs(tuple(_get_current_function_mode_stack())[0], before[0])
        with self.assertRaises(TypeError):
            observe_cpu_device_context(torch, self.state, redundant_cpu=1)

    def test_constructor_error_is_not_masked_by_conditional_cleanup(self):
        error = ValueError('original constructor sentinel')
        def original():
            try:
                torch.set_default_device('cpu')
                raise error
            finally:
                torch.set_default_device('cpu')
        try:
            original()
        except ValueError as caught:
            outcome = restore_redundant_cpu_device_context(torch, self.state)
            self.assertIs(caught, error)
            self.assertEqual(outcome['status'], 'cleaned')
        else:
            self.fail('original exception disappeared')

    def test_original_native_finally_restores_device_value_and_dtype(self):
        self.assertEqual(hashlib.sha256(MODEL_SOURCE.read_bytes()).hexdigest(), MODEL_SHA)
        module = ast.parse(MODEL_SOURCE.read_text())
        fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'build_native_model')
        body = ast.unparse(fn)
        self.assertIn('original_device = torch.get_default_device()', body)
        self.assertIn("torch.set_default_device('npu')", body)
        final = next(node for node in fn.body if isinstance(node, ast.Try)).finalbody
        self.assertEqual([ast.unparse(node) for node in final],
                         ['torch.set_default_dtype(original_dtype)', 'torch.set_default_device(original_device)'])

    def test_failed_mutation_postcondition_never_labels_state_cleaned(self):
        original_set = torch.set_default_device
        def no_cleanup(device):
            if device is not None:
                return original_set(device)
        with patch.object(torch, 'set_default_device', no_cleanup):
            state = capture_before_native_constructor(torch)
            torch.set_default_device('cpu')
            with self.assertRaisesRegex(RuntimeError, 'cleanup postcondition failed'):
                restore_redundant_cpu_device_context(torch, state)
            self.assertEqual(len(tuple(_get_current_function_mode_stack())), 1)

    def test_default_constructor_dtype_explicit_cpu_values_equal(self):
        def outputs():
            values = [torch.tensor([1., 2.]), torch.empty(0), torch.ones(2),
                      torch.arange(4), torch.zeros(2, dtype=torch.bfloat16),
                      torch.tensor([3, 4], dtype=torch.int32, device='cpu')]
            return [(str(t.device), str(t.dtype), tuple(t.shape), t.tolist()) for t in values]
        before = outputs()
        reinstall_redundant_cpu_device_context(torch, self.state)
        redundant = outputs()
        restore_redundant_cpu_device_context(torch, self.state)
        self.assertEqual(before, redundant)
        self.assertEqual(before, outputs())

    def test_non_cpu_tensor_data_without_device_is_a_real_semantic_counterexample(self):
        source = torch.empty(2, device='meta')  # Metadata-only, no accelerator.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for factory in (torch.tensor, torch.as_tensor, torch.asarray):
                self.assertEqual(factory(source).device.type, 'meta')
            reinstall_redundant_cpu_device_context(torch, self.state)
            for factory in (torch.tensor, torch.as_tensor, torch.asarray):
                with self.assertRaises((NotImplementedError, RuntimeError)):
                    factory(source)
            restore_redundant_cpu_device_context(torch, self.state)
            for factory in (torch.tensor, torch.as_tensor, torch.asarray):
                self.assertEqual(factory(source).device.type, 'meta')

    def test_all_explicit_cpu_tree_plan_domain_is_byte_equal_with_fresh_caches(self):
        before = plan_domain_digest('_cleanup_tree_before')
        reinstall_redundant_cpu_device_context(torch, self.state)
        redundant = plan_domain_digest('_cleanup_tree_cpu_context')
        restore_redundant_cpu_device_context(torch, self.state)
        after = plan_domain_digest('_cleanup_tree_after')
        self.assertEqual(before, redundant)
        self.assertEqual(before, after)
        LAST_TREE_EQUIVALENCE.update(before=before, redundant=redundant, after=after)


if __name__ == '__main__': unittest.main(verbosity=2)
