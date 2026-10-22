"""Execute original Native updater/checker AST with explicit HOST device stubs.

The stub CANN/event transcript verifies that only duplicate host validation is
removed. No model, Native module import, tensor upload or NPU initialization.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest

import torch

from scoped_tree_validation_memo import ScopedTreeValidationMemo

NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')


def original_host_module():
    source=ast.parse(NATIVE.read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='NativeACLGraphRunner')
    functions=[next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_tree_fia_mask')]
    functions += [next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
                  for name in ('_update_attention_task_list','_update_fused_infer_attention_task')]
    # Annotations are postponed, just as in the original Native module.
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    transcript=[]
    def record(name):
        return lambda *args,**kw:transcript.append((name,args,kw))
    fake_torch=SimpleNamespace(Tensor=torch.Tensor,bool=torch.bool,npu=SimpleNamespace(
        stream=lambda _:nullcontext(),graph_task_update_begin=record('begin'),
        graph_task_update_end=record('end')))
    class FIA:pass
    namespace=dict(torch=fake_torch,NativeFusedInferAttentionGraphTask=FIA,
        TARGET_FIA_TASK_EVENT_GROUP_SIZES=(1,2,4),TREE_FIA_SPARSE_MODE=1,TREE_FIA_INNER_PRECISE=0,
        torch_npu=SimpleNamespace(npu_fused_infer_attention_score=SimpleNamespace(out=record('FIA'))))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*functions],type_ignores=[])),str(NATIVE),'exec'),namespace)
    class Runner:
        update_stream=object()
        profile_pa_task_update=False
        _update_attention_task_list=namespace['_update_attention_task_list']
        _update_fused_infer_attention_task=namespace['_update_fused_infer_attention_task']
        def _validate_task_event_groups(self,*_):pass
        def _try_parallel_fia_task_update(self,*_,**__):return False
        def _task_event_group_ranges(self,n,size,prefix):return [range(i,min(i+size,n)) for i in range(0,n,size)]
    module=SimpleNamespace(_validate_tree_fia_mask=namespace['_validate_tree_fia_mask'],
                           NativeACLGraphRunner=Runner,torch=fake_torch)
    # The updater's function globals are its actual lookup surface. A proxy
    # forwards patching the validator into those original globals.
    class Proxy:
        NativeACLGraphRunner=Runner
        torch=fake_torch
        @property
        def _validate_tree_fia_mask(self):return namespace['_validate_tree_fia_mask']
        @_validate_tree_fia_mask.setter
        def _validate_tree_fia_mask(self,value):namespace['_validate_tree_fia_mask']=value
    def tasks():
        mask=torch.empty((16,1,9,4096),dtype=torch.bool)
        table=torch.empty((16,32),dtype=torch.int32)
        result=[]
        for layer in range(64):
            task=FIA()
            for name in ('query','key_cache','value_cache','output','softmax_lse','workspace'):
                setattr(task,name,torch.empty((96,1)))
            task.attention_mask=mask;task.block_table=table;task.block_size=128
            task.num_kv_heads=3;task.num_heads=22;task.scale=.125;task.tree_attention=True;task.handle=layer
            task.event=SimpleNamespace(record=record('event'))
            result.append(task)
        return result
    return Proxy(),namespace,transcript,tasks


class MemoHostTests(unittest.TestCase):
    def setUp(self):
        self.module,self.namespace,self.transcript,factory=original_host_module()
        self.tasks=factory();self.runner=self.module.NativeACLGraphRunner()
        self.lengths=[(tuple(range(6,97,6)),(176,)*16)]*64
        self.memo=ScopedTreeValidationMemo(self.module)

    def tearDown(self):
        if self.memo.installed:self.memo.uninstall()

    def run_tasks(self):self.runner._update_attention_task_list(self.tasks,self.lengths)

    def test_original_CANN_and_event_transcript_is_identical(self):
        self.run_tasks();original=list(self.transcript);self.transcript.clear()
        self.memo.install();self.run_tasks()
        self.assertEqual(len(original),256)
        self.assertEqual(self.transcript,original)
        self.assertEqual((self.memo.original_calls,self.memo.memo_hits),(1,63))

    def test_every_replay_must_revalidate(self):
        self.memo.install();self.run_tasks();self.run_tasks()
        self.assertEqual((self.memo.original_calls,self.memo.memo_hits),(2,126))
        self.assertIsNone(self.memo.scope.get())

    def test_changed_interior_layer_bad_KV_is_rejected(self):
        self.memo.install();self.lengths[30]=(self.lengths[30][0],(0,)+(176,)*15)
        with self.assertRaises(ValueError):self.run_tasks()
        self.assertEqual(self.memo.original_calls,2);self.assertIsNone(self.memo.scope.get())
        self.assertEqual(sum(name=='FIA' for name,_,_ in self.transcript),30)

    def test_changed_mask_shape_is_rejected_after_first_success(self):
        self.memo.install();self.tasks[10].attention_mask=torch.empty((16,1,5,4096),dtype=torch.bool)
        with self.assertRaises(ValueError):self.run_tasks()
        self.assertEqual(self.memo.original_calls,2)

    def test_changed_table_capacity_is_rejected(self):
        self.memo.install();self.tasks[10].block_table=torch.empty((16,1),dtype=torch.int32)
        with self.assertRaises(ValueError):self.run_tasks()

    def test_changed_valid_lengths_have_own_validation(self):
        self.memo.install();self.lengths[20]=(self.lengths[20][0],(177,)*16)
        self.run_tasks();self.assertEqual((self.memo.original_calls,self.memo.memo_hits),(2,62))

    def test_capture_or_eager_outside_scope_always_calls_original(self):
        self.memo.install();task=self.tasks[0]
        for _ in range(3):self.module._validate_tree_fia_mask(96,task.attention_mask,task.block_table,
                                                          *self.lengths[0],128)
        self.assertEqual((self.memo.original_calls,self.memo.memo_hits),(3,0))

    def test_mutated_length_list_values_cannot_hit_old_key(self):
        self.memo.install();q=list(self.lengths[0][0]);kv=[176]*16;task=self.tasks[0]
        token=self.memo.scope.set(set())
        try:
            self.module._validate_tree_fia_mask(96,task.attention_mask,task.block_table,q,kv,128)
            kv[0]=0
            with self.assertRaises(ValueError):
                self.module._validate_tree_fia_mask(96,task.attention_mask,task.block_table,q,kv,128)
        finally:self.memo.scope.reset(token)
        self.assertEqual((self.memo.original_calls,self.memo.memo_hits),(2,0))

    def test_new_thread_uses_original_without_inherited_memo(self):
        self.memo.install();task=self.tasks[0];errors=[]
        token=self.memo.scope.set(set())
        def run():
            try:
                self.assertIsNone(self.memo.scope.get())
                self.module._validate_tree_fia_mask(96,task.attention_mask,task.block_table,*self.lengths[0],128)
            except BaseException as e:errors.append(e)
        try:
            thread=threading.Thread(target=run);thread.start();thread.join()
        finally:self.memo.scope.reset(token)
        self.assertEqual(errors,[]);self.assertEqual((self.memo.original_calls,self.memo.memo_hits),(1,0))

    def test_removal_restores_original_functions(self):
        validate=self.module._validate_tree_fia_mask;update=self.runner._update_attention_task_list.__func__
        self.memo.install();self.memo.uninstall()
        self.assertIs(self.module._validate_tree_fia_mask,validate)
        self.assertIs(self.runner._update_attention_task_list.__func__,update)


if __name__=='__main__':unittest.main()
