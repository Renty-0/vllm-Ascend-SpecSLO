"""HOST negative controls for the external V98 auditor; no device work."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import analyze_v98_actual_s35 as audit


class S35ExternalHost(unittest.TestCase):
    def test_gzip_SHA_path_and_length_are_not_inferred(self):
        with tempfile.TemporaryDirectory() as directory,tempfile.TemporaryDirectory() as other:
            path=Path(directory)/'actual.json.gz';raw=b'{"actual":true}';path.write_bytes(gzip.compress(raw,mtime=0))
            pointer=audit.descriptor(path)|dict(uncompressed_bytes=len(raw))
            self.assertEqual(audit.read_gzip(pointer,directory),dict(actual=True))
            for bad in (pointer|dict(sha256='0'*64),pointer|dict(bytes=1),pointer|dict(uncompressed_bytes=1)):
                with self.assertRaises(audit.Rejected):audit.read_gzip(bad,directory)
            with self.assertRaisesRegex(audit.Rejected,'escaped'):audit.read_gzip(pointer,other)

    def test_archive_requires_absolute_normalized_unique_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);source=p/'source.py';source.write_text('actual=1\n')
            destination=p/'bound.tar.gz';name=str(source).lstrip('/')
            with tarfile.open(destination,'w:gz') as stream:stream.add(source,arcname=name)
            expected={str(source):audit.sha(source)}
            self.assertEqual(audit.archive(destination,expected)['regular_members'],1)
            with self.assertRaisesRegex(audit.Rejected,'bytes'):
                audit.archive(destination,{str(source):'0'*64})
            duplicate=p/'duplicate.tar.gz'
            with tarfile.open(duplicate,'w:gz') as stream:
                stream.add(source,arcname=name);stream.add(source,arcname=name)
            with self.assertRaisesRegex(audit.Rejected,'coverage'):audit.archive(duplicate,expected)

    def test_one_new_key_rejects_extra_graph_or_removed_parent(self):
        before=dict(entries=[],draft_entries=[],target_entries=['old'])
        report=dict(rank=1,inventory_before=before,inventory_after=dict(entries=[],draft_entries=[],target_entries=['old','new']))
        self.assertEqual(audit.inventory_transition(report,'new')['new_hidden_keys'],1)
        bad=copy.deepcopy(report);bad['inventory_after']['target_entries'].append('extra')
        with self.assertRaisesRegex(audit.Rejected,'one allowed'):audit.inventory_transition(bad,'new')
        bad=copy.deepcopy(report);bad['inventory_after']['target_entries'].remove('old')
        with self.assertRaises(audit.Rejected):audit.inventory_transition(bad,'new')

    def good_cost(self):
        geometry=dict(tree_queries=128,prompt_queries=8,padding_queries=61,target_head_rows=129)
        delta={key:int(key in ('total_calls','replay_calls')) for key in ('total_calls','capture_replay_calls',
            'replay_calls','eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls',
            'runtime_validation_failures','changed_input_validation_calls','logical_row_expansion_validation_calls')}
        key=repr(('target-hidden:0|steps:1|tree-fia-full:exact-length-task-update|buffers:'
                  '(((35, 1, 128, 4096), (35, 32)),)',(197,)))
        records=[dict(repeat=i,elapsed_ms=10.+i,physical_M=197,physical_S=35,
            **geometry,head_rows=129,sealed=True,native_delta=dict(delta),
            entry=dict(key=key,runtime_validated=True,segments=35,mask_shape=[35,1,128,4096]),
            hidden_and_all64_query_KV_original_matcher=True,exact_heads=True,committed_prefix_unchanged=True)
            for i in range(5)]
        for row in records:row['scratch_queries']=row.pop('padding_queries');row.pop('target_head_rows')
        return dict(controlled_same_phase_repeated_state=True,sequential_repeat_count=5,
            statistically_independent_samples_claimed=False,production_overlap_qualified=False,formal_Goodput=False,
            role='target',records=records),geometry

    def test_sealed_cost_requires_five_finite_true_replays_and_all64_KV(self):
        cost,geometry=self.good_cost()
        self.assertEqual(audit.phase_cost(cost,geometry,rank=1,trace_index=2,scope_index=2),[10.,11.,12.,13.,14.])
        for key,value in (('elapsed_ms',float('nan')),('hidden_and_all64_query_KV_original_matcher',False),('sealed',False)):
            bad=copy.deepcopy(cost);bad['records'][1][key]=value
            with self.assertRaises(audit.Rejected):audit.phase_cost(bad,geometry,rank=1,trace_index=2,scope_index=2)
        bad=copy.deepcopy(cost);bad['records'][0]['native_delta']['runtime_validation_calls']=1
        with self.assertRaisesRegex(audit.Rejected,'resident replay'):audit.phase_cost(bad,geometry,rank=1,trace_index=2,scope_index=2)

    def test_draft_cannot_attest_target_cost_or_widen_scope(self):
        cost,geometry=self.good_cost();cost.update(role='draft',records=[None]*5)
        self.assertEqual(audit.phase_cost(cost,geometry,rank=0,trace_index=0,scope_index=0),[])
        cost['records'][0]={}
        with self.assertRaisesRegex(audit.Rejected,'Draft'):audit.phase_cost(cost,geometry,rank=0,trace_index=0,scope_index=0)
        cost,geometry=self.good_cost();cost['statistically_independent_samples_claimed']=True
        with self.assertRaisesRegex(audit.Rejected,'scope'):audit.phase_cost(cost,geometry,rank=1,trace_index=2,scope_index=2)

    def test_actual_genuine_fixture_matches_old_tokens_without_tokenizer_or_model_import(self):
        value=audit.genuine_inputs(audit.SOURCE)
        self.assertTrue(value['normal_token_SHA_equal_original_V88_actual'])
        self.assertFalse(value['original_V88_checkpoint_retains_token_arrays'])
        self.assertTrue(value['full_prompt_equals_original_V45'])
        self.assertFalse(value['source_token_binding_is_own_AR_equality'])

    def test_frozen_HOST_validators_import_under_device_import_guard(self):
        with audit.validators() as modules:
            self.assertTrue(callable(modules['audit_s35_v98'].audit_reports))
            with self.assertRaisesRegex(audit.Rejected,'forbids'):__import__('torch_npu')

    def test_fusion_unknown_payload_is_not_an_ignored_runtime_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fusion_result.json'
            path.write_text('{"session_and_graph_id_1_1":{"graph_fusion":{"TransdataCastFusionPass":{"effect_times":"0","match_times":2}}}}')
            self.assertEqual(audit.fusion(path)['graph_records'],1)
            path.write_text('{"session_and_graph_id_1_1":{"graph_fusion":{"UnknownPass":{"effect_times":0,"match_times":2}}}}')
            with self.assertRaises(audit.Rejected):audit.fusion(path)

    def test_running_does_not_become_a_terminal_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'local-receipt-running.json'
            path.write_text(json.dumps(dict(schema='V98.mixed-S35-supervisor.v1',status='running')))
            with self.assertRaisesRegex(audit.Rejected,'terminal'):audit.analyze('running',Path(directory))

    def test_all_fifty_two_raw_artifacts_have_unique_rank_trace_scope_names(self):
        paths=audit.expected_raw_artifacts('/owned/run')
        self.assertEqual(len(paths),52)
        self.assertEqual(len(set(paths)),52)
        self.assertIn(Path('/owned/run/rank0-zero2-scope2.json.gz'),paths)
        self.assertIn(Path('/owned/run/rank3-S35-bounded.json.gz'),paths)


if __name__=='__main__':unittest.main()
