"""Actual V96 HOST regression plus strict complete-boundary negative cases."""
import copy
import gzip
import json
from pathlib import Path
import sys
import unittest

import analyze_v96_actual_capacity64 as evidence
import actual_boundary_schema_host as schema


class CompleteBoundaryHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=evidence.SOURCE/'local-output/capacity64-full463-v1/rank0-same-M-qualification.json.gz'
        cls.raw=json.loads(gzip.decompress(cls.path.read_bytes()))
        cls.guard=evidence.frozen_host_modules();cls.modules=cls.guard.__enter__()
        cls.scope=cls.modules['local_qualification_scope'].scopes()[0]

    @classmethod
    def tearDownClass(cls):cls.guard.__exit__(None,None,None)

    def test_actual_complete_raw_events_and_old_error_preserved(self):
        row=self.raw['scopes'][0];before=copy.deepcopy(row)
        result=schema.audit_raw_boundaries(row,self.scope,'draft',0)
        self.assertEqual((result['raw_events'],result['common_status_boundaries'],result['exact_prompt_materialization_events']),(251,243,2))
        self.assertEqual(row,before)
        with self.assertRaisesRegex(RuntimeError,'materialization'):
            self.modules['bounded_report_audit_v78'].chain_record(self.raw,'draft',rank=0)

    def test_status_error_cannot_be_removed_as_metadata(self):
        row=copy.deepcopy(self.raw['scopes'][0]);row['atomic_runtime_evidence']['boundary_events'][0]['ranks'][2]['error']='bad'
        with self.assertRaisesRegex(schema.BoundaryRejected,'agreement'):schema.audit_raw_boundaries(row,self.scope,'draft',0)

    def test_missing_ledger_is_not_inferred_from_common_status(self):
        row=copy.deepcopy(self.raw['scopes'][0]);events=row['atomic_runtime_evidence']['boundary_events']
        del events[next(i for i,e in enumerate(events) if e['stage']=='actual-prompt-draft-materialization-ledger')]
        with self.assertRaises(schema.BoundaryRejected):schema.audit_raw_boundaries(row,self.scope,'draft',0)

    def test_unknown_event_and_bad_commit_fail(self):
        row=copy.deepcopy(self.raw['scopes'][0]);row['atomic_runtime_evidence']['boundary_events'].append(dict(stage='unknown'))
        with self.assertRaisesRegex(schema.BoundaryRejected,'Unknown'):schema.audit_raw_boundaries(row,self.scope,'draft',0)
        row=copy.deepcopy(self.raw['scopes'][0])
        event=next(e for e in row['atomic_runtime_evidence']['boundary_events'] if e['stage']=='original-tree-commit-evidence')
        event['ranks'][1]['numerical_failed']=1
        with self.assertRaisesRegex(schema.BoundaryRejected,'compaction'):schema.audit_raw_boundaries(row,self.scope,'draft',0)

    def test_isolated_AST_keeps_substantive_numeric_rejections(self):
        check,provenance=schema.isolated_chain_validator(self.modules['bounded_report_audit_v78'])
        self.assertTrue(provenance['original_chain_AST_unchanged'])
        check(self.raw,'draft',rank=0)
        bad=copy.deepcopy(self.raw);bad['scopes'][0]['phases'][0]['draft_numeric']['all28_query_KV_original_matcher']=False
        with self.assertRaisesRegex(RuntimeError,'all28'):check(bad,'draft',rank=0)


if __name__=='__main__':unittest.main()
