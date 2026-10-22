"""Pure HOST controls for the external parser; never executes actual run()."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import analyze_v103_actual_active_cap_controls as a


def complete_receipt(tag='HOST-only'):
    hardware=dict(exit_code=0,timed_out=False,output=''.join(f'| {i} 910B2 |\n' for i in (2,5,6,7)))
    idle=dict(hardware=hardware,parsed_hardware=[[i,'910B2'] for i in (2,5,6,7)],four_910B2=True,all_idle=True,
        processes={str(i):dict(argv=['npu-smi','info','-t','proc-mem','-i',str(i),'-c','0'],
            idle=True,exit_code=0,timed_out=False,output='No process in device.') for i in (2,5,6,7)})
    verifier=dict(exit_code=0,argv=['bash',str(a.BASE/'serial-v1/runtime.sh'),'python','-B',
        '/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py',
        '/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json'])
    return dict(schema='V103.persistent-cap-controls-supervisor.v1',status='complete',action=a.ACTION,errors=[],
        local_execution_complete=True,formal_Goodput=False,online_qualified=False,
        child=dict(pid=1234,exit_code=0,argv=['bash',str(a.WORK/'launch_local_qualification.sh'),'--action',a.ACTION,'--tag',tag]),
        owned_remaining=[],physical_cards=[2,5,6,7],limits=dict(seconds=7200,private_bytes=256*1024**2,file_bytes=64*1024**2),
        elapsed_seconds=900.,private_bytes=1024,experimental_sources_unchanged=True,external_sources_unchanged=True,
        devices_before=copy.deepcopy(idle),devices_after=copy.deepcopy(idle),pre_verifier=copy.deepcopy(verifier),post_verifier=copy.deepcopy(verifier))


def trace():
    classes=['tight']*271+['normal']*96+['loose']*96;sl={'tight':40.,'normal':50.,'loose':150.}
    requests=[dict(request_index=i,request_id=f'HOST-{i}',slo_class=c,slo_tpot_ms=sl[c]) for i,c in enumerate(classes)]
    results=[]
    for row in requests:
        p={'tight':40.,'normal':60.,'loose':100.}[row['slo_class']]
        results.append(dict(request_id=row['request_id'],slo_class=row['slo_class'],slo_tpot_ms=row['slo_tpot_ms'],
            completion_token_ids=list(range(256)),paper_tpot_ms=p,observed_tpot_ms=p*256/255,
            paper_slo_attained=p<=row['slo_tpot_ms'],paper_slo_goodput_tokens=256 if p<=row['slo_tpot_ms'] else 0,
            elapsed_seconds=49.,verification_rounds=100))
    events=[]
    for i,row in enumerate(requests):events.append(dict(request_index=i,request_id=row['request_id'],token_ids=[0],elapsed_seconds=1.,finished=False))
    for i,row in enumerate(requests):events.append(dict(request_index=i,request_id=row['request_id'],token_ids=list(range(1,256)),elapsed_seconds=10.,finished=True))
    return dict(requests=requests,generate_results=results,stream_events=events,host_generate_call_seconds=50.,native_elapsed_seconds=49.)


class TerminalTests(unittest.TestCase):
    def test_terminal_complete_original_cleanup_and_commands_control(self):
        r=complete_receipt();self.assertIs(a.terminal_receipt(r,'HOST-only'),r)

    def test_running_and_failed_rejected_before_reading_full_or_profile(self):
        for status in ('running','failed',None):
            r=complete_receipt();r['status']=status
            with patch.object(a,'read',return_value=r),patch.object(a,'source_closure',side_effect=AssertionError('too early')):
                with self.assertRaisesRegex(RuntimeError,'terminal complete'):a.run('HOST-only')

    def test_missing_cleanup_native_exit_and_qualification_claim_are_not_pass(self):
        for mutate in (lambda r:r.pop('owned_remaining'),lambda r:r['child'].__setitem__('exit_code',1),
            lambda r:r.__setitem__('formal_Goodput',True),lambda r:r['devices_after']['processes']['7'].__setitem__('idle',False),
            lambda r:r['devices_after']['processes']['2'].__setitem__('output','running process'),
            lambda r:r['post_verifier'].__setitem__('exit_code',1),lambda r:r['devices_before']['hardware'].__setitem__('output','| 2 910B2 |')):
            r=complete_receipt();mutate(r)
            with self.assertRaises(RuntimeError):a.terminal_receipt(r,'HOST-only')

    def test_interpreter_ABI_is_explicit_and_no_silent_hash_normalization(self):
        with patch.object(a.sys,'version_info',(3,12,0)),patch.object(a,'read',side_effect=AssertionError('must refuse before read')):
            with self.assertRaisesRegex(RuntimeError,'Python3.11 AST ABI'):a.run('HOST-only')


class ArchiveTests(unittest.TestCase):
    def archive(self,path,rows):
        with tarfile.open(path,'w:gz') as t:
            for name,raw in rows:
                member=tarfile.TarInfo(name);member.size=len(raw);t.addfile(member,io.BytesIO(raw))

    def test_all_bytes_and_unique_flat_members_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.tar.gz';self.archive(p,[('x.py',b'actual')])
            expected={'/HOST/x.py':hashlib.sha256(b'actual').hexdigest()}
            self.assertTrue(a.archive(p,expected)['all_member_SHAs_matched'])
            for rows in ([('x.py',b'changed')],[('x.py',b'actual'),('x.py',b'actual')],[('../x.py',b'actual')]):
                self.archive(p,rows)
                with self.assertRaises(RuntimeError):a.archive(p,expected)


class FusionTests(unittest.TestCase):
    def actual(self):return a.BASE/'tree-packed-eager01-cost-v102/fusion_result.json'

    def test_actual_three_top_pairs_remain_three_and_nested_structure_strict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fusion_result.json';p.write_bytes(self.actual().read_bytes())
            observed=a.fusion(p,0.,time.time()+1)
            self.assertEqual((observed['top_level_object_pair_records'],observed['unique_session_graph_names']),(3,1))
            self.assertFalse(observed['writer_identity_inferred']);self.assertFalse(observed['source_input'])
            self.assertEqual(observed['sha256'],a.sha(self.actual()))
            for raw in ('{"code":"exec"}','{"session_and_graph_id_0_0":{"graph_fusion":{"unknown":{}}}}',
                '{"session_and_graph_id_0_0":{"graph_fusion":{"TransdataCastFusionPass":{"effect_times":true,"match_times":0}}}}',
                '{"session_and_graph_id_0_0":{"graph_fusion":{"TransdataCastFusionPass":{"effect_times":0,"effect_times":1,"match_times":0}}}}'):
                p.write_text(raw)
                with self.assertRaises(RuntimeError):a.fusion(p,0.,time.time()+1)

    def test_runtime_output_never_exempts_a_bound_static_name(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=p/'local-receipt-HOST.json';r.write_text('{}');f=p/'fusion_result.json';f.write_text('{"static":"bound"}')
            with patch.object(a,'WORK',p):
                self.assertEqual(a.runtime_outputs(r,dict(started_UTC='2020-01-01T00:00:00+00:00'),{str(f):a.sha(f)}),[])
                with self.assertRaises(RuntimeError):a.runtime_outputs(r,dict(started_UTC='2020-01-01T00:00:00+00:00'),{})

    def test_unknown_code_file_and_runtime_mtime_outside_interval_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=p/'local-receipt-HOST.json';r.write_text('{}');(p/'surprise.so').write_bytes(b'HOST')
            with patch.object(a,'WORK',p):
                with self.assertRaisesRegex(RuntimeError,'unbound'):a.runtime_outputs(r,dict(started_UTC='2020-01-01T00:00:00+00:00'),{})
            f=p/'fusion_result.json';f.write_bytes(self.actual().read_bytes())
            with self.assertRaisesRegex(RuntimeError,'actual run interval'):a.fusion(f,0.,1.)


class ClockTests(unittest.TestCase):
    def test_full463_native_stream_denominators_have_distinct_attainment(self):
        value=a.recompute_clocks(trace());m=value['metrics']
        self.assertEqual(m['legacy_Native_paper']['attained'],367)
        self.assertEqual(m['existing_stream_paper']['attained'],463)
        self.assertEqual(m['legacy_Native_paper']['SLO_Goodput_tokens_per_second'],367*256/50.)
        self.assertEqual(value['Native_inner_elapsed_seconds'],49.)
        self.assertEqual(value['host_minus_Native_inner_seconds'],1.)
        self.assertFalse(value['independent_Native_clock_reconstruction'])

    def test_output_shortening_finish_duplicate_wrong_owner_and_missing_field_rejected(self):
        for mutate in (lambda r:r['generate_results'][0]['completion_token_ids'].pop(),
            lambda r:r['stream_events'][-1].__setitem__('finished',False),
            lambda r:r['stream_events'][0].__setitem__('request_id','foreign'),
            lambda r:r['generate_results'][0].pop('observed_tpot_ms')):
            row=trace();mutate(row)
            with self.assertRaises((RuntimeError,KeyError)):a.recompute_clocks(row)

    def test_native_inner_clock_cannot_be_silently_substituted_for_host_wall(self):
        row=trace();row['generate_results'][0]['elapsed_seconds']=48.
        with self.assertRaisesRegex(RuntimeError,'Native inner elapsed'):a.recompute_clocks(row)

    def test_exact_SLO_threshold_equality_is_retained(self):
        value=a.recompute_clocks(trace())
        self.assertEqual(value['metrics']['legacy_Native_paper']['byclass']['tight']['attained'],271)


class FrozenSourceTests(unittest.TestCase):
    def test_actual_CPU_freeze_and_two_runtime_shells_are_pinned(self):
        self.assertEqual(a.sha(a.WORK/'PREPARED_SOURCE_HASHES_V103.json'),a.FREEZE_SHA)
        self.assertEqual(a.sha(a.WORK/'CPU_RESULT_V103.json'),a.CPU_SHA)
        self.assertEqual(a.sha(a.PREAUDIT),a.PREAUDIT_SHA)
        pre=a.read(a.PREAUDIT);f=a.read(a.WORK/'PREPARED_SOURCE_HASHES_V103.json');cpu=a.read(a.WORK/'CPU_RESULT_V103.json')
        self.assertEqual(cpu['input_hashes'],cpu['input_hashes_after']);self.assertEqual(cpu['input_hashes'],f['files'])
        self.assertEqual(cpu['external_input_hashes'],f['external_files']);self.assertEqual(cpu['tests_run'],32)
        self.assertEqual(set(pre['actual_sourced_runtime_shells_before']),set(a.SHELLS))
        for path,wanted in pre['actual_sourced_runtime_shells_before'].items():self.assertEqual(a.sha(path),wanted)

    def test_stdlib_readonly_auditor_origin_and_no_torch_import(self):
        module=a.host_module('audit_caps_v103')
        self.assertEqual(Path(module.__file__).resolve(),a.WORK/'audit_caps_v103.py')
        self.assertNotIn('torch',sys.modules)

    def test_source_derived_startup_family_counts_are_not_Native_map_counts(self):
        draft=a.expected_native_startup_keys(0);target=a.expected_native_startup_keys(1)
        self.assertEqual({k:len(v) for k,v in draft.items()},dict(entries=0,draft_entries=60,target_entries=54))
        self.assertEqual({k:len(v) for k,v in target.items()},dict(entries=0,draft_entries=0,target_entries=245))
        self.assertEqual(sum(map(len,draft.values())),86+28)

    def test_wrong_or_inherited_CPU_identity_rejected_before_raw_reads(self):
        with patch.object(a,'sha',return_value='changed'):
            with self.assertRaisesRegex(RuntimeError,'reviewed freeze/CPU/prelaunch'):a.source_closure({})

    def test_all_device_or_process_launch_apis_are_absent(self):
        import ast
        tree=ast.parse(Path(a.__file__).read_text())
        modules=[alias.name for n in ast.walk(tree) if isinstance(n,ast.Import) for alias in n.names]
        self.assertFalse(set(modules)&{'torch','torch_npu','subprocess','os'})
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr in ('sleep','Popen','exec','system') for n in ast.walk(tree)))


class PerformanceTests(unittest.TestCase):
    def full(self):
        row=trace();row.update(case='active32-cap3',active_limit=32,cap=3,role='full463',scope_count=463,
            tree_runtime_linear_control=True,genuine_branched_tree_target_qualified=False)
        layout=a.host_module('target_layout');m,s,dummy=layout.envelope(16,64,'causal')
        row['normal_target_calls']=[dict(N=16,budgets=[3]*16,real_query_count=64,family='causal',
            physical_M=m,physical_S=s,physical_query_partition=[4]*16+dummy)]
        row['phase_seconds']=dict(draft=2.,target=3.,exchange=1.,verify=0.,broadcast=1.,state_update=1.)
        row['decode_counters']={key:0 for key in ('spec_rhythm_tree_rounds','spec_rhythm_tree_nodes',
            'spec_rhythm_tree_accepted_nodes','spec_rhythm_verified_tokens','spec_rhythm_tree_eager_promoted',
            'spec_rhythm_tree_eager_invalidated','spec_rhythm_target_causal_tree_rounds','spec_rhythm_target_full_tree_rounds')}
        row['allocator']=dict(referenced_page_highwater=100,pool_blocks=544)
        shared=a.host_module('stream_metrics_v103').summarize(row)
        traces=[]
        for rank in range(4):
            value=copy.deepcopy(row);value['rank']=rank
            if rank!=1:value['generate_results']=None;value['stream_events']=[];value['native_elapsed_seconds']=None
            traces.append(value)
        return traces,shared

    def test_geometry_and_rank_phase_intervals_are_kept_without_independent_trials(self):
        traces,shared=self.full();result=a.performance_case(traces,shared)
        self.assertEqual(result['independent_repeated_trials'],1)
        self.assertEqual(result['synchronized_rank_observations'],4)
        self.assertEqual(result['normal_target']['row_verifications'],16)
        self.assertTrue(result['phase_rank_maxima_are_not_added'])
        self.assertFalse(result['pure_CPU_or_NPU_decomposition_claimed'])

    def test_family_geometry_and_missing_eager_are_rejected(self):
        for mutate in (lambda rows:rows[1]['normal_target_calls'][0].__setitem__('physical_M',999),
            lambda rows:rows[1]['normal_target_calls'][0].__setitem__('family','full9'),
            lambda rows:rows[1]['decode_counters'].pop('spec_rhythm_tree_eager_promoted')):
            traces,shared=self.full();mutate(traces)
            with self.assertRaises((RuntimeError,KeyError)):a.performance_case(traces,shared)

    def test_cross_clock_arithmetic_mismatch_cannot_be_called_comparable(self):
        traces,shared=self.full();shared['metrics']['legacy_Native_paper']['SLO_Goodput_tokens_per_second']+=1
        with self.assertRaisesRegex(RuntimeError,'clock arithmetic'):a.performance_case(traces,shared)


if __name__=='__main__':unittest.main(verbosity=2)
