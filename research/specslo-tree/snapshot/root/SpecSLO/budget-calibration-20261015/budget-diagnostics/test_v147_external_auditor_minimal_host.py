"""Only four new schema/math regressions; no parent suite or actual runtime data."""
import copy,hashlib,importlib.util,io,json,struct,sys,tempfile,time,unittest
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_v147_external_host_subject',P/'analyze_v147_actual_large_mixed.py');subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
class Minimal(unittest.TestCase):
 def test_causal_B3_full_partial_correction_and_invalid_Q(self):
  helpers,_=subject.reused_helpers();rows=[[10,11,12]]*32
  emitted,accepted=helpers['causal_verdict'](rows,[10,11,99,77]+[55,11,12,88]+[10,11,12,88]*30,[3]*32)
  self.assertEqual(emitted[:3],[[10,11,99],[55],[10,11,12,88]])
  self.assertEqual(accepted[:3],[[0,1,-1,-1],[-1]*4,[0,1,2,-1]])
  with self.assertRaises(ValueError):helpers['causal_verdict'](rows,[0]*127,[3]*32)
 def test_exact_head_hash_diff_and_corruption(self):
  a=list(range(128));b=list(a);b[3]=99
  old=dict(source={'same':True},rows=[[1,2,3]]*32,selected=[[0,1,2]]*32,target=dict(normal_query_tokens=a))
  new=copy.deepcopy(old);new['target']['normal_query_tokens']=b
  h=lambda v:hashlib.sha256(struct.pack('<128q',*v)).hexdigest()
  head=dict(shape=[128],dtype='torch.int64',original197_sha256=h(a),large645_sha256=h(b),bit_exact=False,differing_elements=1,max_abs=96.,counterexample=[dict(flat_index=3,original197=3.,large645=99.)])
  tensor=dict(shape=[128,1,1],dtype='torch.bfloat16',original_matcher=True,bit_exact=True,differing_elements=0,max_abs=0.,original197_sha256='a'*64,large645_sha256='a'*64,counterexample=[])
  hidden=dict(tensor,shape=[128,5120])
  record=dict(selected_heads=head,exact_normal_heads=False,hidden=hidden,query_KV=[tensor]*128,all_normal_hidden_KV_original_matcher=True,cross_M_equivalent=False)
  result=subject.cross_head_observation(old,new,record)
  self.assertEqual(result['differing_head_tokens'],1);self.assertFalse(result['full_AR_losslessness_proven'])
  bad=copy.deepcopy(record);bad['selected_heads']['large645_sha256']='0'*64
  with self.assertRaises(ValueError):subject.cross_head_observation(old,new,bad)
 def test_native_counter_schema_arithmetic_and_false_zero(self):
  before=dict.fromkeys(subject.COUNTERS,0);after=dict(before);after['aclgraph_captures']=1
  raw=dict(before=before,after=after,delta=dict(after));self.assertEqual(subject.counter_delta(raw)['aclgraph_captures'],1)
  bad=copy.deepcopy(raw);bad['delta']['aclgraph_captures']=0
  with self.assertRaises(ValueError):subject.counter_delta(bad)
  bad=copy.deepcopy(raw);bad['after']['aclgraph_shape_fallbacks']=bad['delta']['aclgraph_shape_fallbacks']=1
  with self.assertRaises(ValueError):subject.counter_delta(bad)
 def test_missing_or_bad_checkpoint_is_incomplete_not_crash(self):
  with tempfile.TemporaryDirectory(prefix='v147-audit-host-',dir=P) as name:
   directory=Path(name);(directory/'rank0-last-error.json').write_text('{bad json')
   result=subject.first_failure(directory,dict(status='failed',errors=['actual worker stage failed']))
   self.assertEqual(len(result['missing_final_reports']),4);self.assertIn('checkpoint_read_error',result['worker_error_evidence'][0])
   self.assertFalse(result['component_qualification']);self.assertFalse(result['MC2_645_qualified'])
if __name__=='__main__':
 started=time.perf_counter();stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Minimal))
 stamp=time.time_ns();log=P/f'V147_EXTERNAL_AUDITOR_HOST_{stamp}.log';log.write_text(stream.getvalue())
 row=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-started,source_sha256=subject.sha(P/'analyze_v147_actual_large_mixed.py'),test_source_sha256=subject.sha(__file__),log=dict(path=str(log),sha256=subject.sha(log)),Torch_imported='torch' in sys.modules,NPU_queried=False,model_called=False,parent_suites_rerun=False,actual_runtime_or_current_V148_raw_read=False)
 output=P/f'V147_EXTERNAL_AUDITOR_HOST_{stamp}.json';output.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(row,result_path=str(output))))
 if not result.wasSuccessful():raise SystemExit(1)
