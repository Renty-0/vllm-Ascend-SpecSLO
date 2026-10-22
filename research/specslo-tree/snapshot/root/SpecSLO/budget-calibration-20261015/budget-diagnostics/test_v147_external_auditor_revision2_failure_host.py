"""One changed failure-schema regression; no parent math/source suite reruns."""
import ast,hashlib,importlib.util,io,json,sys,tempfile,time,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
subject_path=D/'analyze_v147_actual_large_mixed_revision2.py'
spec=importlib.util.spec_from_file_location('_v147_external_failure_revision2',subject_path);subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)
class FailureSchema(unittest.TestCase):
 def test_original_inner_outer_missing_and_damaged_checkpoint(self):
  old=ast.parse((D/'analyze_v147_actual_large_mixed.py').read_text());new=ast.parse(subject_path.read_text())
  def without(tree):
   tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='first_failure'];return ast.dump(tree,include_attributes=False)
  self.assertEqual(without(old),without(new))
  with tempfile.TemporaryDirectory(prefix='v147-revision2-failure-',dir=D) as name:
   directory=Path(name);receipt=dict(status='failed',errors=['actual worker failed'])
   empty=subject.first_failure(directory,receipt);self.assertEqual(len(empty['missing_final_reports']),4)
   self.assertTrue(empty['failure_stage_unknown_without_reachable_checkpoint'])
   trace="  raise BoundaryRejected('All-rank original boundary rejected '+name+': '+repr(rows))\nnative_boundaries.BoundaryRejected: All-rank original boundary rejected mixed-original-prefix-after: [{error}]"
   (directory/'rank0-last-error.json').write_text(json.dumps(dict(error=trace,boundary_events=[])))
   event=dict(sequence=5,label='pass0/target/final-gate',ranks=[dict(rank=0,error='real error')])
   (directory/'rank1-last-error.json').write_text(json.dumps(dict(error='RuntimeError: actual gate',boundary_events=[event])))
   (directory/'rank2-last-error.json').write_text('{bad json')
   (directory/'rank3-last-error.json').write_text('[]')
   result=subject.first_failure(directory,receipt)
   self.assertEqual(result['worker_error_evidence'][0]['first_original_prompt_boundary_failure'],'mixed-original-prefix-after')
   self.assertEqual(result['first_common_failed_events'][0]['first_common_failed_event'],event)
   self.assertIn('checkpoint_read_error',result['worker_error_evidence'][2]);self.assertIn('checkpoint_schema_error',result['worker_error_evidence'][3])
   self.assertFalse(result['failure_stage_unknown_without_reachable_checkpoint']);self.assertFalse(result['component_qualification'])
if __name__=='__main__':
 started=time.perf_counter();stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FailureSchema))
 stamp=time.time_ns();log=D/f'V147_EXTERNAL_AUDITOR_REVISION2_FAILURE_HOST_{stamp}.log';log.write_text(stream.getvalue())
 row=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-started,source_sha256=subject.sha(subject_path),test_source_sha256=subject.sha(__file__),log=dict(path=str(log),sha256=subject.sha(log)),only_changed_function='first_failure',other_module_AST_identical_to_4fabb=True,original_four_tests_rerun=False,parent_suites_rerun=False,Torch_imported='torch' in sys.modules,NPU_queried=False,model_called=False,actual_runtime_or_current_V148_raw_read=False)
 output=D/f'V147_EXTERNAL_AUDITOR_REVISION2_FAILURE_HOST_{stamp}.json';output.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(row,result_path=str(output))))
 if not result.wasSuccessful():raise SystemExit(1)
