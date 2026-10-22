"""Small new auditor boundaries. No actual run, Torch or source mutation."""
import copy
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
import analyze_v144_actual_variable_B4 as a

class AuditorHostTests(unittest.TestCase):
 def test_archive_exact_bytes_reject_changed_or_duplicate(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'source.py';source.write_text('unchanged original source\n')
   expected={str(source):a.sha(source)};path=root/'good.tar.gz'
   with tarfile.open(path,'w:gz') as tar:tar.add(source,arcname=str(source).lstrip('/'),recursive=False)
   self.assertEqual(a.check_archive(path,expected)['members'],1)
   source.write_text('modified source\n')
   wrong=root/'wrong.tar.gz'
   with tarfile.open(wrong,'w:gz') as tar:tar.add(source,arcname=str(source).lstrip('/'),recursive=False)
   with self.assertRaisesRegex(ValueError,'member bytes'):a.check_archive(wrong,expected)
   duplicate=root/'duplicate.tar.gz'
   with tarfile.open(duplicate,'w:gz') as tar:
    for _ in range(2):tar.add(source,arcname=str(source).lstrip('/'),recursive=False)
   with self.assertRaisesRegex(ValueError,'duplicate'):a.check_archive(duplicate,expected)

 def test_full8_encoded_confidence_reconstruction_and_tamper(self):
  q=[750000,600000,550000,500000,250000,200000,150000,100000]
  rows=[[100+i*4+j for j in range(4)]+[1000+i*4+j for j in range(4)] for i in range(32)]
  encoded=[[[row[j]*1_000_001+q[j],row[4+j]*1_000_001+q[4+j]] for j in range(4)] for row in rows]
  proof=dict(raw_encoded_shape=[32,4,2],raw_pool_nodes=256,actual_draft_model_stages=4,selected_C=96,
   variable_num_draft_tokens=list(a.VECTORS[0]),raw_pool=rows,node_confidences=[[a.f32(v/1_000_000) for v in q] for _ in rows],
   mean_confidence=[a.f32(sum(v/1_000_000 for v in q)/8)]*32,encoded_sha256=a.digest(encoded))
  self.assertEqual(a.draft_ABI(proof,a.VECTORS[0]),encoded)
  bad=copy.deepcopy(proof);bad['raw_pool'][0][0]+=1
  with self.assertRaisesRegex(ValueError,'encoded SHA'):a.draft_ABI(bad,a.VECTORS[0])
  bad=copy.deepcopy(proof);bad['mean_confidence'][0]=0.0
  with self.assertRaisesRegex(ValueError,'decoded mean8'):a.draft_ABI(bad,a.VECTORS[0])
  bad=copy.deepcopy(proof);bad['node_confidences'][0][0]=float('nan')
  with self.assertRaisesRegex(ValueError,'finite float32'):a.draft_ABI(bad,a.VECTORS[0])

 def test_actual_chain_bonus_and_prefix_rejection_semantics(self):
  budgets=a.VECTORS[0];rows=[[100+i*8+j for j in range(b)] for i,b in enumerate(budgets)]
  query=[v for i,row in enumerate(rows) for v in row+[5000+i]]
  emitted,paths=a.causal_verdict(rows,query,budgets)
  self.assertEqual(emitted[0],rows[0]+[5000]);self.assertEqual(paths[0],[0,1,2,3])
  self.assertEqual(paths[-1],[0,-1,-1,-1]);self.assertEqual(emitted[-1],rows[-1]+[5031])
  query[2]=7777;emitted,paths=a.causal_verdict(rows,query,budgets)
  self.assertEqual(paths[0],[0,1,-1,-1]);self.assertEqual(emitted[0],rows[0][:2]+[7777])
  with self.assertRaisesRegex(ValueError,'real Q'):a.causal_verdict(rows,query[:-1],budgets)

 def test_local_config_scope_and_common_stage_order_reject_old_full(self):
  cfg=dict(max_num_seqs=32,max_num_queued_seqs=64,kvcache_block_size=128,num_kvcache_blocks=544,max_model_len=4096,gamma=8,
   draft_tp_size=1,target_tp_size=3,draft_model='/root/data/shared-models/Qwen3-0.6B',target_model='/root/data/shared-models/Qwen3-32B',
   enable_mc2=True,precompile_decode_graphs=False,precompile_serial_draft_graphs=False,enable_prefix_caching=False,
   spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4,spec_rhythm_max_eager_tokens=3,spec_rhythm_cpu_verdict=True,
   draft_use_paged_attention=True,target_use_paged_attention=True)
  a.actual_config(cfg,0)
  bad=dict(cfg,max_num_seqs=64,max_num_queued_seqs=463)
  with self.assertRaisesRegex(ValueError,'local32'):a.actual_config(bad,0)
  bad=dict(cfg,kvcache_block_size=16)
  with self.assertRaisesRegex(ValueError,'block128'):a.actual_config(bad,0)
  labels=a.expected_stages();self.assertEqual(len(labels),1853)
  restored=labels.index('pass0/case0/cycle0/target-greedy/restored/drain')
  hidden=labels.index('pass0/case0/cycle0/target-hidden/first/Native')
  self.assertLess(restored,hidden)
  self.assertLess(labels.index('pass0/case0/cycle0/target-reference/query-and-finite'),labels.index('pass0/case0/cycle0/target-reference/original-same-M-head'))
  events=[dict(sequence=i,label=label,ranks=[dict(rank=r,sequence=i,label=label,error=None) for r in range(4)]) for i,label in enumerate(labels)]
  self.assertEqual(a.boundary_record(events),1853)
  events[restored]['ranks'][2]['error']='a local failure must stop the next TP lane'
  with self.assertRaisesRegex(ValueError,'every target/prefill TP lane'):a.boundary_record(events)

if __name__=='__main__':unittest.main()
