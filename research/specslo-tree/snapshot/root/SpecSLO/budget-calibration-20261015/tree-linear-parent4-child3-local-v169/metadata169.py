"""Original ten-field own-epoch PA checks for genuine P4 frontier at p+4."""
from copy import deepcopy
from base169 import V161,functions,require
FIELDS=('inputs','positions','slot_mapping','context_lens','block_tables','host_Q','host_KV',
 'use_fused_infer_attention','has_attention_mask','has_request_tables')
functions(V161/'metadata161.py',globals(),names=('values','snapshot','observed_snapshot','validate_own','semantic','capture_actual','compare_fresh'))
def owner_geometry(case):
 require(case.normal==case.eager==16 and case.parent_B==4 and case.repair_count in (0,1,16) and case.union_M in (16,20,32),
  'new parent4 finite union scope only')
 return list(range(16,32))+list(case.repair_owners),[case.prefix+4]*16+[case.prefix]*case.repair_count
def geometry(case,f):
 owners,positions=owner_geometry(case)
 tokens=[int(f['sources'][i][1][3]) for i in range(16,32)]+[int(f['catchups'][i][0]) for i in case.repair_owners]
 return owners,positions,tokens

_original_capture_actual=capture_actual
_original_compare_fresh=compare_fresh

def capture_actual(e,case,f,seen,PA):
 proof=_original_capture_actual(e,case,f,seen,PA)
 proof['authoritative_owner_tables']={str(i):list(e.cache_allocation.block_tables[i]) for i in proof['owners']}
 proof['block_size']=e.config.kvcache_block_size
 return proof

def compare_fresh(e,case,actual_proof,reference_call):
 result=_original_compare_fresh(e,case,actual_proof,reference_call)
 proof=result['reference'];proof['authoritative_owner_tables']={str(i):list(e.cache_allocation.block_tables[i]) for i in proof['owners']}
 proof['block_size']=e.config.kvcache_block_size
 return result
