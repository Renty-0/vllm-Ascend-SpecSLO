"""P152 unchanged math/history; exact metadata validated in its own allocation."""
import copy,time
from scope154 import P152,require,digest,bucket
from base161 import functions
from provider154 import source_contexts
from finite151 import require_finite
import source154 as epoch
from metadata161 import capture_actual,compare_fresh

ACTUAL_OLD="return dict(output=out,observed=seen,KV=epoch.logical_KV(e,[128]*64,counts(case)),ledger=epoch.ledger(e),prefix=protected,TTU=copy.deepcopy(e._v152_last_mixed_TTU))"
ACTUAL_NEW="metadata_proof=capture_actual(e,case,f,seen,PA)\n  return dict(output=out,observed=seen,KV=epoch.logical_KV(e,[128]*64,counts(case)),ledger=epoch.ledger(e),prefix=protected,TTU=copy.deepcopy(e._v152_last_mixed_TTU),metadata_proof=metadata_proof)"
COMPARE_OLD="proof=compare(e,case,a['output'],a['observed'],a['KV'],a['ledger'],ref)"
COMPARE_NEW="proof=compare(e,case,a['output'],a['observed'],a['KV'],a['ledger'],ref,actual_metadata=a['metadata_proof'])"
SIGNATURE_OLD='def compare(e,case,actual,observed,actual_KV,actual_ledger,reference):'
SIGNATURE_NEW='def compare(e,case,actual,observed,actual_KV,actual_ledger,reference,*,actual_metadata):'
META_OLD="""  require(torch.equal(meta['inputs'],x) and torch.equal(meta['positions'],p)
   and torch.equal(meta['slot_mapping'],m.slot_mapping) and torch.equal(meta['context_lens'],m.context_lens)
   and torch.equal(meta['block_tables'],m.block_tables) and meta['host_Q']==tuple(m.actual_seq_lengths_q)
   and meta['host_KV']==tuple(m.sequence_lens),'independent original union PA metadata actual owner/slot/context/padding')"""
META_NEW="  metadata_gate=compare_fresh(e,case,actual_metadata,reference['union_call'])"
functions(P152/'draft152.py',globals(),changes=((ACTUAL_OLD,ACTUAL_NEW,1),(COMPARE_OLD,COMPARE_NEW,1),
 (SIGNATURE_OLD,SIGNATURE_NEW,1),(META_OLD,META_NEW,1)))
