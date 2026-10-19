"""Only actual padded projections may select165 chain admission, no deferral."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-M160-actual165-MC2-bridge-host-v330'))
from bridge334 import projected_rows,install
q32=NS(tree_attention=True,actual_seq_lengths_q=tuple(range(5,161,5)))
q33=NS(tree_attention=True,actual_seq_lengths_q=tuple(range(5,166,5)))
ordinary=NS(tree_attention=False,actual_seq_lengths_q=tuple(range(5,161,5)))
cases=[]
for rows,meta,want in [(160,q32,165),(160,q33,160),(160,ordinary,160),(160,None,160),(159,q32,159),(165,q32,165),(128,q32,128)]:
    assert projected_rows(NS(shape=(rows,5120)),meta)==want
    cases.append(dict(physical_rows=rows,tree=bool(meta and meta.tree_attention),chain_admission_rows=want))
rejected=[]
for label,chained,deferred in [('unqualified',set(),set()),('deferred',{165},{165})]:
    try:install(NS(_mc2_chained_row_counts=chained,_mc2_deferred_row_counts=deferred),None)
    except RuntimeError:rejected.append(label)
    else:raise AssertionError('unsupported chain scope accepted')
assert rejected==['unqualified','deferred']
report=dict(status='HOST_pass',cases=cases,rejected=rejected,
            projection_rows_not_FIA_or_input_rows_changed=True,deferred_read_done_enabled=False,
            no_model_or_NPU_run=True,Goodput=None)
(P/'HOST334_ROW_ADMISSION_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
