"""Same P157 actual discovery/protocol/next action with new exact metadata gate."""
import argparse,gzip,json,os,re,sys,traceback
from functools import partial
from pathlib import Path
from base161 import P,Q,B,ACTION,require,functions,bind_source_epoch,sha
functions(B/'tree-union-shared4-protocol-local-v152/runtime152.py',globals(),names=('save','worker','main'),changes=(
 ('from check152 import require_ready','from check161 import require_ready',2),
 ('from actor152 import run_actor,native as snapshot','from actor161 import run_actor,native as snapshot',1),
 ('from audit152 import audit','from audit154 import audit',1),
 ('from scope152 import make_config','from scope154 import make_config',1),
 ('dict(cases=4,complete_zeros=2,timings=0)','dict(cases=1,complete_zeros=0,timings=0)',2),
 ("'V152/'+n","'V161/'+n",1),))
def install(native,graph):
 bind_source_epoch()
 from runtime154 import install as original_install
 origins=original_install(native,graph)
 origins['source_epoch']=dict(path=str(P/'source_epoch.py'),sha256=sha(P/'source_epoch.py'),current_reader_identity='source154.read',only_compatibility_import=True)
 origins['draft161']=dict(path=str(P/'draft161.py'),sha256=sha(P/'draft161.py'),metadata_gate_only=True)
 return origins
if __name__=='__main__':main()
