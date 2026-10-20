"""Original P152 bounded worker shell in a fresh P154 namespace."""
import argparse,gzip,json,os,re,sys,traceback
from functools import partial
from pathlib import Path
from scope154 import P,P152,ACTION,require
from reuse154 import functions
functions(P152/'runtime152.py',globals(),changes=(
 ('from check152 import require_ready','from check154 import require_ready',2),
 ('from actor152 import run_actor,native as snapshot','from actor154 import run_actor,native as snapshot',1),
 ('from audit152 import audit','from audit154 import audit',1),
 ('from scope152 import make_config','from scope154 import make_config',1),
 ('dict(cases=4,complete_zeros=2,timings=0)','dict(cases=1,complete_zeros=0,timings=0)',2),))
if __name__=='__main__':main()
