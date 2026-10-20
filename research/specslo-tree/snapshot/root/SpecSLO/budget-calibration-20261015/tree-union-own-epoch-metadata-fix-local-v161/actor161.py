"""Original P154 actor and P157 gate preview, using the corrected metadata gate."""
from collections import Counter
from base161 import Q,functions
from scope154 import require,case,contract,NATIVE_COUNTERS,FAILURES,digest
import draft161 as draft
import source154 as epoch
import protocol154 as protocol
import target154 as target
functions(Q/'actor154.py',globals(),changes=((
 "gates=stage('discovery/raw-original-finish-gates',lambda:gate_rows(source,outputs,accepted));box['gates']=gates",
 "gates=stage('discovery/raw-original-finish-gates',lambda:gate_rows(source,outputs,accepted));box['gates']=gates\n  report['actual_discovery_gate_preview']=gates",1),))
