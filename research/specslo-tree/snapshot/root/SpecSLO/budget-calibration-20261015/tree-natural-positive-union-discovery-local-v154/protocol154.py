"""Original typed protocol; one all4 hook BEFORE genuine parent commit.

The hook can independently qualify the new union after seeing actual verdicts.
It cannot change output/accepted rows, any ticket, or the original finish guard.
The old generic run() is deliberately not imported: the discovery actor owns
the explicit parent/conditional-normal-child/next phase order.
"""
from collections import Counter
from scope154 import P151,require
from reuse154 import functions
import source154 as epoch
import target154 as target
functions(P151/'protocol151.py',globals(),names=('setup','exchange','check_commit','commit'),changes=(
 ("outputs,accepted,_=stage('original-verdict-broadcast',lambda:e._broadcast_spec_rhythm_tree_verdict(observed,plans,0.))",
  "outputs,accepted,_=stage('original-verdict-broadcast',lambda:e._broadcast_spec_rhythm_tree_verdict(observed,plans,0.))\n if source.get('_v154_before_parent_commit') is not None:\n  source['_v154_before_parent_commit'](outputs,accepted)",1),))
