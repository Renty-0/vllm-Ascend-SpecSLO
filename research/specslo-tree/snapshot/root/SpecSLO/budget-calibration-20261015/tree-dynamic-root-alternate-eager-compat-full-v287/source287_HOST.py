"""Build the exact original adapter-composed LIVE source used by the runtime."""
import sys
from pathlib import Path
def source():
 b=Path(__file__).resolve().parent.parent
 for name in ('tree-high-normal-cap4-native-families-v148','tree-causal80-small-cohort-role-aware-audit-full-v209','tree-causal80-N9to15-allB4-empty-full-v197','tree-active-window32-16-local-v160','tree-static-q5-M120-SLO-first-full463-own-AR-v257','tree-static-q5-M120-full463-own-AR-v256'):
  sys.path.append(str(b/name))
 from live_decode_patch import build_live_runtime
 from window287 import adapt_source
 from empty287 import patch
 return patch(adapt_source(build_live_runtime(),48))
