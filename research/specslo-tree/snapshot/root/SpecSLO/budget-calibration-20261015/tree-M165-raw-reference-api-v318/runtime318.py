"""Original317 model qualification with only the native raw reference fixed."""
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
ORIGINAL = P.parent/'tree-M165-N32-independent-target-math-cost-v317'
sys.path.insert(0,str(ORIGINAL))
import runtime317 as original

original_worker = original.worker

def worker318(*args, **kwargs):
    # Spawned roles install the supported native reference before any tests.
    # Candidate write313, all546 neural fixtures, matchers and costs are intact.
    sys.path.insert(0,str(P))
    import raw_controls317
    from raw_controls318 import run
    raw_controls317.run = run
    return original_worker(*args, **kwargs)

original.worker = worker318
if __name__ == '__main__':
    original.main()
