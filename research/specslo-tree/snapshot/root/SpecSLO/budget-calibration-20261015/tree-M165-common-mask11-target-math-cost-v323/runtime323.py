"""Original four-role protocol with explicit common-mask323 probe sources."""
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
B=P.parent
sys.path[:0]=[str(P),str(B/'tree-M165-raw-reference-api-v318'),str(B/'tree-M165-N32-independent-target-math-cost-v317')]
import runtime318 as original
import probes317
from probes323 import run

probes317.run=run
if __name__=='__main__':
    original.original.main()
