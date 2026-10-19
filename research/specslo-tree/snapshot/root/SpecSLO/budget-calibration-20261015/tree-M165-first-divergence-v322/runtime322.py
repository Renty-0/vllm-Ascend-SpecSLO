"""Unchanged546-case path plus firstcase eager tracing; not a timing run."""
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
B=P.parent
sys.path[:0]=[str(B/'tree-M165-raw-reference-api-v318'),str(B/'tree-M165-N32-independent-target-math-cost-v317')]
import runtime318 as original
import probes317

original_probe=probes317.run

def observed_probe(engine,native,stage,output,fixture):
    from trace322 import install
    install(engine,output)
    return original_probe(engine,native,stage,output,fixture)

probes317.run=observed_probe
if __name__=='__main__':
    original.original.main()
