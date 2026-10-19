"""Fresh capacity329 input binding plus immutable326 model source closure."""
import json
import os
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-M160-S32-N32-target-math-cost-v326'))
import check326 as original

def ready(expected=None):
    original.ready('1c8878b7490ed85684aea9686a25652b2488f30074769b0568243e5c476fbc43')
    expected=expected or os.environ.get('SPECSLO_V329_ROOT_FREEZE_SHA')
    original.require(isinstance(expected,str)and len(expected)==64 and original.sha(P/'FREEZE329.json')==expected,'explicit329 source pin')
    frozen=original.read(P/'FREEZE329.json')
    for path,wanted in frozen['files'].items():
        original.require(original.sha(path)==wanted,'unchanged329 source '+path)
    return frozen

if __name__=='__main__':ready()
