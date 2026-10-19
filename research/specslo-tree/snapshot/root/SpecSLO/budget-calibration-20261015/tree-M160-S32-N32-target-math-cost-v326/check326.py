"""Own M160 source pin plus immutable original317/218 source closure."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
B=P.parent
ORIGINAL=B/'tree-M165-N32-independent-target-math-cost-v317'
sys.path.insert(0,str(ORIGINAL))
import check317 as original
require=original.require
sha=original.sha
read=original.read

def ready(expected=None):
    original.ready('36cbac2bca38b3b69b2ab4ed04f984bb48782fd51d5bb682d04f3d98ba76405f')
    expected=expected or os.environ.get('SPECSLO_V326_ROOT_FREEZE_SHA')
    require(isinstance(expected,str)and len(expected)==64 and sha(P/'FREEZE326.json')==expected,'explicit ROOT326 source pin')
    frozen=read(P/'FREEZE326.json')
    for path,wanted in frozen['files'].items():
        require(sha(path)==wanted,'unchanged326 source '+path)
    return frozen

if __name__=='__main__':ready()
