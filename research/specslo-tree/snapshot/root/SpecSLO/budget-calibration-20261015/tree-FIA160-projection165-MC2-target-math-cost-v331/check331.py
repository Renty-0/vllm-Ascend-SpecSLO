"""Exact parent329 and finished HOST330 plus own closure."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-M160-C128-capacity-binding-v329'))
from check329 import ready as parent_ready

def ready(expected=None):
    parent_ready('8ad8a253fbf54cb026b2a8498efc8d3b1a7b7a28e4d08f671d94f64b4e3fd949')
    expected=expected or os.environ.get('SPECSLO_V331_ROOT_FREEZE_SHA')
    manifest=P/'FREEZE331.json'
    if not expected or hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Explicit331 source pin required')
    frozen=json.loads(manifest.read_text())
    for path,sha in frozen['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise RuntimeError('Changed331 source '+path)
    return frozen

if __name__=='__main__':ready()
