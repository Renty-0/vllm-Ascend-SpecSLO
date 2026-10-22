"""Original459 source-auth environment plus the new full integration sources."""
from pathlib import Path
import os,sys
P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'tree-adaptive-own-geometry-node-math-v459'))
from host_env459 import source_environment as original
def source_environment():
    value=original();value['PYTHONPATH']=':'.join((str(P),str(B/'tree-observed-normal-history-audit-host-v439'),value['PYTHONPATH']));return value
def setup():
    value=source_environment();os.environ.update(value);sys.path[:0]=value['PYTHONPATH'].split(':');return value
