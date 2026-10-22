"""Original457 environment, plus this explicit own-physical node protocol."""
from pathlib import Path
import os,sys
P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'tree-M80-own-geometry-whole-model-v457'))
from host_env457 import source_environment as original
def source_environment():
    value=original()
    value['PYTHONPATH']=':'.join((str(P),str(B/'tree-adaptive-M80-M120-target-math-cost-v378'),str(B/'tree-adaptive-M80-M120-target-host-v377'),value['PYTHONPATH']))
    return value
def setup():
    value=source_environment();os.environ.update(value);sys.path[:0]=value['PYTHONPATH'].split(':');return value
