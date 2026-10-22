"""Original455 source environment, with this diagnostic directory first."""
from pathlib import Path
import os,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-M160-qualified-profile-whole-model-v455'))
from host_env455 import source_environment as parent_environment
def source_environment():
 value=parent_environment();value['PYTHONPATH']=str(P)+':'+value['PYTHONPATH'];return value
def setup():
 value=source_environment();os.environ.update(value);sys.path[:0]=value['PYTHONPATH'].split(':');return value
