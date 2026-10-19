"""Apply the pinned contract repair, then run immutable component374."""
import sys
from pathlib import Path
import runpy
B=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(B/'tree-fastv5-frontier-boolean-guard-v375'),str(B/'tree-fastv5-eager-frontier-chain-host-v373')]
from boolean375 import install
install()
runpy.run_path(str(B/'tree-fastv5-eager-frontier-chain-device-v374/probe374.py'),run_name='__main__')
