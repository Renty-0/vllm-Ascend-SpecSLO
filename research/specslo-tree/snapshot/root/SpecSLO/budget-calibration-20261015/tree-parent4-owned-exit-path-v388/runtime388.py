"""Add the existing owned shutdown helper path before original386 spawns workers."""
from pathlib import Path
import sys

B=Path(__file__).parent.parent
sys.path.insert(0,str(B/'tree-contiguous-branch-tail-final-metadata-math-v234'))
sys.path.insert(0,str(B/'tree-parent4-child4-original-bounded-v386'))
import runtime386


if __name__=='__main__':runtime386.build_runtime()['main']()
