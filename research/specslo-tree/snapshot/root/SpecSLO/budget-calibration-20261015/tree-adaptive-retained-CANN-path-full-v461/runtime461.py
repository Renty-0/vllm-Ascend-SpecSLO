"""Keep every460 numerical callback and full lifecycle byte unchanged."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'tree-adaptive-own-history-full-v460'))
from check461 import ready
import runtime460
def main():
    ready();runtime460.build_runtime()['main']()
if __name__=='__main__':main()
