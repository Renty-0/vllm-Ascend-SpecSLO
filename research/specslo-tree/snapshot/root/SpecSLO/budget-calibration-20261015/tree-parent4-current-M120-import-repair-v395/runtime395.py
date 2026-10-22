"""Launch the exact391 numerical runtime with repaired dependency lookup."""
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
PREVIOUS = P.parent / 'tree-parent4-current-M120-target-protocol-v391'
sys.path.insert(0, str(PREVIOUS))


def main():
    from check395 import ready
    ready()
    from runtime391 import build_runtime
    build_runtime()['main']()


if __name__ == '__main__':
    main()
