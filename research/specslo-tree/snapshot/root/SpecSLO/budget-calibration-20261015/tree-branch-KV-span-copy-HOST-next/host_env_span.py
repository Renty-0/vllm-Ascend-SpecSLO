"""Current-card diagnostic environment; no device initialization on import."""
from pathlib import Path
import os
import sys

P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(B/'tree-M80-own-geometry-whole-model-v457'))
from host_env457 import source_environment as parent_environment


def source_environment():
    value = parent_environment()
    additions = [P, B/'tree-parallel-root-fork-overlap-cost-HOST-next', B/'tree-parallel-root-fork-coupled-HOST-next', B/'tree-parallel-root-fork-target-HOST-next', B/'tree-parallel-root-fork-mode-init-HOST-v2', B/'tree-adaptive-own-geometry-node-math-v459', B/'tree-adaptive-own-history-full-v460', B/'tree-genuine-alternate-draft-observed-component-v277',
                 B/'tree-private-alternate-draft-tail-host-v271']
    value['PYTHONPATH'] = ':'.join(map(str, additions))+':'+value['PYTHONPATH']
    return value


def setup():
    value = source_environment()
    os.environ.update(value)
    sys.path[:0] = value['PYTHONPATH'].split(':')
    return value
