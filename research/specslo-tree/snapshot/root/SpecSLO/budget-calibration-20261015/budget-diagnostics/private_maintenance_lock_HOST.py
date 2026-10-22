"""Storage maintenance and NPU execution share one exclusive filesystem lock."""
from contextlib import contextmanager
from pathlib import Path
import fcntl

LOCK = Path('/root/SpecSLO/budget-calibration-20261015/.ROOT_NPU_EXECUTION.lock')


@contextmanager
def exclusive_private_maintenance():
    with LOCK.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('NPU execution owns the shared lock; storage maintenance must wait without changing files') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
