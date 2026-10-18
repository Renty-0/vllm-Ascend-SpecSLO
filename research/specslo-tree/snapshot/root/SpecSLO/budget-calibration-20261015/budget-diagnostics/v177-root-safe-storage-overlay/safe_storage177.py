"""Storage accounting tolerates only vanished atomic-output temporary files."""
import errno
import os
from pathlib import Path


def bytes_under(path, *, live_roots, require):
    path = Path(path)
    if not path.exists():
        return 0
    require(path.resolve() == path and not path.is_symlink(),
            'private storage root no alias')
    roots = tuple(Path(root) for root in live_roots)
    total = 0
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in (*dirs, *files):
            require(not (Path(base) / name).is_symlink(),
                    'private storage contains unexpected link')
        for name in files:
            item = Path(base) / name
            try:
                total += item.stat().st_size
            except FileNotFoundError as error:
                # os.walk can list an atomic writer's temporary path immediately
                # before os.replace removes that path.  Immutable inputs and
                # non-temporary artifacts must still raise their original error.
                if error.errno != errno.ENOENT or item.suffix != '.tmp' or not any(
                        item.is_relative_to(root) for root in roots):
                    raise
    return total
