"""Restore missing original aliases to unchanged retained CANN/ATB files."""
import hashlib
import json
import os
from pathlib import Path

LOCK = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json')
ALIASES = {
    Path('/usr/local/Ascend/cann-9.0.0'): Path('/root/data/specslo-cann-runtime/cann-9.0.0'),
    Path('/opt/specslo-cann/nnal'): Path('/root/data/specslo-cann-runtime/nnal'),
}

def restore():
    expected=json.loads(LOCK.read_text())['external_files_sha256']
    verified=[]
    for alias,actual in ALIASES.items():
        if os.path.lexists(alias) and (not alias.is_symlink() or alias.resolve()!=actual):
            raise RuntimeError('Existing runtime path conflicts; will not overwrite: '+str(alias))
        for filename,wanted in expected.items():
            old=Path(filename)
            if old.is_relative_to(alias):
                retained=actual/old.relative_to(alias)
                if hashlib.sha256(retained.read_bytes()).hexdigest()!=wanted:
                    raise RuntimeError('Retained runtime bytes changed: '+str(retained))
                verified.append(filename)
    if len(verified)!=13:
        raise RuntimeError('Original exact13 external CANN/ATB bindings required')
    created=[]
    for alias,actual in ALIASES.items():
        if not os.path.lexists(alias):
            alias.parent.mkdir(parents=True,exist_ok=True)
            alias.symlink_to(actual,target_is_directory=True)
            created.append(str(alias))
    return dict(verified_original_files=13,created_missing_aliases=created,library_bytes_modified=False,
                installed_packages_modified=False)
