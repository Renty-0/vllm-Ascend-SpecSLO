"""Declare the measured160 profile before the actual Native constructor."""
from dataclasses import replace
from pathlib import Path
import hashlib
from startup_config_v132 import make_config as original

P=Path(__file__).resolve().parent
EXPECTED_PROFILE='cdef680f8ae1231d143d05162715564dc8c1b248b41eb961239c7d34df37635a'


def make_config():
    path=P/'mc2-profile-with-qualified160.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_PROFILE:
        raise RuntimeError('ROOT455 exact qualified160 profile bytes')
    before=original()
    # PEARLConfig is frozen. All ordinary prefill, queue, B, dtype and physical
    # mapping settings are kept; only a qualified profile path is changed.
    after=replace(before,mc2_profile=str(path))
    return after
