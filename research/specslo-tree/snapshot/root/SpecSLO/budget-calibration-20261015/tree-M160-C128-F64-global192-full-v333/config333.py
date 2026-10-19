"""Original startup capacity retained; own LIVE policy switches after controls."""
from dataclasses import replace
from startup_config_v132 import make_config as previous

def make_config():
    return replace(previous(),worker_timeout_seconds=7200)
