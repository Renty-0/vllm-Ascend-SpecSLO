"""Priority-protected refill for the exact348 model, cost gate and causal gains."""
import hashlib
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(BASE / 'tree-priority-protected-live-refill-host-v354'),
               str(BASE / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348')]
from admission354 import descriptor as admission_descriptor
from admission354 import patch as admission_patch
from generation348 import _original_tree_descriptor, _original_tree_patch


MODE = 'protect_tight_normal'


def patch(source):
    return admission_patch(_original_tree_patch(source), MODE)


def descriptor(source):
    value = _original_tree_descriptor(source)
    actual = patch(source)
    value.update(projected_source_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                 ROOT348_LIVE_source_sha256=hashlib.sha256(actual.encode()).hexdigest(),
                 ROOT356_admission_mode=MODE,
                 ROOT356_admission=admission_descriptor(_original_tree_patch(source), MODE),
                 empty_active_only=False, admission_only_control=True,
                 target_draft_maths_and_actual_B1to8_cost_policy_unchanged=True)
    return value
