"""Retain the complete425 audit and require the actual428 reference binding."""
import json
from pathlib import Path
import audit425 as original
from metadata428 import descriptor


def audit(directory,selected_windows):
    value=original.audit(directory,selected_windows)
    for rank in range(4):
        report=json.loads((Path(directory)/f'rank{rank}-local-diagnostic.json').read_text())
        current=report['ROOT428_reference_metadata_binding']
        if not(current['source']==descriptor() and current['current_cases']==75
               and current['original425_backend_unchanged']is True
               and current['Native_flags_assigned']is False
               and (rank==0 or current['actual_target_metadata_matches']>0)):
            raise RuntimeError('ROOT428 actual current-selector75-case evidence required')
    return dict(value,ROOT428_current_reference_selector='uniform_KV_bucket425_enabled',
        ROOT428_original425_backend_and_all_math_gates_unchanged=True)
