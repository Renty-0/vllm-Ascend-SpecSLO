"""All432 numerical/clock/own-AR audits plus exact434 source routing."""
import json
from pathlib import Path
import audit432 as original


def audit(directory,selected_windows):
    value=original.audit(directory,selected_windows)
    for rank in range(4):
        report=json.loads((Path(directory)/f'rank{rank}-local-diagnostic.json').read_text())
        row=report['ROOT434_frozen_implementation']
        if not(row['frozen432_implementation_unchanged']is True
               and row['global_active_limit']==64 and row['normal_and_single_target_limit']==24
               and row['original427_current_model_NN_required']is True
               and row['actual_incumbent_time_charge_required']is True
               and row['Native_flags_assigned']is False):
            raise RuntimeError('ROOT434 own exact432 current-model implementation required')
    return dict(value,ROOT434_frozen432_implementation_and_all_math_retained=True)
