"""Keep the complete428 audit and the original captured commit identity."""
import json
from pathlib import Path
import audit428 as original


def audit(directory, selected_windows):
    value = original.audit(directory, selected_windows)
    bindings = []
    for rank in range(4):
        report = json.loads((Path(directory) / f'rank{rank}-local-diagnostic.json').read_text())
        current = report['ROOT430_original_commit_binding']
        if not (current['same_prestartup_protocol_and_original_commit_identity'] is True
                and current['exactly_original_runtime199_capture_and_engine_binding'] is True
                and current['Native_flags_assigned'] is False
                and current['NN_permission_granted'] is False):
            raise RuntimeError('ROOT430 actual original commit function binding required')
        bindings.append(current)
    if not all(v == bindings[0] for v in bindings):
        raise RuntimeError('ROOT430 all4 original commit source identities must agree')
    return dict(value, ROOT430_original_prestartup_commit_binding=bindings[0])
