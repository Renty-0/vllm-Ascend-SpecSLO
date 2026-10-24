"""Actual original full463/own-AR audit plus the newly installed copy path."""
from pathlib import Path
import json
import audit466 as original


def audit(directory, windows):
    result = original.audit(directory, windows)
    details = []
    for rank in range(4):
        path = Path(directory)/f'rank{rank}-local-diagnostic.json'
        value = json.loads(path.read_text())
        geometry = value['ROOT257_actual_geometry']
        data = geometry['ROOT470_span_copy']
        if not (geometry['original_B1to5_budget_allocator_unchanged'] is True
                and geometry['original_own_AR_implementation_unchanged'] is True):
            raise RuntimeError('Only the declared target copy implementation changes')
        if rank == 0:
            if data != dict(draft_unchanged=True):
                raise RuntimeError('The actual draft implementation remains unchanged')
            continue
        proof = data['raw_proof']
        if not (data['actual_layers'] == 64 and data['bindings_unchanged'] is True
                and data['original_gather_scatter_bypassed'] is True
                and data['BF16_arithmetic'] is False and data['checked_real_layouts'] > 42
                and proof['raw16_input_patterns'] == 65536 and proof['width'] in (256,384)
                and proof['full_storage_compared'] is True and proof['model_KV_mutated'] is False
                and proof['same_descriptors_changed_and_restored'] is True
                and [(x['case'], x['variant']) for x in proof['cases']] == [(i,v) for i in range(7) for v in (0,1,0)]
                and all(x['eager_and_graph_full_raw16_exact'] is True for x in proof['cases'])):
            raise RuntimeError('Actual complete copy proof and same-instance64-layer binding required')
        details.append(dict(rank=rank, **data))
    return dict(result, ROOT470_original466_all_checks_retained=True,
                ROOT470_actual_target_span_copy=details)
