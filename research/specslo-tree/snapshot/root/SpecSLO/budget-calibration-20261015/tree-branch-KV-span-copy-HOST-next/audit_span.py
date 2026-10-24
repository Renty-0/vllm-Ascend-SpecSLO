"""Retain actual original neural/timing audit; require new raw-copy evidence."""
from pathlib import Path
import json


def audit(directory):
    from audit_overlap import audit as original
    result, label, detail = original(directory)
    proofs = []
    for rank in range(4):
        data = json.loads((Path(directory) / f'rank{rank}-numeric.json').read_text())['span_copy']
        if rank == 0:
            if data != dict(draft_unchanged=True):
                raise RuntimeError('Draft copy path is not replaced')
            continue
        proof = data['raw_proof']
        if not (data['actual_layers'] == 64 and data['bindings_unchanged'] is True
                and data['original_gather_scatter_bypassed'] is True
                and data['BF16_arithmetic'] is False and data['checked_real_layouts'] >= 42
                and proof['width'] in (256, 384) and proof['raw16_input_patterns'] == 65536
                and proof['initialized_private_rows'] == 16384
                and proof['full_storage_compared'] is True and proof['model_KV_mutated'] is False
                and proof['same_descriptors_changed_and_restored'] is True
                and len(proof['cases']) == 21
                and all(c['eager_and_graph_full_raw16_exact'] is True for c in proof['cases'])):
            raise RuntimeError('Actual complete span-copy precheck and model bindings required')
        if [(c['case'], c['variant']) for c in proof['cases']] != [(i, v) for i in range(7) for v in (0, 1, 0)]:
            raise RuntimeError('Actual changing descriptors and raw16 restoration order')
        proofs.append(dict(rank=rank, **data))
    return dict(result, target_span_copy_proofs=proofs), label, detail
