"""Retained actual finite M120 metadata and full-history proof audit."""
import gzip
import json
from pathlib import Path
import audit199


def audit(output):
    value=audit199.audit(output)
    output=Path(output)
    for rank in range(4):
        report=json.loads((output/f'rank{rank}-local-diagnostic.json').read_text())
        binding=report['component']['current_target_binding391']
        if not binding['actual_best376_target_backend']or binding['best376_draft_claimed']:
            raise RuntimeError('ROOT391 exact current target/local draft scope')
        if rank and report['component']['actual_metadata_matches391']<=0:
            raise RuntimeError('ROOT391 actual Native physical metadata not witnessed')
        for path in (output/'bounded199').glob(f'rank{rank}-pass*.json.gz'):
            raw=json.loads(gzip.decompress(path.read_bytes()))
            for phase in raw['phases']:
                if rank:
                    shape=phase['target']['shape']
                    if not(shape['M']==120 and shape['S']==24 and shape['family']=='static_FULL_M120_Q5_S24'):
                        raise RuntimeError('ROOT391 actual current M120 finite phase')
    return dict(value,current_target_backend391='actual376 M120/Q5/S24 uniform KV',
        original173_fresh_baseline_retained=True,best376_draft_claimed=False,
        numerical_audit_reexecuted_by_HOST=False,production_permission=False,Raw=None,Goodput=None)
