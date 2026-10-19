"""All inherited fresh693 numerical/cost gates plus actual hybrid route checks."""
import json
from pathlib import Path


def audit(directory):
    from audit378 import audit as original
    result,reason,evidence=original(directory)
    for rank in range(4):
        report=json.loads((Path(directory)/f'rank{rank}-numeric.json').read_text())
        s=report['hybrid382_final']
        if rank==0:
            if s!=dict(applied=False,original_optimized_draft_unchanged=True):
                raise RuntimeError('ROOT382 original draft retained')
            continue
        c=s['actual']
        if not(s['applied'] is True and s['precision']=='BF16'and s['quantized'] is False
            and s['Native_qualification_flags_assigned'] is False
            and s['original_column_padding_attributes_unchanged'] is True
            and c['column_modules']==128 and c['head_modules']==1
            and c['model_frames80']>0 and c['model_frames_other']>0
            and c['plain_GEMM80_to120']>0 and c['MC2_GEMM80_to120']>0):
            raise RuntimeError('ROOT382 actual conditional80/120 original-precision routing required')
    result=dict(result,hybrid382_actual_routes_checked=True,full463=False,adopted=False)
    return result,reason,evidence
