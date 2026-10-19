"""Actual331 late auditor return type regression, model-free."""
import json
from pathlib import Path
import tempfile
import guard332

previous=guard332.previous
original=previous.previous.actual_audit
with tempfile.TemporaryDirectory(prefix='ROOT332-HOST-',dir=str(Path(__file__).resolve().parent))as folder:
    root=Path(folder)
    for rank in range(1,4):
        bridge=dict(construction_witnesses=[dict(layer=i,K=k,calls=2)for i in range(64)for k in(3072,8576)],
                    FIA_rows=160,actual_projection_MC2_rows=165,profile160_claimed=False,
                    pending_tickets=0,deferred_chain_enabled=False)
        (root/f'rank{rank}-numeric.json').write_text(json.dumps(dict(actual165_MC2_bridge330=bridge)))
        (root/f'rank{rank}-bridge330-before-tree.json').write_text(json.dumps(dict(rank=rank,
            installed_before_diagnostic_tree=True,physical_FIA_rows160=True,actual_projection165=True)))
    def prior(directory):
        assert Path(directory)==root
        return (dict(status='original532-audit-double'), 'diagnostic_only', dict(original=True))
    previous.previous.actual_audit=prior
    try:
        try:previous.actual_audit(root)
        except TypeError as error:
            assert 'tuple' in str(error)
        else:raise AssertionError('Original guard331 tuple error must reproduce')
        result=guard332.actual_audit(root)
        assert result[0]['bridge331_actual_FIA160_projection165_fresh532_passed']is True
        assert result[1:] == ('diagnostic_only',dict(original=True))
    finally:previous.previous.actual_audit=original
report=dict(status='HOST_pass',original_tuple_error_reproduced=True,
            corrected_actual_audit_code_all3routes128_checked=True,
            parent3tuple_contract_preserved=True,whole_original_audit_inverse=True,
            no_model_or_NPU_run=True,no_neural_or_performance_qualification=True)
(Path(__file__).resolve().parent/'HOST332_ACTUAL_AUDIT_CONTRACT_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
