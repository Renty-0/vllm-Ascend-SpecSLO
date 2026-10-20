"""Exercise original199 source/preparation and real projected worker before models."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

P=Path(__file__).parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'),
    str(B/'tree-contiguous-branch-tail-final-metadata-math-v234')]


def main():
    from check199 import require_ready
    parent=require_ready()
    import runtime386,guard386
    source=runtime386.projected_source();compile(source,'HOST386.actual-original199-projection','exec')
    assert source.count('from check386 import require_ready')==2
    assert 'from shutdown234 import finish'in source
    with patch('check386.ready',return_value={'HOST_unfrozen_construction_only':True}):
        actual=runtime386.build_runtime()
    assert actual['worker'].__globals__['worker'] is runtime386.worker
    assert actual['worker'].__globals__['P']==B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'
    ns=guard386.execution_namespace()
    assert ns['P']==B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'
    assert ns['ACTION']=='bounded-parent4-child4-shared4-diagnostic'
    assert ns['MAX_SECONDS']==10800
    try:guard386.actual_audit('/nonexistent-HOST386')
    except FileNotFoundError:pass
    except RuntimeError as exc:
        if 'actual'not in str(exc)and 'directory'not in str(exc):raise
    else:raise AssertionError('expected absent actual output')
    # This executes the real pre-install helper preparation without invoking
    # the installer, the model constructor, or a device operation.
    import extension199
    with tempfile.TemporaryDirectory(prefix='HOST386-prepare-',dir=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199')as directory:
        entry,controls=extension199.prepare_install(directory)
        origins=extension199.loaded_origins()
        assert callable(entry['install'])and controls is not None and origins
    initialized=False
    if 'torch'in sys.modules:
        initialized=bool(sys.modules['torch'].npu.is_initialized())
    assert initialized is False
    value=dict(status='PASS_HOST386_ORIGINAL199_READY_PREPARE_ORIGINS_AND_ACTUAL_RUNTIME',
        NPU_used=False,source_gate_mocked_only_for_unfrozen_worker_construction=True,
        actual_original199_require_ready_passed=True,parent199_identity=parent['identity'],
        actual_original_prepare_install_and_origins_passed=True,
        Torch_imported='torch'in sys.modules,model_constructed=False,
        models_qualified=False,production_permission=False,Raw=None,Goodput=None)
    (P/'HOST386_CONSTRUCTION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
