"""Construct real worker/trace/auditor with only the unfrozen source gate mocked."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import host_paths383

P=Path(__file__).parent;B=P.parent


def main():
    import runtime383,guard383,trace383
    with patch('check383.ready',return_value={'HOST_unfrozen_construction_only':True}):
        actual=runtime383.build_runtime()
    assert actual['worker'].__globals__['worker'] is runtime383.worker
    assert actual['worker'].__globals__['P']==B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
    projected=actual['projected_source']
    assert 'from scope383 import issue'in projected and 'from trace383 import run_windows'in projected
    assert 'from audit383 import audit'in projected
    ns=guard383.execution_namespace()
    assert ns['MAX_SECONDS']==10800 and ns['SCOPE']['fresh_same_instance_adaptive_hybrid_cases']==693
    assert ns['actual_audit'] is guard383.actual_audit
    try:guard383.actual_audit('/nonexistent-HOST383-construction')
    except FileNotFoundError:pass
    except RuntimeError as exc:assert str(exc)=='V209: own actual directory'
    else:raise AssertionError('expected missing actual output after actual audit imports')
    trace=trace383.namespace()
    from scope383 import report
    assert trace['standalone_report']is report
    for name in ['startup383.sh','launch383.sh']:
        subprocess.run(['/bin/bash','-n',str(P/name)],check=True)
    env=lambda f:[s for s in f.read_text().splitlines()if s.startswith(('export VLLM_','unset VLLM_'))]
    assert env(P/'startup383.sh')==env(B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257/startup257.sh')
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    value=dict(status='PASS_HOST383_ACTUAL_WORKER_TRACE_AUDITOR_CONSTRUCTION',NPU_used=False,
        source_gate_mocked_for_unfrozen_construction_only=True,actual_frozen_gate_required_before_launch=True,
        original_worker_main_and_trace_bodies_retained=True,actual_audit_imports_checked=True,
        model_environment_unchanged=True,model_qualified=False,Raw=None,Goodput=None)
    (P/'HOST383_CONSTRUCTION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
