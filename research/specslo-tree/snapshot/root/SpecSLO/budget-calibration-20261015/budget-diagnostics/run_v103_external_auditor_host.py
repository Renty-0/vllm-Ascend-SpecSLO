"""Record actual HOST parser tests; never run the terminal actual analyzer."""
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
FILES=('analyze_v103_actual_active_cap_controls.py','test_analyze_v103_actual_active_cap_controls_host.py',
       'run_v103_external_auditor_host.py','analyze_v103_actual_warmup_profile.py')


def hashes():return {str(P/name):hashlib.sha256((P/name).read_bytes()).hexdigest() for name in FILES}


def main():
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Original Native AST ABI requires retained Python3.11')
    before=hashes();start=time.perf_counter();text=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromName('test_analyze_v103_actual_active_cap_controls_host')
    result=unittest.TextTestRunner(stream=text,verbosity=2).run(suite);after=hashes()
    row=dict(schema='V103.external-actual-auditor-preparation-HOST-tests.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_HOST_argv=[sys.executable,*sys.argv],Python=sys.version,
        tests_run=result.testsRun,failures=result.failures,errors=result.errors,elapsed_seconds=time.perf_counter()-start,
        input_hashes=before,input_hashes_after=after,status='passed' if result.wasSuccessful() and before==after else 'failed',
        torch_imported='torch' in sys.modules,NPU_initialized=False,model_initialized=False,
        actual_full463_audit_executed=False,actual_supervisor_receipt_read=False,no_waiter_installed=True)
    with (P/'V103_EXTERNAL_ACTUAL_AUDITOR_HOST_RESULT.json').open('x') as stream:json.dump(row,stream,indent=2);stream.write('\n')
    with (P/'V103_EXTERNAL_ACTUAL_AUDITOR_HOST.log').open('x') as stream:stream.write(text.getvalue())
    print(json.dumps(row));return int(row['status']!='passed' or row['torch_imported'])


if __name__=='__main__':raise SystemExit(main())
