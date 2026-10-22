"""Keep every original437 model/protocol audit; audit the tighter policy."""
import gzip,json
from pathlib import Path
from audit437 import audit as original


def audit(directory,windows):
    value=original(directory,windows)
    from admission438 import audit_trace
    directory=Path(directory);out=[]
    for path in sorted(directory.glob('**/rank1-*-actual-trace.json.gz')):
        trace=json.loads(gzip.decompress(path.read_bytes()))
        if trace['role']not in('warmup65','zero65-1','zero65-2','full463'):raise RuntimeError('ROOT438 explicit known trace role')
        out.append(dict(path=str(path),generation=trace['source_generation'],policy=audit_trace(trace)))
    if not out:raise RuntimeError('ROOT438 actual scheduling policy traces required')
    return dict(value,ROOT438_actual_SLO_aware_refill_audits=out,global_active_limit=64)
