"""Method-independent identity of the exogenous replay workload."""
import hashlib
import json
import math


def fingerprint(rows, expected_requests=463):
    if len(rows) != expected_requests:
        raise ValueError('complete replay workload required')
    canonical = []
    seen = set()
    for row in rows:
        rid = row['request_id']
        if not isinstance(rid, str) or rid in seen:
            raise ValueError('unique actual request IDs required')
        seen.add(rid)
        prompt = row.get('prompt_sha256')
        if prompt is None:
            if not isinstance(row.get('prompt'), str):
                raise ValueError('original prompt text or SHA256 required')
            prompt = hashlib.sha256(row['prompt'].encode()).hexdigest()
        if not isinstance(prompt, str) or len(prompt) != 64 or any(c not in '0123456789abcdef' for c in prompt):
            raise ValueError('actual prompt SHA256 required')
        arrival = row['arrival_offset_sec']
        threshold = row['slo_tpot_ms']
        if (type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < 0
                or type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0):
            raise ValueError('actual finite relative arrival and SLO required')
        canonical.append(dict(request_id=rid, prompt_sha256=prompt, arrival_offset_sec=float(arrival),
                              slo_class=row['slo_class'], slo_tpot_ms=float(threshold)))
    # Input order is preserved: reordering equal-time arrivals may affect the
    # scheduler, so set equality alone does not establish identical replay.
    raw = json.dumps(canonical, separators=(',', ':'), ensure_ascii=True).encode()
    return dict(requests=len(canonical), sha256=hashlib.sha256(raw).hexdigest(),
                fields=['request_id', 'prompt_sha256', 'arrival_offset_sec', 'slo_class', 'slo_tpot_ms'],
                order_preserved=True, method_internal_budget_and_batch_configuration_excluded=True)
