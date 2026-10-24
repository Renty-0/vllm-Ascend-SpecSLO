"""Bounded independent CPU mapping, actual projection and terminal checks."""
from pathlib import Path
import hashlib
import json
import random
import sys
from host_env_span import setup
setup()
P = Path(__file__).resolve().parent


def main():
    from spans import pack
    from span_fixture import mappings
    rng = random.Random(2301)
    cases = list(mappings())
    for _ in range(48):
        source, dest = [], []
        for block in range(rng.randrange(1, 13)):
            count = rng.randrange(1, 133)
            for j in range(count):
                source.append(block * 256 + j)
                dest.append(8192 + block * 256 + j)
        for j in range(1792 - len(source)):
            source.append(12288 + j)
            dest.append(12288 + j)
        cases.append((source, dest))
    for source, dest in cases:
        result = pack(source, dest, rows=16384)
        # Independent full-byte buffer snapshot, rather than reusing pack's
        # inverse checker. Include untouched rows and identity padding.
        before = bytearray(rng.randbytes(16384 * 4))
        expected, actual = bytearray(before), bytearray(before)
        for s, d in zip(source, dest, strict=True):
            expected[d * 4:d * 4 + 4] = before[s * 4:s * 4 + 4]
        for s, d, n in result['spans']:
            actual[d * 4:(d + n) * 4] = before[s * 4:(s + n) * 4]
        assert actual == expected
    source, dest = cases[1]
    negatives = []
    for bad in ('alias', 'duplicate', 'bounds', 'capacity'):
        ss, dd = list(source), list(dest)
        if bad == 'alias':
            dd[0] = ss[1]
        elif bad == 'duplicate':
            dd[0] = dd[1]
        elif bad == 'bounds':
            ss[0] = 16384
        else:
            ss = list(range(0, 130, 2)) + list(range(12288, 12288 + 1792 - 65))
            dd = list(range(8192, 8257)) + ss[65:]
        try:
            pack(ss, dd, rows=16384)
        except (ValueError, RuntimeError):
            negatives.append(bad)
        else:
            raise AssertionError('Invalid unsafe mapping accepted: ' + bad)
    from probe_span import prepared
    from overlap_cost import run as costs
    body = prepared()
    assert body.__globals__['_concurrent_cost'] is costs
    assert 'span_install' in body.__code__.co_names
    from runtime_span import worker_source
    source = worker_source()
    assert source.count('from probe_span import run') == 1
    assert source.count('dict(numeric_cases=21,performance=True)') == 2
    from guard_span import execution_namespace
    ns = execution_namespace()
    assert ns['execute_locked'].__globals__ is ns and ns['SCOPE']['performance_measured'] is True
    from audit_span import audit
    old = P.parent/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/parallel-common-root-fork-current0127-concurrent-cost-v1'
    try:
        audit(old)
    except KeyError as error:
        assert str(error) == "'span_copy'"
    else:
        raise AssertionError('Old original-copy run cannot qualify new span kernel')
    for p in P.glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_SPAN_COPY_MAPPING_AND_ORIGINAL_COUPLED_LIFECYCLE',
                  full_snapshot_cases=len(cases), unsafe_mappings_rejected=negatives,
                  actual_original_proof_and_timing_routed=True, old_copy_evidence_rejected=True,
                  NPU_used=False, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in P.glob('*.py')})
    (P/'HOST_SPAN_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
