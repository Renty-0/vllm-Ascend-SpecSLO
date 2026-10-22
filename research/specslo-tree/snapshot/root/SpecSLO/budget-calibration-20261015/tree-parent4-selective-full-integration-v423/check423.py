"""Frozen source/origin gate; all actual neural qualification stays fresh."""
import hashlib
import json
from pathlib import Path
import sys
from source_paths423 import install as install_source_paths
install_source_paths()

P = Path(__file__).resolve().parent
B = P.parent
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
OLD = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
OLD_SHA = 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'
PARENT_FREEZE = 'b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'


def require(value, message):
    if value is not True:
        raise RuntimeError('ROOT423: '+message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024**2), b''):
            value.update(data)
    return value.hexdigest()


def ready(expected=None):
    expected = expected or read(P/'SOURCE423_RECEIPT.json')['freeze_sha256']
    require(sha(P/'FREEZE423.json') == expected, 'explicit423 source freeze')
    value = read(P/'FREEZE423.json')
    for filename, digest in value['files'].items():
        require(sha(filename) == digest, 'immutable423 input '+filename)
    for root, number in [(B/'tree-normal24-derived-callback-repair-v405', 405),
                         (B/'tree-parent4-normal24-SLO-mixed-protocol-v419', 419)]:
        record = read(root/f'SOURCE{number}_RECEIPT.json')
        require(sha(root/f'FREEZE{number}.json') == record['freeze_sha256']
                and sha(root/f'SOURCE{number}.tar.gz') == record['archive_sha256'],
                'retained source/archive '+str(number))
        frozen = read(root/f'FREEZE{number}.json')
        require(all(value['files'].get(p) == h for p, h in frozen['files'].items()),
                'all original source/evidence bytes retained '+str(number))
    import check405
    import check419
    check405.ready('4972b71c2217af361c95da2516de5f50c92223d367ac3d8175a5e3a729931926')
    check419.ready('56dd68d6eb08960596c028da80ec732082df58e496725ace8a93c238e50dd037')
    for filename in ('HOST423_FACTORY_RESULT.json', 'HOST423_LIVE_BINDING_RESULT.json',
                     'HOST423_CONSTRUCTION_RESULT.json', 'HOST423_PAID_BINDING_RESULT.json',
                     'HOST423_CLEAN_STARTUP_RUN.json'):
        host = read(P/filename)
        require(host['status'].startswith('PASS_HOST423') and host['NPU_used'] is False,
                'actual HOST prerequisite '+filename)
    return value


if __name__ == '__main__':
    ready()
