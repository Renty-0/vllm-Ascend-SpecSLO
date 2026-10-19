"""Correct one launch pin while reusing all frozen351 numerical code."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
B = HERE.parent
PARENT = B/'tree-small-refill-incumbent-prefix-protection-v351'
sys.path.insert(0, str(PARENT))
import guard351 as parent

PINS = json.loads((PARENT/'SOURCE351_RECEIPT.json').read_text())
ORIGINAL218 = 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'
EXPECTED_FREEZE = None
EXPECTED_ARCHIVE = None


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepared():
    assert sha(HERE/'FREEZE352.json') == EXPECTED_FREEZE
    own = json.loads((HERE/'FREEZE352.json').read_text())
    assert all(sha(path) == digest for path, digest in own['files'].items())
    parent.EXPECTED_FREEZE = PINS['freeze_sha256']
    parent.EXPECTED_ARCHIVE = PINS['archive_sha256']
    _, sources, previous = parent.prepared()
    assert own['parent_freeze_sha256'] == PINS['freeze_sha256']
    additive = dict(own['files'])
    additive[str(HERE/'FREEZE352.json')] = EXPECTED_FREEZE
    assert not set(additive) & set(sources)
    assert sha(HERE/'SOURCE352.tar.gz') == EXPECTED_ARCHIVE
    with tarfile.open(HERE/'SOURCE352.tar.gz', 'r:gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(additive)
        assert {m.name for m in members} == {name.lstrip('/') for name in additive}
        for member in members:
            assert member.isfile()
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == additive['/'+member.name]
    sources.update(additive)
    assert sha(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218/FREEZE218.json') == ORIGINAL218
    return own, sources, dict(own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, parent_numeric_source_verified_in_place=previous,
        numeric_source_version=351, adapter_version=352, device_qualification_inherited=False)


def launch_argv(tag):
    parent.EXPECTED_FREEZE = PINS['freeze_sha256']
    parent.EXPECTED_ARCHIVE = PINS['archive_sha256']
    argv = parent.launch_argv(tag)
    fields = [i for i, value in enumerate(argv) if value.startswith('SPECSLO_V218_ROOT_FREEZE_SHA=')]
    assert len(fields) == 1
    index = fields[0]
    assert argv[index].split('=',1)[1] == ORIGINAL218.replace('350','351')
    argv[index] = 'SPECSLO_V218_ROOT_FREEZE_SHA='+ORIGINAL218
    return argv


def storage(tag=None):
    value = parent.storage(tag)
    own_bytes = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    value['aggregate_private_bytes'] += own_bytes
    assert value['aggregate_private_bytes'] <= parent.MAX_AGGREGATE_BYTES
    return value


def execution_namespace():
    ns = parent.execution_namespace()
    assert ns['execute_locked'].__globals__ is ns
    ns.update(G=HERE, prepared=prepared, launch_argv=launch_argv, storage=storage)
    ns['SCOPE'].update(numeric_source_version=351, launch_adapter_version=352,
                       original218_launch_pin_verified=True)
    return ns


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag)
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = args.expected_freeze_sha256, args.expected_archive_sha256
    assert all(re.fullmatch('[0-9a-f]{64}', value) for value in (EXPECTED_FREEZE, EXPECTED_ARCHIVE))
    prepared()
    with parent.LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        execution_namespace()['execute_locked'](args.tag)


if __name__ == '__main__': main()
