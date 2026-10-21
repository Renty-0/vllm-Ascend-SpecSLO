"""Retain399 mathematics and400 supervision; add one original import path."""
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
PARENT = B / 'tree-frugal399-guard-metadata-repair-v400'
sys.path.insert(0, str(PARENT))
import guard400 as parent

PARENT_FREEZE = '63bb159305a421bcee894b5716057ad5b4c444f4cffca07876dc4273637510bf'
PARENT_ARCHIVE = '31853bbfff18f4784149d32663f1cf3fe4eaebd2205c7cc38bc756036926c82d'
EXPECTED_FREEZE = EXPECTED_ARCHIVE = None
require = parent.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepared():
    parent.EXPECTED_FREEZE, parent.EXPECTED_ARCHIVE = PARENT_FREEZE, PARENT_ARCHIVE
    _, sources, inherited = parent.prepared()
    freeze, archive = HERE / 'FREEZE406.json', HERE / 'SOURCE406.tar.gz'
    require(sha(freeze) == EXPECTED_FREEZE and sha(archive) == EXPECTED_ARCHIVE,
            'explicit406 source and archive pins')
    value = json.loads(freeze.read_text())
    require(value['parent400_freeze_sha256'] == PARENT_FREEZE
            and value['parent400_archive_sha256'] == PARENT_ARCHIVE, 'same400 frozen donor')
    files = dict(value['files'], **{str(freeze): EXPECTED_FREEZE})
    require(not set(files) & set(sources), 'disjoint406 startup source closure')
    for path, digest in value['existing_inputs'].items():
        require(sha(path) == digest, 'exact406 existing input ' + path)
    with tarfile.open(archive, 'r:gz') as saved:
        members = saved.getmembers()
        require(len(members) == len(files) and {m.name for m in members}
                == {name.lstrip('/') for name in files}, 'complete406 unique archive')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest()
                    == files['/' + member.name] == sha('/' + member.name), 'exact406 archive source')
    sources.update(files)
    return value, sources, dict(parent=inherited, own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE, numerical_bodies_changed=False,
        startup_import_path_repair406=True, model_qualification_inherited=False)


def launch_argv(tag):
    args = parent.execution_namespace()['launch_argv'](tag)
    old = str(B / 'tree-M120-S32-frugal-independent-target-host-v399/launch399.sh')
    require(args.count(old) == 1, 'one original399 launcher replacement')
    return [str(HERE / 'launch406.sh') if value == old else value for value in args]


def storage(tag=None):
    result = parent.storage(tag)
    result['aggregate_private_bytes'] += sum(f.stat().st_size for f in HERE.rglob('*') if f.is_file())
    require(result['aggregate_private_bytes'] <= 3584 * 1024**2, 'same406 bounded private storage')
    return result


def execution_namespace():
    result = parent.execution_namespace()
    result.update(prepared=prepared, launch_argv=launch_argv, storage=storage, G=HERE)
    result['SCOPE'] = dict(result['SCOPE'], startup_import_repair406=True)
    require(result['execute_locked'].__globals__ is result, 'original400 supervision globals')
    return result


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = args.expected_freeze_sha256, args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None
            and all(re.fullmatch('[0-9a-f]{64}', value)
                    for value in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)), 'safe406 tag and pins')
    prepared()
    namespace = execution_namespace()
    with (B / '.ROOT_NPU_EXECUTION.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__ == '__main__':
    main()
