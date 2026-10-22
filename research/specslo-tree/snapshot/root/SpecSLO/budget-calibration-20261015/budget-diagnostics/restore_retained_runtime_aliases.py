"""Check, or restore missing aliases to the exact previously frozen runtime.

No download, installer, package change, runtime-byte mutation or overwrite.
"""
import argparse
import hashlib
import json
from pathlib import Path

LOCK=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json')
ALIASES={
 '/usr/local/Ascend/cann-9.0.0':'/root/data/specslo-cann-runtime/cann-9.0.0',
 '/opt/specslo-cann/nnal':'/root/data/specslo-cann-runtime/nnal',
}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore',action='store_true',help='Create only missing, hash-verified aliases.')
    args=parser.parse_args()
    lock=json.loads(LOCK.read_text())
    checked=[]
    for name,wanted in lock['external_files_sha256'].items():
        for alias,target in ALIASES.items():
            if name.startswith(alias+'/'):
                retained=Path(target+name[len(alias):])
                if not retained.is_file() or sha(retained)!=wanted:
                    raise RuntimeError('Retained runtime does not match original frozen bytes: '+str(retained))
                checked.append(name)
                break
    if len(checked)!=13:
        raise RuntimeError('Original frozen external runtime file set differs')
    missing=[]
    for alias,target in ALIASES.items():
        path=Path(alias)
        if path.exists() or path.is_symlink():
            if path.resolve()!=Path(target).resolve():
                raise RuntimeError('Refusing to replace an existing different runtime path: '+alias)
        else:
            missing.append((path,target))
    if missing and not args.restore:
        raise RuntimeError('Missing aliases; verified retained bytes. Run with --restore: '+', '.join(str(p) for p,_ in missing))
    for path,target in missing:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.symlink_to(target,target_is_directory=True)
    for name,wanted in lock['external_files_sha256'].items():
        if not Path(name).is_file() or sha(name)!=wanted:
            raise RuntimeError('Original external runtime identity mismatch: '+name)
    print(json.dumps({'status':'passed','retained_files_verified':len(checked),
        'original_external_files_verified':len(lock['external_files_sha256']),
        'created_aliases':[str(p) for p,_ in missing],
        'runtime_bytes_or_packages_changed':False},ensure_ascii=False))

if __name__=='__main__':
    main()
