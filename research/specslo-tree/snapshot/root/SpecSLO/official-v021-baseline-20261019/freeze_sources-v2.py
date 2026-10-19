"""Freeze the prepared official checkout, runtime binaries and measurement controls."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/root/SpecSLO/official-v021-baseline-20261019')
D = Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')

def main():
    target = ROOT / 'SOURCE_MANIFEST-v2.json'
    assert not target.exists(), 'Never overwrite a frozen measurement manifest'
    files = set()
    revisions = {}
    for name, expected in [('vllm-ascend','80610e4438dba05011b05f89fc45d91e96992671'),
                           ('vllm','ad7125a431e176d4161099480a66f0169609a690')]:
        repo = ROOT / name
        git = lambda *args: subprocess.check_output(['git', '-C', str(repo), *args])
        assert git('rev-parse', 'HEAD').decode().strip() == expected
        assert not git('status', '--porcelain', '--untracked-files=no').strip()
        revisions[name] = expected
        for raw in git('ls-files', '-z').split(b'\0'):
            if raw:
                path = repo / raw.decode()
                if path.is_file():
                    files.add(path)
        for extra in repo.glob('*.egg-info'):
            files.update(p for p in extra.rglob('*') if p.is_file())
    for directory in [ROOT / 'python-v2', ROOT / 'python', ROOT / 'harness', ROOT / 'vllm-ascend/vllm_ascend/_cann_ops_custom']:
        assert directory.is_dir()
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    cann_python = Path('/root/data/specslo-cann-runtime/cann-9.0.0/python/site-packages')
    files.add(cann_python / 'acl.so')
    files.update(p for p in (cann_python / 'acl').rglob('*') if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts)
    extension = ROOT / 'vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'
    assert extension.is_file()
    files.add(extension)
    files.update(p for p in (ROOT / 'vllm-ascend/vllm_ascend').glob('_build_info*') if p.is_file())
    files.update(p for p in (ROOT / 'vllm/vllm').glob('_version*') if p.is_file())
    files.update(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.json') and not p.name.startswith('HOST'))
    for name in ['publication_witness.py','run_publication_benchmark.py','SOURCE_PINS.json']:
        files.add(ROOT / 'publication-witness' / name)
    for name in ['final_owned_processes.py','final_baseline_cell.py','final_common_comparison.py',
                 'read_common_publication_metrics.py','publication_request_identity.py','prepare_final_comparison_plan.py']:
        files.add(D / name)
    files.add(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json'))
    files.add(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py'))
    files.add(Path('/root/SpecSLO/resume-20261012/run_baseline_calibration-v1.py'))
    files.add(Path('/root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh'))
    manifest = dict(schema='old-official-v021-prepared-source.v1',
                    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), official_revisions=revisions,
                    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    with target.open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(manifest=str(target), files=len(files), sha256=hashlib.sha256(target.read_bytes()).hexdigest())))

if __name__ == '__main__':
    main()
