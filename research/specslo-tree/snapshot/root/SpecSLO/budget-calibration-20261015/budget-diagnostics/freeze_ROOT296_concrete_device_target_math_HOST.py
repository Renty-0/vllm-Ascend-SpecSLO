"""Freeze a fresh296 diagnostic without editing or inheriting291 results."""
import datetime
import hashlib
import io
import json
import tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-concrete-device-span-dual-cache-target-cost-v296'
OLD = B / 'tree-all-layer-prefix-dual-cache-target-cost-v291'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert not (P / 'FREEZE296.json').exists()
    assert sha(OLD / 'FREEZE291.json') == 'd6cb884db523720b1d1aa240377cfeae0b2b29d172138f3bc1c6604def2069ea'
    assert read(P / 'HOST296_RESULT.json')['cases'] == 408
    assert read(P / 'HOST_GRID296_RESULT.json')['case_count'] == 408
    assert read(P / 'HOST_SPLIT296_SPANS_RESULT.json')['actual_cases'] == 408
    assert read(P / 'HOST_DEVICE296_RESULT.json')['invalid_bindings_rejected'] == 5
    for filename in ('FIXTURE', 'DECLARED'):
        ending = '.json' if filename == 'FIXTURE' else '_CASES.json'
        assert sha(P / f'{filename}296{ending}') == sha(OLD / f'{filename}291{ending}')
    old_text = (OLD / 'static_state291.py').read_text().replace('291', '296')
    expected = old_text.replace('state.prefix296_copies=SpanCopies(caches,engine.device)',
        'from device296 import resolve_cache_device\n'
        '    actual_device=resolve_cache_device(engine.device,caches[0][0].device,torch.npu.current_device())\n'
        '    state.prefix296_copies=SpanCopies(caches,actual_device)')
    assert expected == (P / 'static_state296.py').read_text()
    review = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        schema='ROOT296.concrete-device-span-target-NN.v1',
        change='Resolve current concrete NPU from actual cache tensor before unchanged strict all128-cache pointer/storage checks',
        kernel_and_attention_bodies_unchanged=True,
        unchanged_geometry='M120/S24/uniformQ5/KV2560/C96',
        unchanged_fixture_and_408_case_declaration=True,
        numeric_scope='408 actual NN cases on three targets; all128KV/hidden/greedy bitwise and72 own prefixes protected',
        cost_scope='11 configurations, B4..8,20 rotated synchronous target pairs, after math passes',
        qualification_inherited=False, performance=False, Goodput=None, adopted=False)
    (P / 'REVIEW296.json').write_text(json.dumps(review, indent=2) + '\n')
    files = dict(read(OLD / 'FREEZE291.json')['files'])
    files[str(OLD / 'FREEZE291.json')] = sha(OLD / 'FREEZE291.json')
    for path in P.iterdir():
        if path.is_file():
            if path.suffix == '.py':
                compile(path.read_text(), str(path), 'exec')
            files[str(path)] = sha(path)
    for path in (Path(__file__), D / 'build_ROOT296_concrete_cache_device_HOST.py',
                 D / 'ROOT291_INITIALIZATION_DEVICE_FAILURE.json'):
        files[str(path)] = sha(path)
    for path, digest in files.items():
        assert sha(path) == digest, path
    freeze = P / 'FREEZE296.json'
    with freeze.open('x') as stream:
        json.dump(dict(schema='ROOT296.frozen-target-math-diagnostic.v1',
                       files=files, review=review, NPU_used=False,
                       model_qualification_inherited=False), stream, indent=2)
        stream.write('\n')
    inventory = {**files, str(freeze): sha(freeze)}
    archive = P / 'SOURCE296.tar.gz'
    with tarfile.open(archive, 'x:gz') as output:
        for path, digest in sorted(inventory.items()):
            data = Path(path).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
            member = tarfile.TarInfo(path.lstrip('/'))
            member.mode, member.size, member.mtime = 0o600, len(data), 0
            output.addfile(member, io.BytesIO(data))
    with tarfile.open(archive, 'r:gz') as saved:
        assert len(saved.getmembers()) == len(inventory)
        for member in saved:
            assert member.isfile()
            assert hashlib.sha256(saved.extractfile(member).read()).hexdigest() == inventory['/' + member.name]
    pins = dict(freeze_sha256=sha(freeze), archive_sha256=sha(archive), members=len(inventory))
    (P / 'SOURCE296_RECEIPT.json').write_text(json.dumps(pins, indent=2) + '\n')
    print(json.dumps(pins))


if __name__ == '__main__':
    main()
