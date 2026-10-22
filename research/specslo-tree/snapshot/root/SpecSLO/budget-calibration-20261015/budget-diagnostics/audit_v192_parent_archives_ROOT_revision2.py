"""ROOT independent source archive audit; no runtime/backend imports or suites."""
from pathlib import Path
import datetime, hashlib, json, tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-live-small-mixed197-cap64-active32-full-v192'
OUT = D / 'ROOT_V192_PARENT163_172_ARCHIVES_IN_PLACE_AUDIT_REVISION2.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def pin(p):
    return dict(path=str(p), sha256=sha(p))

def merge(*maps):
    result = {}
    for values in maps:
        for path, digest in values.items():
            assert path not in result or result[path] == digest, path
            result[path] = digest
    return result

def verify(path, expected, wanted):
    assert sha(path) == expected, path
    with tarfile.open(path, 'r:gz') as stream:
        rows = stream.getmembers()
        assert len(rows) == len(wanted) == len({v.name for v in rows})
        for row in rows:
            name = '/' + row.name.lstrip('/')
            assert row.isfile() and name in wanted, row.name
            assert hashlib.sha256(stream.extractfile(row).read()).hexdigest() == wanted[name], name
    return dict(**pin(path), regular_members=len(rows), all_member_SHA256_checked=True,
                expected_member_map=dict(wanted), archive_recopied=False)

inputs = read(P / 'SOURCE_INPUTS192.json')
descriptor = B / 'tree-high-normal-cap4-native-families-v148/host-archive/V148_FINAL_CPU_SOURCE.json'
base = read(descriptor)
archived = {'/' + n.lstrip('/'): h for n, h in base['members'].items()}
base_audit = verify(Path(base['path']), base['sha256'], archived)
reports = []
for epoch in (163, 172):
    prior = inputs[f'original_P{epoch}']
    root = Path(prior['path'])
    frozen_path = root / f'FROZEN_SOURCE{epoch}.json'
    cpu_path = root / f'CPU_RESULT{epoch}.json'
    assert sha(frozen_path) == prior['freeze_sha256']
    assert sha(cpu_path) == prior['CPU_sha256']
    frozen = read(frozen_path)
    assert frozen['CPU_sha256'] == sha(cpu_path) and frozen['ROOT_source_reviewed'] is True
    selected = merge(*(frozen[k] for k in (
        'files', 'external_files', 'retained_HOST_evidence', 'execution_guard_inputs', 'peer_reviews')),
        {str(cpu_path): sha(cpu_path), str(frozen_path): sha(frozen_path),
         frozen['root_review']['path']: frozen['root_review']['sha256']})
    for path, digest in selected.items():
        assert Path(path).is_file() and not Path(path).is_symlink() and sha(path) == digest, path
    wanted = {path: h for path, h in selected.items() if archived.get(path) != h}
    result = verify(Path(prior['archive_path']), prior['archive_sha256'], wanted)
    result.update(epoch=epoch, freeze=pin(frozen_path), CPU=pin(cpu_path),
                  selected_current_sources=len(selected), old_HOST_not_rerun=True)
    reports.append(result)
    archived.update(selected)

report = dict(schema='ROOT.V192.parent163-172-archives-in-place.v1', status='passed',
              UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              candidate=str(P), source_inputs=pin(P / 'SOURCE_INPUTS192.json'),
              script=pin(Path(__file__)), base_descriptor=pin(descriptor),
              base_archive=base_audit, parent_archives=reports,
              Torch_imported=False, NPU_used=False, model_initialized=False,
              old_suites_not_rerun=True, source_changed=False,
              READY_default_archiveFalse_claim_corrected=True,
              device_qualification_inherited=False, formal_Goodput=False)
with OUT.open('x') as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(dict(audit=pin(OUT), archives=[dict(epoch=r['epoch'], sha256=r['sha256'],
                 regular_members=r['regular_members']) for r in reports], NPU_used=False)))
