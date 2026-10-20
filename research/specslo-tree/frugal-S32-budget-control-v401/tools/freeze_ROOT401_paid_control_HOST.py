"""Retain HOST-only control candidate without issuing any runtime permission."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-frugal-S32-paid-budget-control-host-v401'
REPO = Path('/root/SpecSLO/checkpoints/github-SpecSLO-Tree')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((P / 'HOST401_RESULT.json').read_text())
    assert result['paid_wide_shaper_cases'] == 346
    assert result['exact_legacy_shaper_cases'] == 254
    assert result['whole_ready_ticket_resource_cases'] == 600
    parent = B / 'tree-active64-whole-proposal-segment-limit-v398'
    assert sha(parent / 'FREEZE398.json') == 'dfb6cfd44d2f524d7ad28892ec055d50d1035910fb71dbbc5486dc060cb5e73d'
    plan = dict(
        kind='HOST_ONLY_NOT_INSTALLED', global_max_num_seqs=64,
        narrow='original shaper unchanged when normal group is at most24',
        wide='25..32 newly proposed normal owners, C<=88, at most24 B3 owners; others B<=2',
        per_request_caps='strict SLO class then original deferred age, urgency and expected progress',
        live_state='unchanged original request states; shadow caps supplied to original shaper',
        whole_tickets='existing tickets never shortened; independent large-group packing constraint',
        optional_eager='disabled for this unqualified wide candidate, narrow behavior unchanged',
        nn_dependency='finite399 via guard400 not yet launched or qualified',
        future_integration=[
            'Require399 independent hidden/all128KV/graph-eager passes on all3 targets first.',
            'Require actual paired S32 and S24 graph costs; host layout is not a speedup.',
            'Use original S24 branch path for B4/B5 and narrow target calls.',
            'Use S32 only for25..32 whole linear B1..3 tickets with paid C88 and large-group quota.',
            'Actual pad, prepare and observed dispatch must agree on the selected mode on every rank.',
            'Preserve same M120 weights and uniform KV extents, sampler, arrival/SLO clocks and guards.',
            'Requalify dual dispatch, full463 with global64, two zero traces and independent own-AR.',
            'Compare matching global64 fixed gamma4 and official MC2 baselines; no historical48 claim.',
        ],
        current64_control=dict(Raw=599.5623377435959, Goodput=214.96187487135404,
                               attained=166, requests=463, independently_own_AR_qualified=True),
        neural_sources_modified=False, NPU_used=False, models_constructed=0,
        production_permission=False, full463=False, Raw=None, Goodput=None)
    (P / 'PLAN401.json').write_text(json.dumps(plan, indent=2) + '\n')
    names = ['policy401.py', 'packing401.py', 'check401_HOST.py', 'HOST401_RESULT.json', 'PLAN401.json']
    files = {str(P / name): sha(P / name) for name in names}
    files[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    freeze = dict(schema='ROOT401.HOST-only-paid-S32-control-freeze.v1', files=files,
                  parent398_freeze_sha256=sha(parent / 'FREEZE398.json'),
                  frozen_control_sources=result['source_sha256'],
                  parent_packing398_sha256=sha(parent / 'packing398.py'),
                  projection397_sha256=sha(B / 'tree-M120-S32-frugal-budget-projection-host-v397/projection397.py'),
                  model_qualification=False, production_permission=False)
    (P / 'FREEZE401.json').write_text(json.dumps(freeze, indent=2) + '\n')
    archive = P / 'SOURCE401.tar.gz'
    with tarfile.open(archive, 'w:gz') as saved:
        for name in files:
            saved.add(name, arcname=name.lstrip('/'), recursive=False)
    with tarfile.open(archive, 'r:gz') as saved:
        for name, digest in files.items():
            data = saved.extractfile(name.lstrip('/')).read()
            assert hashlib.sha256(data).hexdigest() == digest
    receipt = dict(source_freeze_sha256=sha(P / 'FREEZE401.json'),
                   source_archive_sha256=sha(archive), source_archive_bytes=archive.stat().st_size,
                   archived_members=len(files), NPU_used=False, model_qualification=False)
    (P / 'SOURCE401_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    dest = REPO / 'research/specslo-tree/frugal-S32-budget-control-v401'
    dest.mkdir(exist_ok=False)
    manifest = {}
    for name in names + ['FREEZE401.json', 'SOURCE401_RECEIPT.json', 'SOURCE401.tar.gz']:
        source = P / name
        shutil.copyfile(source, dest / name)
        manifest[name] = dict(source=str(source), sha256=sha(source))
    tools = dest / 'tools'
    tools.mkdir()
    recipe = Path(__file__).resolve()
    shutil.copyfile(recipe, tools / recipe.name)
    manifest['tools/' + recipe.name] = dict(source=str(recipe), sha256=sha(recipe))
    (dest / 'EXPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for relative, item in manifest.items():
        assert sha(dest / relative) == item['sha256']
    (dest / 'VALIDATION.json').write_text(json.dumps(dict(
        HOST=result, runtime_entry=False, NPU_qualification=False, model_tensors_used=False,
        format_ci='unavailable: pre-commit not installed; current environment retained',
        active64_production_ceiling_preserved=True, performance_result=None), indent=2) + '\n')
    print(json.dumps(dict(status='FROZEN_EXPORTED_HOST_ONLY401', **receipt)))


if __name__ == '__main__':
    main()
