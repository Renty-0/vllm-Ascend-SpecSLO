"""Fresh200 same-engine strict cases; compact retained bank descriptors."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activation(engine, native):
    if (engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed
            or engine._canonical257_phase != 'qualification'
            or engine._uniform_KV_bucket409_proof['cases'] != 681):
        raise RuntimeError('ROOT409 full prior227+681 and released cache before wide math')
    if not engine.is_draft and not engine._branch_tail_state.uniform_KV_bucket409_enabled:
        raise RuntimeError('ROOT409 actual uniform dual target projection required')
    return dict(existing_dual_projection=True, reinstalled=False, Native_flags_assigned=False)


def descriptor(folder, rank, value):
    path = Path(folder) / f'rank{rank}-bank399.json'
    if not path.is_file():
        raise RuntimeError('ROOT409 actual complete bank file must precede compact descriptor')
    return dict(extent=value['extent'], path=str(path.resolve()), sha256=sha(path),
                cases=40, producer_rank=rank)


def projected_run():
    import probes399
    from projection409 import project_function
    return project_function(probes399.run, (
        ('from static_state399 import install', 'from s32_409 import activation as install'),
        ('reports.append(value)', 'reports.append(_ROOT409_descriptor(folder, engine.rank, value))'),
    ), {'_ROOT409_descriptor': descriptor})


def qualify(engine, native, stage, output):
    root = Path(output) / 'target-S32-independent-math409'
    stage('ROOT409/wide-private-output', lambda: root.mkdir(exist_ok=False) if engine.rank == 0 else None)
    fixture = json.loads((Path(__file__).resolve().parent.parent /
        'tree-M120-S32-frugal-independent-target-host-v399/FIXTURE399.json').read_text())
    report = projected_run()(engine, native, stage, root, fixture)
    report['model_dtype'] = str(engine.model.embed_tokens.weight.dtype)
    path = root / f'rank{engine.rank}-numeric.json'
    with path.open('x') as stream:
        json.dump(report, stream, separators=(',', ':'), allow_nan=False)
    proof = dict(path=str(path.resolve()), sha256=sha(path), cases=200,
                 same_model_instance=True, independent_reference_S=24,
                 actual_target_S=32, Native_flags_assigned=False)
    engine._S32_409_current_proof = proof
    engine._S32_409_proof_owner = engine
    return proof


def load_banks(value, root):
    root = Path(root).resolve()
    banks = []
    for record in value['banks']:
        path = Path(record['path']).resolve()
        if (not path.is_relative_to(root) or record['producer_rank'] != value['rank']
                or record['cases'] != 40 or sha(path) != record['sha256']):
            raise RuntimeError('ROOT409 exact current complete bank digest and rank required')
        bank = json.loads(path.read_text())
        if bank['extent'] != record['extent']:
            raise RuntimeError('ROOT409 bank visible extent differs from compact descriptor')
        banks.append(bank)
    return dict(value, banks=banks)


def audit(root):
    from projection409 import project_function
    from audit399 import audit as original
    anchor = "        value = json.loads((folder/f'rank{rank}-numeric.json').read_text())"
    function = project_function(original, ((anchor, anchor + '\n        value = _ROOT409_load(value, folder)'),),
                                {'_ROOT409_load': load_banks})
    return function(root)
