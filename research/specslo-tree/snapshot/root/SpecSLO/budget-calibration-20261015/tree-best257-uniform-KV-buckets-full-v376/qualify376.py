"""Fresh same-model additional math; no component timing or graph flag writes."""
import json
import hashlib
from pathlib import Path


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def qualify(engine, native, stage, output):
    from bank257 import build
    from oracle376 import run
    from oracle257 import cases
    from dispatch376 import audit_calls
    from prefixes376 import mid_prefixes, short_prefixes
    fixture = json.loads((Path(__file__).parent.parent /
        'tree-static-q5-M120-SLO-first-full463-own-AR-v257/FIXTURE257.json').read_text())
    target = not engine.is_draft
    if engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed:
        raise RuntimeError('ROOT376 current original227 bank released before additional qualification')
    if target:
        engine._branch_tail_state.uniform_KV_bucket376_enabled = True
    reports = []
    root = Path(output) / 'target-uniform-KV-bucket-math376'
    stage('ROOT376/create-output', lambda: root.mkdir(exist_ok=False) if engine.rank == 0 else None)
    for name, prefixes in [('mixed', fixture['prefixes']),
            ('mid', mid_prefixes(fixture['prefixes'])),
            ('short', short_prefixes(fixture['prefixes']))]:
        lane = lambda label, fn: stage('ROOT376/' + name + '/' + label, fn)
        folder = root / name
        lane('create-output', lambda: folder.mkdir(exist_ok=False) if engine.rank == 0 else None)
        bank = build(engine, native, lane, prefixes)
        before = len(engine._branch_tail_state.static_graph_calls)
        math = run(engine, native, lane, bank, folder)
        calls = list(engine._branch_tail_state.static_graph_calls[before:])
        dispatch = lane('audit-real-dispatch', lambda: audit_calls(cases(), calls, prefixes) if target else None)
        proof = dict(bank=bank['evidence'], prefix_lengths=[len(p) for p in prefixes],
            math=math, actual_graph_calls=calls, dispatch=dispatch)
        saved = folder / f'rank{engine.rank}-proof376.json'
        lane('save-fresh-math', lambda: save(saved, proof))
        # Retain one complete proof per bank. Final worker reports refer to its
        # digest instead of storing several copies of all KV hashes and maps.
        reports.append(dict(path=str(saved), sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
            cases=227, name=name, neural_prefixes=72 if target else 0))
        lane('release-bank', lambda: engine._release_cache() if target else None)
    proof = dict(cases=681, independent_prefills=216 if target else 0, banks=reports,
        same_model_instance=True, original227_FULL_math_preserved=True,
        original_best257_draft_and_branch_copy_unchanged=True, native_flags_assigned=False,
        production_qualified=False, whole463_qualified=False, Raw=None, Goodput=None, adopted=False)
    stage('ROOT376/save-complete-fresh-math', lambda: save(root / f'rank{engine.rank}-proof376.json', proof))
    engine._uniform_KV_bucket376_proof = proof
    return proof
