"""Uninstalled B3 composite draft prototype; no device permission is issued.

Reuse the original packed-suffix/root and explicit top2 head. Predict three
levels with a two-step tail, keeping the four-level public row shape and the
original FastV5 known-KV ledger. A fresh device comparison is still required.
"""
import ast
import hashlib
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent.parent / 'tree-high-normal-cap4-native-families-v148'
STEPS = 3
TAIL_STEPS = STEPS - 1
CANDIDATES = 2 * STEPS
MODE = 'normal-packed-root-seed-top2-B3-v410'


def sources():
    text = (BASE / 'root_seed.py').read_text()
    install = next(n for n in ast.parse(text).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'install')
    node = next(n for n in install.body
                if isinstance(n, ast.FunctionDef) and n.name == 'draft')
    before = ast.get_source_segment(text, node)
    edits = (
        ('or min(4, max(int(p.candidate_budget) for p in plans)) != 4',
         'or max(int(p.candidate_budget) for p in plans) != STEPS'),
        ('ids, [0]*len(ids), 3, head_rows)', 'ids, [0]*len(ids), TAIL_STEPS, head_rows)'),
        ('rows.append([pair[0] for pair in tokens]+[pair[1] for pair in tokens])',
         'rows.append([pair[0] for pair in tokens]+[0]+[pair[1] for pair in tokens]+[0])'),
        ('scores.append([pair[0] for pair in confidence]+[pair[1] for pair in confidence])',
         'scores.append([pair[0] for pair in confidence]+[0.]+[pair[1] for pair in confidence]+[0.])'),
        ('The last predicted m4 is still deferred', 'The last predicted m3 is still deferred'),
        ("counts['chain_real_queries'] += 3*len(ids)",
         "counts['chain_real_queries'] += TAIL_STEPS*len(ids)"),
        ("counts['chain_padded_queries'] += 3*head_rows",
         "counts['chain_padded_queries'] += TAIL_STEPS*head_rows"),
        ("counts['generated_candidates'] += 8*len(ids)",
         "counts['generated_candidates'] += CANDIDATES*len(ids)"),
        ("counts['root_seed_model_stages'] += 4", "counts['root_seed_model_stages'] += STEPS"),
        ('draft_confidence=torch.tensor([sum(row)/8 for row in scores])',
         'draft_confidence=torch.tensor([sum(row)/CANDIDATES for row in scores])'),
        ('query_count=physical+3*head_rows', 'query_count=physical+TAIL_STEPS*head_rows'),
        ('graph_padding_query_count=physical-len(packed.tokens)+3*(head_rows-len(ids))',
         'graph_padding_query_count=physical-len(packed.tokens)+TAIL_STEPS*(head_rows-len(ids))'),
        ('model_calls=2, graph_calls=2, model_stages=4',
         'model_calls=2, graph_calls=2, model_stages=STEPS'),
    )
    after = before
    for old, new in edits:
        if after.count(old) != 1:
            raise RuntimeError('ROOT410 exact original composite anchor: ' + old)
        after = after.replace(old, new)
    inverse = after
    for old, new in reversed(edits):
        if inverse.count(new) != 1:
            raise RuntimeError('ROOT410 ambiguous declared inverse')
        inverse = inverse.replace(new, old)
    if inverse != before:
        raise RuntimeError('ROOT410 entire original callback inverse')
    return before, after


def build(native, previous):
    """Return a private callback; do not install it on a Native engine class."""
    if not callable(previous):
        raise TypeError('ROOT410 explicit existing fallback callback required')
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))
    import torch
    import root_seed
    import fast_tree_draft_v5 as fast
    namespace = dict(vars(root_seed), native=native, original=previous,
                     fast=fast, torch=torch, STEPS=STEPS, TAIL_STEPS=TAIL_STEPS,
                     CANDIDATES=CANDIDATES, MODE=MODE)
    exec(compile(sources()[1], __file__ + '::private-packed-B3', 'exec',
                 dont_inherit=True), namespace)
    return namespace['draft']


def descriptor():
    before, after = sources()
    return dict(original_root_seed_file_sha256=hashlib.sha256((BASE / 'root_seed.py').read_bytes()).hexdigest(),
                original_callback_sha256=hashlib.sha256(before.encode()).hexdigest(),
                projected_callback_sha256=hashlib.sha256(after.encode()).hexdigest(),
                original_callback_complete_inverse=True, public_row_depth=4,
                predicted_levels=STEPS, padded_sibling_rows=8,
                confidence_denominator=CANDIDATES, model_stages=STEPS,
                NPU_qualified=False, serving_installed=False,
                Raw=None, Goodput=None)
