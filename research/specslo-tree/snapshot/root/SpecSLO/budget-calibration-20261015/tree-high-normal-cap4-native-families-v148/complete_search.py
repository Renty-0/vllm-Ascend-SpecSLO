"""HOST min(M,S) candidate among exactly44 exported existing Native keys."""
import json
from pathlib import Path

P=Path(__file__).resolve().parent
EXPORT=json.loads((P/'EXACT_QUALIFIED_KEY_INVENTORY.json').read_text())
QUALIFIED_KEYS=tuple(sorted(tuple(row) for row in EXPORT['geometries']))
if len(QUALIFIED_KEYS)!=44 or len(set(QUALIFIED_KEYS))!=44:
    raise RuntimeError('Need exactly the exported44 current qualified target keys')


def _lengths(real_count,real_queries,tokens,segments):
    dummy=segments-real_count
    extra=tokens-real_queries
    if dummy==0 and extra==0:return []
    if not 1<=dummy<=32 or not 2*dummy<=extra<=9*dummy:return None
    return [extra//dummy+int(i<extra%dummy) for i in range(dummy)]


def envelope(real_count,real_queries,family):
    if family not in ('causal','full9'):
        raise ValueError('Target family must describe the actual real plans')
    if (not 1<=real_count<=32
            or not 2*real_count<=real_queries<=min(6*real_count,real_count+96,128)
            or (family=='causal' and real_queries>5*real_count)
            or (family=='full9' and real_queries<2*real_count+4)):
        raise ValueError('Target geometry is outside the admitted budget/family domain')
    for tokens,segments,entry_family in QUALIFIED_KEYS:
        if entry_family!=family:continue
        lengths=_lengths(real_count,real_queries,tokens,segments)
        if lengths is not None:return tokens,segments,lengths
    raise ValueError('No admitted bounded bucket preserves the real target geometry')
