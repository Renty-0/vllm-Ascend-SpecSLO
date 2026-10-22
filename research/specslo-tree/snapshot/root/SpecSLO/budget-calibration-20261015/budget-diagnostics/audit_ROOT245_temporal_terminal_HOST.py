"""Original terminal245 independent audit with the corrected temporal predicate."""
import hashlib
from pathlib import Path

ORIGINAL=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/audit_ROOT245_terminal_full_HOST.py')
ORIGINAL_SHA='589d93675f1ee628a66d0bf940f3bdfd22dbb6531610e7f4ac19490b8a12d77f'


def main():
    source=ORIGINAL.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_SHA:raise RuntimeError('unchanged original terminal245 audit')
    changes=[('ROOT245.independent-terminal-compute-audit.v1','ROOT245.independent-temporal-terminal-compute-audit.v1'),
        ('import audit245\n','import audit245_temporal_full_HOST as audit245\n'),
        ('from audit_reference245 import audit as audit_reference',
         'from audit245_temporal_reference_HOST import audit as audit_reference')]
    changed=source
    for before,after in changes:
        if changed.count(before)!=1:raise RuntimeError('one original terminal audit anchor '+before)
        changed=changed.replace(before,after)
    restored=changed
    for before,after in reversed(changes):restored=restored.replace(after,before)
    if restored!=source:raise RuntimeError('entire independent terminal audit inverse')
    namespace=dict(__file__=str(ORIGINAL),__name__='_ROOT245_temporal_terminal_audit_only')
    exec(compile(changed,str(Path(__file__).resolve())+'::original-independent-audit','exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
