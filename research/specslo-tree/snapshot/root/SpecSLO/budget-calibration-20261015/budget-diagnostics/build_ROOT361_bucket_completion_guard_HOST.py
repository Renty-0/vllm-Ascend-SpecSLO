"""Build additive guard/startup only; never launch or mutate original344."""
from pathlib import Path
import re


B = Path('/root/SpecSLO/budget-calibration-20261015')
OLD = B / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'
NEW = B / 'tree-Q5-bucket-completion-and-paired-mode-cost-v361'


def checked(source, changes):
    projected = source
    for before, after, count in changes:
        if projected.count(before) != count:
            raise RuntimeError('exact361 additive anchor required: ' + before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after, _ in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('complete original344 guard/startup inverse required')
    if sorted(re.findall(r'\b[0-9a-f]{64}\b', projected)) != sorted(re.findall(r'\b[0-9a-f]{64}\b', source)):
        raise RuntimeError('all original literal source pins must be byte identical')
    return projected


def main():
    if (NEW / 'FREEZE361.json').exists():
        raise RuntimeError('cannot edit a frozen361 candidate')
    source = (OLD / 'guard344.py').read_text()
    guard = checked(source, (
        ("B/'tree-Q5-uniform-KV-bucket-branch-math-cost-v343'", "B/'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'", 1),
        ('import guard343 as parent', 'import guard344 as parent', 1),
        ("PARENT/'SOURCE343_RECEIPT.json'", "PARENT/'SOURCE344_RECEIPT.json'", 1),
        ("HERE/'FREEZE344.json'", "HERE/'FREEZE361.json'", 3),
        ("HERE/'SOURCE344.tar.gz'", "HERE/'SOURCE361.tar.gz'", 2),
        ('MAX_SECONDS = 1800', 'MAX_SECONDS = 3600', 1),
        ('MAX_TAG_BYTES = 96*1024**2', 'MAX_TAG_BYTES = 192*1024**2', 1),
        ('MAX_AGGREGATE_BYTES = 2560*1024**2', 'MAX_AGGREGATE_BYTES = 3584*1024**2', 1),
        ("'SPECSLO_V344_ROOT_FREEZE_SHA='+EXPECTED_FREEZE,",
         "'SPECSLO_V361_ROOT_FREEZE_SHA='+EXPECTED_FREEZE,\n        'SPECSLO_V344_ROOT_FREEZE_SHA='+sha(PARENT/'FREEZE344.json'),", 1),
        ("HERE/'startup344.sh'", "HERE/'startup361.sh'", 1),
        ('from audit344 import audit', 'from audit361 import audit', 1),
        ('ROOT.V344.original-root-linear-target-supervisor.v1', 'ROOT.V361.original-root-linear-target-supervisor.v1', 1),
        ('ROOT344 supervisor received', 'ROOT361 supervisor received', 1)))
    startup = checked((OLD / 'startup344.sh').read_text(), (
        (str(OLD / 'check344.py'), str(NEW / 'check361.py'), 1),
        (str(OLD / 'runtime344.py'), str(NEW / 'runtime361.py'), 1),
        ('export PYTHONPATH=', 'export PYTHONPATH=' + str(NEW) + ':', 1)))
    for name, value in (('guard361.py', guard), ('startup361.sh', startup)):
        path = NEW / name
        if path.exists():
            raise RuntimeError('do not overwrite retained361 source: ' + str(path))
        path.write_text(value)
    print('additive361 guard and startup constructed; no NPU launch')


if __name__ == '__main__':
    main()
