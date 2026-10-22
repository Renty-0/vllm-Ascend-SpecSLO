"""Same original API receiver, qualification-only21600s immutable config view."""
import ast,copy
from pathlib import Path
import wait436 as original
_NS=None
def namespace():
    global _NS
    if _NS is not None:return _NS
    source=Path(original.__file__).read_text();changes=(
        ('QUALIFICATION_TIMEOUT_SECONDS=18000','QUALIFICATION_TIMEOUT_SECONDS=21600'),
        ('outer_supervisor_total_limit_seconds=18000','outer_supervisor_total_limit_seconds=21600'))
    projected=source
    for before,after in changes:
        if projected.count(before)!=1:raise RuntimeError('ROOT462 exact original qualification wait anchor')
        projected=projected.replace(before,after)
    inverse=projected
    for before,after in reversed(changes):inverse=inverse.replace(after,before)
    if inverse!=source:raise RuntimeError('ROOT462 entire unchanged original receiver source inverse')
    ns=dict(__name__='ROOT462.original-qualification-only-wait',__file__=__file__)
    exec(compile(ast.parse(projected),__file__+'::original-wait-with-explicit21600s-bound','exec'),ns)
    _NS=ns;return ns
def receive_qualification(engine,operation):return namespace()['receive_qualification'](engine,operation)
def snapshot():return namespace()['snapshot']()
