"""Exact385 worker/main lifecycle, ordinary original constructor profile."""
import ast,copy
from host_env457 import setup
setup()
import runtime385 as original
ACTION='M80-own-geometry-whole-model-reference-cost'
_RUNTIME=None
def worker_source():
 before=original.worker_source();after=before
 changes=(('from check385 import ready','from check457 import ready',2),
  ('from probe385 import run','from probe457 import run',1),
  ('real-M120-deferred-chain-original-graph-qualifier',ACTION,2))
 for a,b,n in changes:
  if after.count(a)!=n:raise RuntimeError('ROOT457 exact385 worker/main anchor '+a)
  after=after.replace(a,b)
 inverse=after
 for a,b,n in reversed(changes):
  if inverse.count(b)!=n:raise RuntimeError('ROOT457 exact worker inverse anchor '+b)
  inverse=inverse.replace(b,a)
 if inverse!=before:raise RuntimeError('ROOT457 entire original385 worker/main inverse')
 return after
def build_runtime():
 global _RUNTIME
 if _RUNTIME is not None:return _RUNTIME
 from check457 import ready
 ready()
 # This is the exact original385 base namespace. Avoid rebuilding unrelated
 # intermediate diagnostics; every original source/archive is still rehashed.
 import runtime311
 ns=dict(vars(runtime311),__name__='ROOT457.original385-lifecycle-with-own80-diagnostic')
 nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
 if len(nodes)!=2:raise RuntimeError('ROOT457 both complete original lifecycle functions')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::exact385-own80-diagnostic','exec'),ns)
 actual=ns['worker'];ns['worker']=worker;_RUNTIME=dict(worker=actual,main=ns['main']);return _RUNTIME
def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
