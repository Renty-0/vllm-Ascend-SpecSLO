"""Explicit original209/218 worker projection, with new declared math scope."""
import ast
import copy
import importlib.util
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
sys.path.extend([str(B/'tree-coalesced-live-refill-host-v345'),str(B/'tree-current-prepare-cost-eligibility-host-v312')])
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-prepare-host-v305', '/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-descriptors-host-v304'])
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-concrete-device-span-dual-cache-target-cost-v296', '/root/SpecSLO/budget-calibration-20261015/tree-spare-capacity-budget-host-v290', '/root/SpecSLO/budget-calibration-20261015/tree-request-conditional-risk-gain-host-v294', '/root/SpecSLO/budget-calibration-20261015/tree-prefix-precopy-dual-cachewrite-host-v288', '/root/SpecSLO/budget-calibration-20261015/tree-contiguous-prefix-span-copy-host-v292'])
OLD = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path[:0] = [str(P), str(OLD), str(DONOR)]
sys.path.extend([str(B/'tree-spare-branch-group-budget-full-v260'),str(B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'),str(B/'tree-static-q5-M120-full463-own-AR-v256'),str(B/'tree-static-q5-M120-node-math-v255'),str(B/'tree-static-q5-M128-full463-own-AR-v251'),str(B/'tree-static-q5-M128-projection-host-v249'),str(B/'tree-static-q5-immutable-index-cache-host-v247'),str(B/'tree-static-q5-long-prefix-origins-node-math-v246'),str(B/'tree-static-q5-projection-host-v243'),str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-contiguous-branch-tail-final-metadata-math-v234')])

sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-prefix-grid-target-replay-node-cost-v280', '/root/SpecSLO/budget-calibration-20261015/tree-prefix-grid-byte-KV-copy-v278', '/root/SpecSLO/budget-calibration-20261015/tree-genuine-alternate-draft-observed-component-v277', '/root/SpecSLO/budget-calibration-20261015/tree-private-alternate-draft-tail-host-v271', '/root/SpecSLO/budget-calibration-20261015/tree-alternate-chain-target-host-v269', '/root/SpecSLO/budget-calibration-20261015/tree-fused-byte-branch-KV-copy-v263', '/root/SpecSLO/budget-calibration-20261015/tree-fused-branch-KV-row-copy-v262', '/root/SpecSLO/budget-calibration-20261015/tree-paired-alternate-gain-host-v274', '/root/SpecSLO/budget-calibration-20261015/tree-genuine-branch-followup-host-v281', '/root/SpecSLO/budget-calibration-20261015/tree-root-alternate-live-policy-host-v276'])
_RUNTIME = None


def project_source(source):
    changes = (
        ("extension_module='extension209'", "extension_module='extension346'"),
        ("require(extension_module=='extension209'", "require(extension_module=='extension346'"),
        ('  require_ready(own_identity)', '  from check346 import ready as own_ready\n  own_ready()\n  require_ready(own_identity)'),
        (' readiness=require_ready();spec=', ' from check346 import ready as own_ready\n own_ready()\n readiness=require_ready();spec='),
        ("   policy=entry['install'](fast=True,host=True,transport=True,replay=True,repair=True,prefill=True)",
         '   policy=extension.install_entry(entry,native)'),
        ('     from scope209 import issue,report as scope_report', '     from scope346 import issue,report as scope_report'),
        ('     from trace209 import run_windows,warm_to_fixedpoint', '     from trace346 import run_windows,warm_to_fixedpoint'),
        ("     report['Native_final']=", "     report['postfull_independent_FIA_AR']=extension.postfull_AR(engine,native,stage,report['windows'],output_directory)\n     report['ROOT218_slot_preflight']=__import__('slot_preflight218').snapshot()\n     report['ROOT218_cached_parent_dispatch']=extension.cache_snapshot()\n     report['ROOT345_small_refill_current_component']=getattr(engine,'_ROOT345_actual_small_refill')\n     report['ROOT346_current_math']=getattr(engine,'_canonical346_actual_proof')\n     report['ROOT346_actual_geometry']=extension.snapshot(engine)\n     report['Native_final']="),
        ('   from audit209 import audit', '   from audit346 import audit'),
        ('   engine._v101_output_directory=output_directory',
         '   engine._v101_output_directory=output_directory\n   from fault_handler346 import install as register_fault_dump\n   register_fault_dump(engine,output_directory)'),
        ("parser.add_argument('--tag',required=True);parser.add_argument('--windows',default='32')", "parser.add_argument('--tag',required=True);parser.add_argument('--windows',default='48')"),
        (" windows=parse_windows(args.windows);require(windows==(32,),'only active32 in this new arm')", " from window346 import parse_windows\n windows=parse_windows(args.windows);require(windows==(48,),'explicit active48 in this new arm')"),
        ('   if engine is not None:engine.exit()', 
         "   from shutdown234 import finish\n   exited,exit_wait=finish(engine)\n   save(output,'worker-exit-wait.json',exit_wait)"),
        ("   save(output,'worker-exit.json',dict(closed=bool(engine and engine._closed),all_exited=bool(engine and all(not p.is_alive() for p in engine._processes))))",
         "   save(output,'worker-exit.json',exited)"),
    )
    projected = source
    for before, after in changes:
        if projected.count(before) != 1:
            raise RuntimeError('ROOT346 exact original runtime anchor '+before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after in reversed(changes):
        if inverse.count(after) != 1:
            raise RuntimeError('ROOT346 exact inverse runtime anchor '+after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT346 entire original runtime inverse')
    return projected, changes


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    from check346 import ready
    ready()
    path = DONOR/'runtime209.py'
    spec = importlib.util.spec_from_file_location('_ROOT346_original_runtime209', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = dict(vars(module), __name__='ROOT346.projected-original-runtime209')
    source = path.read_text()
    projected, changes = project_source(source)
    names = {'save', 'worker', 'main'}
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::explicit-original-runtime-projection', 'exec'), namespace)
    actual_worker = namespace['worker']
    namespace['worker'] = worker
    _RUNTIME = dict(worker=actual_worker, main=namespace['main'], save=namespace['save'],
                    changes=changes, projected_source=projected)
    return _RUNTIME


def save(*args, **kwargs):
    return build_runtime()['save'](*args, **kwargs)


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
