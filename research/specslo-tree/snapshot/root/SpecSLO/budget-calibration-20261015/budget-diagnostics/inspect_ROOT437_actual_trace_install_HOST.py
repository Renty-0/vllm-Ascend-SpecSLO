"""Exercise the real trace installation path without constructing an engine."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-nonempty-qualified-RPC-full-v436'
F = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
REPO = F/'source/vllm-ascend-hust'


def inside():
    source=(D/'run_ROOT436_actual_CPU_HOST.py').read_text()
    # Reuse its exact retained startup settings, then run this independent probe.
    start=source.index('    lines = [line for line in (P /')
    end=source.index('    import runpy', start)
    setup='def setup():\n'+source[start:end]+'\n'
    ns=dict(globals())
    exec(setup,ns)
    ns['setup']()
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    parent=require_ready()
    with activate_local_diagnostic(dict(action=PARENT_ACTION,identity=parent['parent']['identity'])):
        import extension436, trace432
        folder=Path(tempfile.mkdtemp(prefix='HOST437-trace-install-',dir=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'))
        entry,_=extension436.prepare_install(folder)
        from vllm_ascend.spec_decode.pearl import native_engine as native, native_graph as graph
        def blocked(*args,**kwargs):
            raise AssertionError('HOST437 no Native construction')
        native.NativePearlEngine.__init__=blocked
        from original_tree_cycle_v75 import capture_original_protocol_before_startup
        from lifecycle430 import capture
        protocol=capture_original_protocol_before_startup(native)
        capture(protocol,native)
        extension436.install_entry(entry,native)
        import tail_hidden_observer,live_decode_patch
        tail_hidden_observer.install(graph)
        extension436.after_install(native,graph,live_decode_patch)
        fn,state=trace432.namespace()['make_trace'](64,entry=entry,output=folder)
        actual_install=fn.__globals__['install']
        actual_install(native,lambda actual:None)
        installed=native.NativePearlEngine._generate_spec_rhythm_tree_decode
        delegate=dict(zip(installed.__code__.co_freevars,installed.__closure__))['decode_impl'].cell_contents
        import inspect
        delegate=inspect.unwrap(delegate)
        def codes(code):
            yield code
            for c in code.co_consts:
                if hasattr(c,'co_name'):yield from codes(c)
        nested=[c for c in codes(delegate.__code__)if c.co_name=='admit_available']
        result=dict(status='HOST437_ACTUAL_TRACE_INSTALL_INSPECTED',
            actual_trace_globals_install_is_closure=actual_install.__closure__ is not None,
            projected_source_has_witness431='from witness431 import' in state['source'],
            compiled_admission_has_witness431=bool(nested)and'witness431' in nested[0].co_names,
            compiled_admission_names=[list(c.co_names)for c in nested],
            delegate_name=delegate.__qualname__,delegate_filename=delegate.__code__.co_filename,
            delegate_globals=delegate.__globals__.get('__file__'),delegate_freevars=delegate.__code__.co_freevars,
            all_nested_code_names=[c.co_name for c in codes(delegate.__code__)],
            closure_values=[str(c.cell_contents)[:300]for c in delegate.__closure__ or ()],
            trace_module=fn.__globals__.get('__file__'),
            canonical_patch_module=trace432.namespace()['canonical_patch'].__module__,
            NPU_used=False,Native_constructor_calls=0)
        (D/'ROOT437_ACTUAL_TRACE_INSTALL_INSPECTION_HOST.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))


def main():
    tree=ast.parse((D/'run_ROOT436_actual_CPU_HOST.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    env_nodes=[n for n in main.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='environment'for t in n.targets)
               or isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='environment.update']
    ns={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=env_nodes,type_ignores=[])),'HOST437.retained-env','exec'),ns)
    argv=['/bin/bash',str(B/'serial-v1/runtime.sh'),'/root/miniconda3/envs/vllm-hust-dev/bin/python','-B',
        str(REPO/'examples/run_with_vllm_ascend_extension.py'),'--extension',
        str(REPO/'vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'),
        '--required-op','allreduce_add_rmsnorm_chained',str(Path(__file__).resolve()),'--inside']
    r=subprocess.run(argv,cwd=REPO,env=ns['environment'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (D/'ROOT437_ACTUAL_TRACE_INSTALL_INSPECTION_HOST.log').write_text(r.stdout)
    print(r.stdout[-7000:])
    if r.returncode:raise SystemExit(r.returncode)


if __name__=='__main__':
    inside()if '--inside'in sys.argv else main()
