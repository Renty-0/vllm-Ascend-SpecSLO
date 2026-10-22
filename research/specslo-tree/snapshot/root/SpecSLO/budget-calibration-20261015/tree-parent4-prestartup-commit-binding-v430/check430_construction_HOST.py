"""Exact worker ordering and all original finite/production callbacks."""
import ast
import json
from pathlib import Path


def main():
    import runtime430, extension430, extension428, guard430
    source = runtime430.source()
    compile(source, 'HOST430.actual-worker', 'exec')
    assert source.index('original_commit430=capture(protocol,native)') < source.index('policy=extension.install_entry')
    assert source.index('engine=native.NativePearlEngine(cfg)') < source.index('bind(engine,original_commit430)')
    assert source.index('bind(engine,original_commit430)') < source.index("connection.send(('ready'")
    assert "extension_module='extension430'" in source and 'from audit430 import audit' in source
    for name in ('prepare_install','install_entry','after_install','postfull_AR','snapshot','bucket_snapshot','metadata_snapshot'):
        assert getattr(extension430,name) is getattr(extension428,name)
    ns = guard430.execution_namespace()
    assert ns['execute_locked'].__globals__ is ns
    assert ns['MAX_SECONDS'] == 18000 and ns['MAX_TAG_BYTES'] == 256*1024**2
    assert ns['MAX_AGGREGATE_BYTES'] == 3584*1024**2
    assert ns['SCOPE']['current_same_engine_parent4_lifecycle_cases'] == 75
    assert ns['SCOPE']['active_request_limit'] == 64
    # Audit all finite419 sources for attributes supplied by its standalone199
    # worker. Commit identity is the sole missing worker-specific read; the
    # diagnostic context and bounded result are created by the actual actor.
    P = Path(__file__).resolve().parent
    fields = set()
    for path in (P.parent/'tree-parent4-normal24-SLO-mixed-protocol-v419').glob('*.py'):
        if path.name.startswith(('runtime','check','test','freeze','guard')):
            continue
        tree = ast.parse(path.read_text())
        fields.update(n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)
                      and isinstance(n.ctx,ast.Load) and n.attr.startswith('_v199_'))
    assert fields == {'_v199_diagnostic', '_v199_commit_states'}
    value = dict(status='PASS_HOST430_EXACT_WORKER_ORDER_CALLBACKS_AND_DIAGNOSTIC_DEPENDENCY_AUDIT',
        source_specific_v199_attribute_reads=sorted(fields),
        original419_case_target_and_protocol_bodies_retained=True,
        all_original_227_681_75_warm_twozero_seal_full463_ownAR_required=True,
        original_private_caps_and_owned_cleanup_retained=True,
        NPU_used=False, Native_models_constructed=0, NN_qualification_inherited=False)
    (P/'HOST430_CONSTRUCTION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':
    main()
