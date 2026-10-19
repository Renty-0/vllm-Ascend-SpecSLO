"""Source-only full integration. Exact331 audit is still a hard prerequisite."""
from pathlib import Path
import re

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-M160-C128-F64-global192-full-v333'


def geometry(text):
    # Only reviewed old full geometry/proof files, never new owners/ranges or
    # numeric version identifiers. Protect every64-hex source identity.
    parts=re.split(r'([0-9a-f]{64})',text)
    values={'24':'32','96':'128','120':'160','408':'532'}
    return ''.join(part if re.fullmatch('[0-9a-f]{64}',part)else
        re.sub(r'\b(?:24|96|120|408)\b',lambda m:values[m.group()],part).replace('M120','M160').replace('S24','S32')
        for part in parts)


def main():
    for name in('extension333.py','scope333.py','audit333.py','audit_reference333.py','reference333.py','generation333.py'):
        path=P/name;path.write_text(geometry(path.read_text()))
    path=P/'runtime333.py';s=path.read_text().replace('"   from config209 import make_config", "   from config333 import make_config"','"  from config209 import make_config", "  from config333 import make_config"');path.write_text(s)
    path=P/'reference333.py';s=path.read_text();assert s.count('range(used,used+8)')==1
    s=s.replace('range(used,used+8)','range(used,used+16)').replace('append eight independent','append sixteen independent')
    path.write_text(s)
    path=P/'extension333.py';s=path.read_text()
    before="        return entry['install'](fast=True, host=True, transport=True, replay=True, repair=True, prefill=True)"
    after="        policy = entry['install'](fast=True, host=True, transport=True, replay=True, repair=True, prefill=True)\n        from capacity329 import bind\n        policy._ROOT333_actual_capacity = bind(policy, native)\n        return policy"
    assert s.count(before)==1;s=s.replace(before,after)
    s=s.replace('engine._branch_tail_state.prefix305_copies.width','engine._branch_tail_state.prefix326_copies.width')
    before='    from static_state333 import install\n'
    after="    from raw_controls326 import run as raw_controls\n    raw_proof = stage('ROOT333/fresh-raw160-controls', lambda:raw_controls(engine))\n    from static_state333 import install\n"
    assert s.count(before)==1;s=s.replace(before,after)
    before='    math = run(engine, native, stage, bank, output=folder)\n'
    after=before+'''    def strict_math():
        if engine.is_draft:return None
        if not all(c['hidden_bitwise_equal'] and c['KV_bitwise_equal'] and c['exact_greedy']
                   and c['graph_eager_KV_bitwise_equal'] for c in math['cases']):
            raise RuntimeError('ROOT333 all532 fresh target cases must be bitwise equal')
        return True
    stage('ROOT333/strict-all532-before-LIVE-scope',strict_math)
'''
    assert s.count(before)==1;s=s.replace(before,after)
    s=s.replace('proof = dict(activation=activation, bank=', 'proof = dict(fresh160_raw_operator=raw_proof, actual_capacity=policy._ROOT333_actual_capacity, activation=activation, bank=')
    path.write_text(s)
    (P/'static_state333.py').write_text('''"""Exact326 physical geometry with actual165 MC2 projection adapter."""
from static_state326 import ensure_zero, bind_checked_prepare
from static_state326 import install as original_install
from static_state326 import require_copy_bindings as original_bindings


def install(engine,native):
    activation=original_install(engine,native)
    engine._ROOT333_bridge=None
    engine._ROOT333_bridge_bindings=()
    if not engine.is_draft:
        from bridge330 import install as bridge_install
        from vllm_ascend.spec_decode.pearl import native_model
        bridge=bridge_install(engine.model,native_model)
        engine._ROOT333_bridge=bridge
        bindings=[]
        for layer in engine.model.layers:
            bindings.extend([(layer,'forward',layer.forward),
                             (layer.mlp,'forward_down_mc2',layer.mlp.forward_down_mc2)])
        for module in engine.model.modules():
            if isinstance(module,native_model.NativeRMSNorm):
                bindings.append((module,'forward_mc2',module.forward_mc2))
        engine._ROOT333_bridge_bindings=tuple(bindings)
    return dict(activation,actual_projection_rows165=True,FIA_rows160=True,
                new_profile160_claimed=False,full_instance_qualified=False)


def require_copy_bindings(engine):
    original_bindings(engine)
    if not engine.is_draft:
        if engine.model._specslo_bridge330 is not engine._ROOT333_bridge:
            raise RuntimeError('ROOT333 actual MC2 bridge instance changed')
        if engine._ROOT333_bridge.pending or engine._ROOT333_bridge.active_layer is not None:
            raise RuntimeError('ROOT333 MC2 bridge scope/ticket not closed')
        for owner,name,value in engine._ROOT333_bridge_bindings:
            if getattr(owner,name)is not value:
                raise RuntimeError('ROOT333 qualified native function binding changed')
    return True
''')
    # Full raw geometry evaluator expects this signature (not the physical
    # projection transform); reviewed original mapper values are updated.
    (P/'mapper333.py').write_text('''"""Declared physical M160/S32 envelope for original full raw auditor."""\nfrom plan326 import envelope\n''')
    for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
    print('SOURCE333 geometry/state/constructor bindings prepared; not frozen or launchable')

if __name__=='__main__':main()
