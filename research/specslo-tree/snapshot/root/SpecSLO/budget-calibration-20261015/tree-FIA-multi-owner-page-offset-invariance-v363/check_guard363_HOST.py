"""Source-only guard, scope, mapping and retained parent provenance checks."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


P = Path(__file__).resolve().parent
B = P.parent


def main():
    output = P / 'HOST363_GUARD_RESULT.json'
    assert not output.exists()
    source = (P / 'guard363.py').read_text()
    tree = ast.parse(source)
    values = {node.targets[0].id: node.value for node in tree.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    assert ast.literal_eval(values['MAX_SECONDS']) == 600
    assert ast.unparse(values['MAX_TAG_BYTES']) == '64 * 1024 ** 2'
    assert ast.unparse(values['MAX_AGGREGATE_BYTES']) == '256 * 1024 ** 2'
    assert 'synthetic-FIA-Q5-S24-multi-owner-page-offset-invariance' in source
    assert 'ROOT.V363.synthetic-FIA-page-offset-supervisor.v1' in source
    assert 'groups=[24]' in source and 'KV_heads=[2, 3]' in source and 'rows=1920' in source
    assert 'full463=False' in source and 'models_loaded=False' in source and 'formal_Goodput=False' in source
    assert 'flock' in source and 'execute_locked' in source and 'parent.donor_guard()' in source
    assert 'from audit363 import audit' in source and "host['probe_sha256'] == sha(HERE / 'probe363.py')" in source
    compile(source, str(P / 'guard363.py'), 'exec')
    startup = (P / 'startup363.sh').read_text()
    assert str(P) in startup and 'probe363.py' in startup and '--logical-device 1' in startup
    subprocess.run(['/bin/bash', '-n', str(P / 'startup363.sh')], check=True)
    runtime = (B / 'serial-v1/runtime.sh').read_text()
    assert 'SPECSLO_VISIBLE_DEVICES:-2,5,6,7' in runtime
    assert 'map-current.py "$SPECSLO_VISIBLE_DEVICES"' in runtime
    assert 'export ASCEND_RT_VISIBLE_DEVICES=$container_runtime_devices' in runtime
    host = json.loads((P / 'HOST363_RESULT.json').read_text())
    assert host['probe_sha256'] == hashlib.sha256((P / 'probe363.py').read_bytes()).hexdigest()
    assert host['NPU_used'] is False and host['counts']['cases'] == 640
    value = dict(status='PASS_HOST_original_owned_synthetic_guard363_and_explicit_four_card_mapping',
                 deadline_seconds=600, tag_private_limit_MiB=64, aggregate_limit_MiB=256,
                 actual_logical_target_device=1, original_requested_cards=[2, 5, 6, 7],
                 serial_runtime_mapping_preserved=True, source_gate_matches_current_HOST_probe=True,
                 all_model_and_serving_qualification_false=True, NPU_used=False, Goodput=None)
    with output.open('x') as saved:
        json.dump(value, saved, indent=2)
        saved.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
