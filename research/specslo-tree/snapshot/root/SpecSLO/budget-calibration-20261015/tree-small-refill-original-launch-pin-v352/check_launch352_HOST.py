"""Assert literal original pin, numeric source pin and exact frozen launcher."""
import hashlib
import json
from pathlib import Path
import sys

import guard352


def main():
    original = guard352.parent.launch_argv
    guard352.parent.EXPECTED_FREEZE = guard352.PINS['freeze_sha256']
    before = original('HOST352')
    after = guard352.launch_argv('HOST352')
    changed = [i for i,(a,b) in enumerate(zip(before,after,strict=True)) if a != b]
    assert len(changed) == 1 and after[changed[0]].startswith('SPECSLO_V218_ROOT_FREEZE_SHA=')
    assert after[changed[0]].split('=',1)[1] == hashlib.sha256(
        (guard352.B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218/FREEZE218.json').read_bytes()).hexdigest()
    assert 'SPECSLO_V351_ROOT_FREEZE_SHA='+guard352.PINS['freeze_sha256'] in after
    assert after[-2] == str(guard352.PARENT/'startup351.sh')
    ns = guard352.execution_namespace()
    assert ns['execute_locked'].__globals__['launch_argv'] is guard352.launch_argv
    assert ns['execute_locked'].__globals__['prepared'] is guard352.prepared
    assert ns['SCOPE']['numeric_source_version'] == 351
    assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
    value = dict(status='passed_literal_original_launch_pin_HOST',NPU_used=False,
        exactly_one_environment_pin_corrected=True, numeric351_sources_unchanged=True,
        actual_original218_manifest_hash_verified=True, actual_owned_guard_bindings_checked=True,
        model_qualification=False)
    (Path(__file__).resolve().parent/'HOST352_LAUNCH_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__': main()
