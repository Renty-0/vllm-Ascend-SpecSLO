"""Actual typed wide policy; no independent policy/model replacement."""
from wide409 import shape
from wide409 import snapshot as original_snapshot


def snapshot(engine):
    result = original_snapshot(engine)
    for key in ('production_runtime_installed', 'NPU_qualified', 'Raw', 'Goodput'):
        result.pop(key, None)
    binding = getattr(engine, '_typed404_binding', None)
    controls = getattr(engine, '_root409_controls', None)
    if binding is not None:
        if controls is None or binding.policy is not controls.policy or binding.method is not controls.candidate_shape:
            raise RuntimeError('ROOT409 actual engine-owned typed control binding changed')
        import normal_admission_v140
        result['actual_original_normal_admission'] = normal_admission_v140.snapshot(binding.policy)
        result['actual_policy_object_preserved'] = True
    result['actual_LIVE_policy'] = 'normal<=32; wide C88 and at most24 B3 owners; original B3 eager'
    result['model_qualification_inherited'] = False
    return result
