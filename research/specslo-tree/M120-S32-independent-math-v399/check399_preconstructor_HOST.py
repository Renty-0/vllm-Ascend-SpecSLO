"""Actual finite install prefix with model construction and NPU init blocked."""
import json
from pathlib import Path
import tempfile

from bootstrap399 import install
install()


def main():
    import torch_npu
    from vllm_ascend.spec_decode.pearl import native_engine as native
    blocked_calls = []

    def blocked(*args, **kwargs):
        blocked_calls.append('blocked')
        raise AssertionError('HOST399 may not initialize an NPU or construct models')

    native.NativePearlEngine.__init__ = blocked
    torch_npu.npu._lazy_init = blocked
    from check398 import ready
    ready()
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    readiness = require_ready()
    with activate_local_diagnostic(dict(action=PARENT_ACTION, identity=readiness['parent']['identity'])):
        from prepare399 import before_model
        donor = Path('/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209')
        output = Path(tempfile.mkdtemp(prefix='HOST399-before-model-', dir=donor))
        actual, proof = before_model(output)
        assert actual is native and proof['actual_code_reconstructed_equal'] is True
        from startup_config_v132 import make_config
        cfg = make_config()
        assert cfg.max_num_seqs == 64
        import runtime399
        import probes399
        import oracle399
        import audit399
        assert runtime399.worker.__module__ == 'runtime399'
        assert not blocked_calls
        result = dict(status='PASS_HOST399_ACTUAL_FINITE_PREMODEL_INSTALL_AND_REPLAY_SOURCE',
                      NPU_used=False, model_constructor_calls=0, NPU_lazy_init_calls=0,
                      actual_original_replay_dispatch=proof, output=str(output),
                      configured_max_num_seqs=64, candidate_S=32, reference_S=24,
                      original_full_components_executed=False, production_permission=False,
                      model_qualification=False, Raw=None, Goodput=None)
        (Path(__file__).resolve().parent/'HOST399_PRECONSTRUCTOR_RESULT.json').write_text(
            json.dumps(result, indent=2)+'\n')
        print(json.dumps({key:result[key] for key in ('status', 'NPU_used', 'model_constructor_calls')}))


if __name__ == '__main__':
    main()
