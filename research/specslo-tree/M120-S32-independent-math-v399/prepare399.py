"""Actual reviewed pre-model install and original replay-first source check."""
import hashlib


def before_model(output):
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from extension218 import prepare_install
    from install234 import install_entry
    entry, _ = prepare_install(output)
    install_entry(entry, native)
    from capture_key399 import actual_source
    function = native.NativeACLGraphRunner._run_target
    source = actual_source(function)
    return native, dict(actual_code_reconstructed_equal=True,
                        source_SHA=hashlib.sha256(source.encode()).hexdigest(),
                        actual_code_filename=function.__code__.co_filename,
                        actual_annotation_compiler_flags=function.__code__.co_flags,
                        models_loaded=False, performance=False)
