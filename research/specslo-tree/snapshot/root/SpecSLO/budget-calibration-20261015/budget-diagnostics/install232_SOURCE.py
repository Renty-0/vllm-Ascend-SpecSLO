"""Diagnostic-only installation; never claims a full production capability."""
def install_entry(entry,native):
    import host_target_metadata as host
    from root_padding_forward231 import host_installer
    original=host.install;host.install=host_installer(original)
    try:policy=entry['install'](fast=True,host=True,transport=True,replay=True,repair=True,prefill=False)
    finally:host.install=original
    return policy

def activate(native):
    """Call after constructor and worker readiness; explicit diagnostic epoch."""
    from padding231 import pad
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    old=native._dynamic_tree_make_host_metadata
    def full(engine,*args,**kwargs):
        kwargs=dict(kwargs,allow_causal_fast_path=False)
        return old(engine,*args,**kwargs)
    native._dynamic_tree_make_host_metadata=full
    return dict(M=80,FULL9=True,original_sampler_unchanged=True,production_capability_issued=False)
