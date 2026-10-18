"""Diagnostic-only installation; never claims a full production capability."""
def install_entry(entry,native):
    import host_target_metadata as host
    from root_padding_forward231 import host_installer
    original=host.install;host.install=host_installer(original)
    try:policy=entry['install'](fast=True,host=True,transport=True,replay=True,repair=True,prefill=False)
    finally:host.install=original
    return policy

def activate(engine,native):
    from branch_tail231 import install
    return install(engine,native)
