"""Original target257 and all original gates, plus current-engine draft math."""
import extension257 as original

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
cache_snapshot = original.cache_snapshot
postfull_AR = original.postfull_AR
snapshot = original.snapshot


def loaded_origins():
    from check372 import ready
    value = ready()
    from pathlib import Path
    return dict(original.loaded_origins(), ROOT372_fresh_source_files={p: h for p, h in value['files'].items() if Path(p).parent == Path(__file__).resolve().parent})


def qualify_component(engine, native, stage, policy, output):
    component = original.qualify_component(engine, native, stage, policy, output)
    from pathlib import Path
    from qualify372 import qualify
    qualification = stage('ROOT372/fresh-same-instance-draft306-allN-eager',
        lambda: qualify(engine, native, Path(output) / 'draft-tail3-same-instance') if engine.is_draft else None)
    if engine.is_draft:
        from live372 import install
        install(engine, native, qualification)
    return component


def tail_snapshot(engine):
    from live372 import snapshot as tail
    return tail(engine)
