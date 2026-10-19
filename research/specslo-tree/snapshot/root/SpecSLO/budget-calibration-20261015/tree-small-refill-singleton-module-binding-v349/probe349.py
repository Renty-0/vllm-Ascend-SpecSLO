"""Actual original exact singleton graph and small multiowner eager controls."""
def run(engine, native, stage, output, fixture):
    import prefill_exact
    stage('ROOT349/install-original-exact-singleton-graph', lambda: prefill_exact.install(native))
    from prefill349 import run as component
    return component(engine, native, stage, output)
