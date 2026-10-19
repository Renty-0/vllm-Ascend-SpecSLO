"""Actual original exact singleton graph and small multiowner eager controls."""
def run(engine, native, stage, output, fixture):
    import prefill_exact
    stage('ROOT347/install-original-exact-singleton-graph', lambda: prefill_exact.install(native))
    from prefill347 import run as component
    return component(engine, native, stage, output)
