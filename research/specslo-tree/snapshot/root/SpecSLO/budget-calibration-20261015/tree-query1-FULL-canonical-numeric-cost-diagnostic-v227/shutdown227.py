"""Retain original API exit, then bound waiting for its own child teardown."""
import time

def finish(engine,*,grace_seconds=90):
    started=time.monotonic()
    if engine is None:return dict(closed=False,all_exited=False),dict(additional_grace_seconds=grace_seconds,waited_for_owned_children=False)
    engine.exit()
    pending=[p for p in engine._processes if p.is_alive()]
    deadline=time.monotonic()+grace_seconds
    for process in pending:process.join(timeout=max(0.,deadline-time.monotonic()))
    return dict(closed=bool(engine._closed),all_exited=all(not p.is_alive()for p in engine._processes)),dict(
        original_API_exit_called=True,additional_grace_seconds=grace_seconds,waited_for_owned_children=bool(pending),
        elapsed_total_seconds=time.monotonic()-started,external_processes_touched=False,measurement_clock_changed=False)
