"""Explicit owned-process Python stack signal; no polling or device operations."""
import faulthandler,json,os,signal
from pathlib import Path

def install(engine,output):
    if hasattr(engine,'_ROOT346_fault_stream'):raise RuntimeError('ROOT346 signal logger installed once')
    path=Path(output)/f'rank{engine.rank}-python-stacks.log'
    stream=path.open('x')
    faulthandler.register(signal.SIGUSR1,file=stream,all_threads=True,chain=False)
    engine._ROOT346_fault_stream=stream
    engine._ROOT346_debug_output=str(output)
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    report=dict(rank=engine.rank,pid=os.getpid(),proc_start_ticks=int(stat[19]),
        signal_number=int(signal.SIGUSR1),signal_name='SIGUSR1',stack_file=str(path),
        registered_after_constructor=True,locals_dumped=False,automatic_sampling=False,
        algorithm_modified=False,device_operation=False)
    with(Path(output)/f'rank{engine.rank}-fault-signal-ready.json').open('x')as f:
        json.dump(report,f,indent=2);f.write('\n')
    return report
