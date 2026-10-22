"""Read bounded progress; request verified registered stacks after full stalls."""
import datetime,json,subprocess,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';T='dynamic-root-alternate-root-v287-eager3-compatible-full463-ownAR-v1';A=O/'actual'/T
STATE=D/'ROOT287_AUTOMATIC_STALL_WATCH.json'

def stalled(generation,snapshot,now):
    if generation.get('generation')!=5 or generation.get('status')!='running'or generation.get('sealed')is not True:return False
    start=generation['started_monotonic'];last=snapshot.get('monotonic',start)if snapshot.get('generation')==5 else start
    return now-start>=180 and now-last>=60

def write(value):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATE)

def main():
    started=time.monotonic();last_write=0
    while time.monotonic()-started<6900:
        receipt=O/f'receipt-{T}.json'
        if not receipt.exists():time.sleep(5);continue
        r=json.loads(receipt.read_text())
        if r['status']!='running':write(dict(status='stopped_on_guard_terminal',guard_status=r['status'],signals_sent=False));return
        paths=list((A/'generation-progress').glob('rank0-*.json'));g=json.loads(paths[0].read_text())if paths else{}
        path=A/'rank0-online-state-latest.json';snapshot=json.loads(path.read_text())if path.exists()else{}
        now=time.monotonic()
        failures=list(A.glob('rank*-generation*-decode-failure-HOST-state.json.gz'))
        if failures or stalled(g,snapshot,now):
            result=subprocess.run([sys.executable,'-B',str(D/'dump_ROOT287_owned_registered_stacks_HOST.py')],capture_output=True,text=True,timeout=15)
            write(dict(status='registered_stack_request_sent'if result.returncode==0 else'registered_stack_request_refused',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),generation=g,snapshot_generation=snapshot.get('generation'),last_snapshot_monotonic=snapshot.get('monotonic'),snapshot_rows=snapshot.get('observations_since_process_start'),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,signals_sent=result.returncode==0,performance_result_invalidated=result.returncode==0))
            print(json.dumps(dict(status='watch_complete',signal_collector_exit=result.returncode,stdout=result.stdout)),flush=True);return
        if g.get('generation')==5 and g.get('status')=='complete':
            write(dict(status='full_finished_without_stall',signals_sent=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('full finished without stall signal',flush=True);return
        if now-last_write>=15:
            write(dict(status='watching_read_only',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),generation=g.get('generation'),generation_status=g.get('status'),sealed=g.get('sealed'),snapshot_rows=snapshot.get('observations_since_process_start'),signals_sent=False));last_write=now
        time.sleep(5)
    write(dict(status='watch_timeout_no_signal',signals_sent=False))

if __name__=='__main__':main()
