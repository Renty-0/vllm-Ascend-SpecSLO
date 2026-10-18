"""Remember only fresh-child descendants, binding every PID to its start tick."""
import os,signal,time
from pathlib import Path


def snapshot():
    rows={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw=path.read_text();fields=raw[raw.rfind(')')+2:].split()
            pid=int(path.parent.name)
            rows[pid]=dict(pid=pid,ppid=int(fields[1]),state=fields[0],start_ticks=int(fields[19]))
        except(OSError,ValueError,IndexError):continue
    return rows


class Owned:
    def __init__(self,pid,initial):
        if pid not in initial:raise RuntimeError('fresh child identity must exist')
        self.seed=pid;self.identities={pid:initial[pid]['start_ticks']}

    def update(self,rows):
        active={pid for pid,start in self.identities.items()
                if pid in rows and rows[pid]['start_ticks']==start and rows[pid]['state']!='Z'}
        while True:
            extra={pid for pid,row in rows.items()if row['ppid']in active and pid not in active and row['state']!='Z'}
            if not extra:break
            for pid in extra:self.identities[pid]=rows[pid]['start_ticks']
            active.update(extra)
        return[rows[pid]for pid in sorted(active)]

    def remaining(self):return self.update(snapshot())

    def cleanup(self):
        before=self.remaining()
        for signum in(signal.SIGTERM,signal.SIGKILL):
            rows=self.remaining()
            if not rows:break
            for row in reversed(rows):
                # Re-read immediately before signalling so PID reuse never
                # turns an earlier owned descendant into an unrelated target.
                current=snapshot().get(row['pid'])
                if current is None or current['start_ticks']!=self.identities[row['pid']]:continue
                try:os.kill(row['pid'],signum)
                except ProcessLookupError:pass
            until=time.monotonic()+5
            while self.remaining()and time.monotonic()<until:time.sleep(.05)
        return dict(before=before,after=self.remaining(),scope='fresh child descendants only; PID plus immutable start tick; no global pkill')
