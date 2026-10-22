"""Read-only CPython3.11 frames via /proc; no signals, ptrace or target mutation."""
import json, pathlib, struct, datetime, argparse

def sample(pid):
    maps=pathlib.Path(f'/proc/{pid}/maps').read_text().splitlines()
    row=next(x for x in maps if '/bin/python3.11' in x and x.split()[2]=='00000000')
    base=int(row.split('-')[0],16)
    with open(f'/proc/{pid}/mem','rb',buffering=0) as f:
        def read(p,n):f.seek(p);b=f.read(n);assert len(b)==n;return b
        def ptr(p):return struct.unpack('<Q',read(p,8))[0]
        def unicode(p):
            n=ptr(p+16);state=struct.unpack('<I',read(p+32,4))[0]
            if not (state&(1<<5)) or n>8192:return '<noncompact>'
            kind=(state>>2)&7;offset=48 if state&(1<<6) else 72
            return read(p+offset,n*kind).decode({1:'latin1',2:'utf-16-le',4:'utf-32-le'}[kind])
        interp=ptr(base+0x57a948+40);threads=[]
        while interp:
            thread=ptr(interp+16)
            while thread:
                cframe=ptr(thread+56);frame=ptr(cframe+8) if cframe else 0;frames=[];seen=set()
                while frame and frame not in seen and len(frames)<128:
                    seen.add(frame);code=ptr(frame+32)
                    frames.append(dict(file=unicode(ptr(code+112)),function=unicode(ptr(code+120)),firstline=struct.unpack('<i',read(code+72,4))[0],instruction_offset=ptr(frame+56)-(code+184)))
                    frame=ptr(frame+48)
                if frames:threads.append(dict(native_thread=ptr(thread+160),frames=frames))
                thread=ptr(thread+8)
            interp=ptr(interp)
        return dict(pid=pid,threads=threads)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('pids',type=int,nargs='+');ap.add_argument('--output');a=ap.parse_args()
    result=dict(schema='READONLY.CPython311-stack-snapshot.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,target_signals_sent=False,target_memory_written=False,target_processes=[])
    for pid in a.pids:
        try:result['target_processes'].append(sample(pid))
        except Exception as exc:result['target_processes'].append(dict(pid=pid,error=repr(exc)))
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:pathlib.Path(a.output).write_text(text+'\n')
    print(text)
