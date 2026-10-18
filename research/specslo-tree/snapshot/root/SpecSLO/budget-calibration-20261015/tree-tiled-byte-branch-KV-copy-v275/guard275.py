"""ROOT lock, frozen bytes, original native verifier, owned synthetic-copy-only probe."""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,re,signal,subprocess,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
HELPER=Path('/root/SpecSLO/resume-20261012/run_baseline_calibration-v1.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value,message):
    if value is not True:raise RuntimeError(message)


def ready(expected,expected_archive):
    freeze=P/'FREEZE275.json';need(sha(freeze)==expected,'explicit synthetic-copy-only source freeze')
    need(sha(P/'SOURCE275.tar.gz')==expected_archive,'explicit complete synthetic-copy-only source archive')
    files=json.loads(freeze.read_text())['files']
    for path,wanted in files.items():need(sha(path)==wanted,'unchanged exact metadata-probe source '+path)
    return files


def verify(root,phase):
    argv=['bash',str(B/'serial-v1/runtime.sh'),'python','-B',
        '/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py',
        '/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json']
    result=subprocess.run(argv,capture_output=True,text=True,timeout=180)
    output=root/(phase+'-verifier.log')
    with output.open('x')as f:f.write(result.stdout+result.stderr)
    rows=[]
    for line in result.stdout.splitlines():
        try:x=json.loads(line)
        except json.JSONDecodeError:continue
        if type(x)is dict and 'checked_frozen_files'in x:rows.append(x)
    need(result.returncode==0 and len(rows)==1 and rows[0]['status']=='pass'
         and rows[0]['checked_frozen_files']==10496 and rows[0]['checked_external_files']==15
         and rows[0]['checked_packages']==212 and rows[0]['errors']==[],
         'original full10496/15/212 native environment verifier')
    return dict(argv=argv,exit_code=result.returncode,path=str(output),sha256=sha(output),actual_result=rows[0])


def execute(tag,expected,expected_archive):
    need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',tag)is not None,'fresh safe tag')
    sources=ready(expected,expected_archive);root=P/'actual'/tag;need(not root.exists(),'preserve earlier actual attempts')
    lock=(B/'.ROOT_NPU_EXECUTION.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir(parents=True,mode=0o700)
    row=dict(schema='ROOT275.synthetic-copy-only-owned-execution.v1',status='running',tag=tag,
        started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=expected,archive_sha256=expected_archive,
        physical_cards=[2,5,6,7],models_loaded=False,neural_qualified=False,Goodput=None,errors=[])
    def save():
        temp=root/'receipt.tmp';temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(root/'receipt.json')
    spec=importlib.util.spec_from_file_location('_ROOT275_owned_helpers',HELPER)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.PHYSICAL=(2,5,6,7)
    child=None;save()
    def interrupted(sig,frame):raise RuntimeError('synthetic-copy-only guard interrupted '+str(sig))
    handlers={sig:signal.signal(sig,interrupted)for sig in(signal.SIGINT,signal.SIGTERM)}
    try:
        row['devices_before']=h.device_state();need(row['devices_before']['all_idle']is True,'actual four cards idle')
        row['pre_verifier']=verify(root,'pre');ready(expected,expected_archive)
        argv=['bash',str(B/'serial-v1/runtime.sh'),'python','-B',str(P/'probe275.py'),str(root/'paired-copy.json')]
        row['argv']=argv
        with(root/'child.log').open('x')as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row['owned_child_pid']=child.pid;save();row['exit_code']=child.wait(timeout=600)
        need(row['exit_code']==0,'synthetic-copy-only child success')
        measured=json.loads((root/'paired-copy.json').read_text())
        need(measured['status']=='complete'and measured['models_loaded']is False
             and measured['model_attention_or_KV_executed']is True and measured['synthetic_KV_copy_executed']is True and measured['attention_executed']is False and measured['real_model_weights_loaded']is False
             and measured['candidate_source_sha256']==sources[str(P/'copy275.py')],
             'actual same frozen synthetic-copy-only child result')
        row['paired_cost_path']=str(root/'paired-copy.json');row['paired_cost_sha256']=sha(root/'paired-copy.json')
        row['actual_probe_complete']=True
    except BaseException:row['errors'].append(traceback.format_exc())
    finally:
        if child is not None:
            for sig in(signal.SIGTERM,signal.SIGKILL):
                members=h.owned_group(child.pid)
                for member in members:
                    try:os.kill(member['pid'],sig)
                    except ProcessLookupError:pass
                if members:time.sleep(2)
            row['owned_remaining']=h.owned_group(child.pid)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:row['errors'].append('owned child not reaped')
        else:row['owned_remaining']=[]
        try:
            ready(expected,expected_archive);row['sources_unchanged']=True
            row['post_verifier']=verify(root,'post');row['devices_after']=h.device_state()
            need(not row['owned_remaining']and row['devices_after']['all_idle']is True,'owned exit and four actual idle')
        except BaseException:row['errors'].append(traceback.format_exc())
        row['status']='complete'if not row['errors']and row.get('actual_probe_complete')else'failed'
        row['finished_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps(dict(status=row['status'],receipt=str(root/'receipt.json'),errors=row['errors'])))
    return 0 if row['status']=='complete'else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--expected-freeze-sha256',required=True);p.add_argument('--expected-archive-sha256',required=True)
    a=p.parse_args();raise SystemExit(execute(a.tag,a.expected_freeze_sha256,a.expected_archive_sha256))
