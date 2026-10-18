"""Explicit next diagnostic: inspect ancestor root and side at equal masks."""
import ast,datetime,hashlib,json,pathlib,tarfile
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-static-q5-M128-long-branch-input-diagnostic-v252'
P=B/'tree-static-q5-M128-root-ancestry-input-diagnostic-v253'
P.mkdir(exist_ok=False)
for f in OLD.iterdir():
    if not f.is_file()or f.name in ['FREEZE252.json','SOURCE252.tar.gz']:continue
    s=f.read_text().replace(str(OLD),str(P)).replace('252','253')
    if f.name=='probes252.py':
        s=s.replace("row=5;group=1","row,group=(0,0)if mode=='candidate-root'else(5,1)")
        s=s.replace("traces.setdefault(mode,{})[layer]=dict(values=values", "traces.setdefault(mode,{})[layer]=dict(values=values")
        # Capture actual root and side on two unchanged eager executions.
        anchor="stage('ROOT253/eager-actual-attention-inputs',inspect_actual)"
        assert s.count(anchor)==1
        s=s.replace(anchor,anchor+"\n    def inspect_root():\n        if not target:return None\n        capture['mode']='candidate-root'\n        try:return engine._branch_tail_state.eager_last().cpu()\n        finally:capture['mode']=None\n    stage('ROOT253/eager-actual-root-attention-inputs',inspect_root)")
        a="if target:capture['mode']='reference-cap'+str(capacity)if label=='independent-reference/frame1'else None"
        z="if target:capture['mode']=('reference-cap'+str(capacity)if label=='independent-reference/frame1'else 'reference-root-cap'+str(capacity)if label=='independent-reference/frame0'else None)"
        assert s.count(a)==1;s=s.replace(a,z)
        s=s.replace("other=base[layer]","other=traces['candidate-root'][layer]if 'root'in mode else base[layer]")
        a="candidate=fingerprint(a),reference=fingerprint(v))"
        z="candidate=fingerprint(a),reference=fingerprint(v),different_positions=(a!=v).reshape(a.shape[0],-1).any(-1).nonzero().flatten().tolist())"
        assert s.count(a)==1;s=s.replace(a,z)
    (P/f.name.replace('252','253')).write_text(s)
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
f=OLD/'FREEZE252.json';files=dict(json.loads(f.read_text())['files']);files[str(f)]=sha(f)
for f in P.iterdir():
    if f.suffix=='.py':ast.parse(f.read_text(),filename=str(f))
    if f.is_file():files[str(f)]=sha(f)
f=B/'budget-diagnostics/build_ROOT253_root_ancestry_probe.py';files[str(f)]=sha(f)
f=P/'FREEZE253.json';f.write_text(json.dumps(dict(schema='ROOT253.bounded-root-ancestry-attention-inputs.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,prior_frozen_sources_edited=False,performance=False,production_qualification=False),indent=2)+'\n')
with tarfile.open(P/'SOURCE253.tar.gz','w:gz')as t:
    for name in sorted([*files,str(f)]):t.add(name,arcname=name.lstrip('/'),recursive=False)
print(json.dumps(dict(freeze=sha(f),archive=sha(P/'SOURCE253.tar.gz'),members=len(files)+1)))
