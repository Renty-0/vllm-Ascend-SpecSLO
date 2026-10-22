"""Clone the exact failed full candidate; add a process-local stack signal only."""
from pathlib import Path
import json,datetime
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-dynamic-root-alternate-paired-prefix-grid-full-v282';P=B/'tree-dynamic-root-alternate-paired-prefix-grid-full-v284'
assert not P.exists();P.mkdir()
for f in OLD.iterdir():
 if f.is_file()and((f.suffix in('.py','.sh')and not f.name.startswith('HOST_'))or f.name=='FIXTURE282.json'):
  target=P/f.name.replace('282','284');target.write_text(f.read_text().replace('282','284'))
  if f.suffix=='.sh':target.chmod(0o700)
p=P/'runtime284.py';s=p.read_text();before="        ('   from audit209 import audit', '   from audit284 import audit'),"
after="""        ('   from audit209 import audit', '   from audit284 import audit'),
        ('   engine._v101_output_directory=output_directory',
         '   engine._v101_output_directory=output_directory\\n   from fault_handler284 import install as register_fault_dump\\n   register_fault_dump(engine,output_directory)'),"""
assert s.count(before)==1;p.write_text(s.replace(before,after))
p=P/'check_full284_HOST.py';s=p.read_text();assert s.count('len(changes)==13')==1;p.write_text(s.replace('len(changes)==13','len(changes)==14'))
review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='source_only_unfrozen_unrun',parent=str(OLD),algorithm_and_model_changes=False,change='register SIGUSR1 after Native constructor; writes owned Python thread stacks only when explicitly requested; no locals and no NPU action',whole463_failure_reproducer=True,all_original_and_new_numerical_gates_retained=True,performance_if_any_signal_sent=False,goal_achieved=False)
(P/'INSTRUMENTATION284_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print(P)
