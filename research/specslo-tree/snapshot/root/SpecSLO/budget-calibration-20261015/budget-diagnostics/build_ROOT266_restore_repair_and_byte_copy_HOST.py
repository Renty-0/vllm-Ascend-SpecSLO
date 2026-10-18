"""Fresh epoch: restore actual218 repair and use proven raw-byte branch copies."""
import ast,collections,re
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-spare-branch-group-budget-full-v260';P=B/'tree-restored-cost-repair-byte-KV-full-v266'
prefixes={'ROOT','_ROOT','V','SPECSLO_V','_canonical','_root','full','v','FREEZE','SOURCE','FIXTURE','HOST_FULL','HOST_ADMISSION','HOST_PRIORITY','INTEGRATION_REVIEW'}
prefixes.update(f.stem.removesuffix('260')for f in OLD.iterdir()if f.suffix in('.py','.sh'))
pattern=re.compile(r'\b('+ '|'.join(re.escape(s)for s in sorted(prefixes,key=len,reverse=True))+r')260(?=\b|_)')
def renamed(source):return pattern.sub(lambda m:m.group(1)+'266',source.replace(str(OLD),str(P)))
for f in OLD.iterdir():
 if not f.is_file()or f.suffix not in('.py','.sh')or f.name in('budget260.py','static_state260.py','check_budget260_HOST.py'):continue
 target=P/f.name.replace('260','266');assert not target.exists(),str(target)
 s=renamed(f.read_text())
 if f.name in('runtime260.py','guard260.py'):s=s.replace('sys.path.extend([','sys.path.extend([str(B/\'tree-spare-branch-group-budget-full-v260\'),',1)
 if f.name=='startup260.sh':s=s.replace('export PYTHONPATH=','export PYTHONPATH='+str(OLD)+':'+str(B/'tree-fused-byte-branch-KV-copy-v263')+':'+str(B/'tree-fused-branch-KV-row-copy-v262')+':',1)
 if f.name=='scope260.py':
  s=s.replace("    permission = getattr(engine, '_canonical266_permission', None)","    from static_state266 import require_copy_bindings\n    require_copy_bindings(engine)\n    permission = getattr(engine, '_canonical266_permission', None)",1)
  s=s.replace('inherited_original_cap4_wrapper_not_called_by_this_LIVE_shape=True,','actual_original218_repair_solver_and_generation_state_preserved=True,\n                checked_raw_byte_KV_copy_installed=True,',1)
 if f.name=='check_full260_HOST.py':
  s=s.replace('from static_state266 import install','from static_state255 import install',1)
  s=s.replace("sys.path.append('/root/SpecSLO/budget-calibration-20261015/tree-static-q5-M120-SLO-first-full463-own-AR-v257')","sys.path.append("+repr(str(OLD))+")\nsys.path.append('/root/SpecSLO/budget-calibration-20261015/tree-static-q5-M120-SLO-first-full463-own-AR-v257')",1)
 target.write_text(s)
(P/'FIXTURE266.json').write_bytes((OLD/'FIXTURE260.json').read_bytes())
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
print(P)
