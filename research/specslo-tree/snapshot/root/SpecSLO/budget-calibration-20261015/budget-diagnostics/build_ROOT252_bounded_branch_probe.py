"""Fresh private diagnostic epoch. Frozen prior epochs remain unchanged."""
import datetime,hashlib,json,pathlib,tarfile
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-static-q5-M128-long-branch-input-diagnostic-v252'
OLD=B/'tree-static-q5-M128-real-prefix-validated-component-v250'
FULL=B/'tree-static-q5-M128-full463-own-AR-v251'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
P.mkdir(exist_ok=False)
for name in ['runtime','check','launch','startup','guard']:
    ext='.sh'if name in ['launch','startup']else'.py'
    s=(OLD/(name+'250'+ext)).read_text().replace(str(OLD),str(P)).replace('250','252')
    if name=='runtime':
        anchor="sys.path.append(str(B/'tree-contiguous-branch-tail-final-metadata-math-v234'))"
        assert s.count(anchor)==1
        s=s.replace(anchor,anchor+"\nfor folder in ['tree-static-q5-M128-full463-own-AR-v251','tree-static-q5-M128-projection-host-v249','tree-static-q5-immutable-index-cache-host-v247','tree-static-q5-long-prefix-origins-node-math-v246','tree-static-q5-projection-host-v243','tree-canonical-contiguous-full463-own-AR-v235']:\n    sys.path.append(str(B/folder))")
    if name=='guard':
        s=s.replace('MAX_AGGREGATE_BYTES=512*1024**2','MAX_AGGREGATE_BYTES=768*1024**2')
        a=s.index('def actual_audit(directory):');z=s.index('\n\ndef execution_namespace()',a)
        s=s[:a]+'''def actual_audit(directory):
    directory=Path(directory)
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    require(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all diagnostic roles exited')
    require(read(directory/'completed.json')==dict(replies=[['calibrated',r,dict(numeric_cases=120,performance=False)]for r in range(4)],performance=False,adopted=False),'complete original diagnostic protocol;120 is preserved protocol label only')
    require([v['rank']for v in reports]==list(range(4)),'all four diagnostic reports')
    require(all(v['performance']is False and v['whole_workload_AR_qualified']is False and v['adopted']is False for v in reports),'no production or performance qualification')
    require(all(len(v['cases'])==3 and v['input_trace']for v in reports[1:]),'three actual math cases and captured model attention inputs')
    return dict(status='bounded_long_branch_diagnostic_completed',reports=reports,formal_Goodput=False,adopted=False),'diagnostic_only',dict(original_startup_inventory_not_run=True,protocol_120_label_is_not_actual_node_count=True)
'''+s[z:]
    (P/(name+'252'+ext)).write_text(s)
(P/'FIXTURE252.json').write_bytes((FULL/'FIXTURE251.json').read_bytes())
print(P)
