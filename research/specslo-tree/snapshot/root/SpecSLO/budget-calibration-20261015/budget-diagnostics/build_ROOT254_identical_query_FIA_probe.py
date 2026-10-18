"""New counterfactual operator-only diagnostic, with actual model Q/K/V."""
import ast,datetime,hashlib,json,pathlib,tarfile
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-static-q5-M128-root-ancestry-input-diagnostic-v253'
P=B/'tree-identical-query-FIA-geometry-diagnostic-v254'
P.mkdir(exist_ok=False)
for name in ['runtime253.py','check253.py','launch253.sh','startup253.sh','guard253.py','FIXTURE253.json']:
    s=(OLD/name).read_text().replace(str(OLD),str(P)).replace('253','254')
    if name=='guard253.py':
        a="require(all(len(v['cases'])==3 and v['input_trace']for v in reports[1:]),'three actual math cases and captured model attention inputs')"
        z="require(all(v['operator_tests']and v['actual_model_case']['N']==1 for v in reports[1:]),'actual model inputs followed by operator geometry diagnostics')"
        assert s.count(a)==1;s=s.replace(a,z)
    (P/name.replace('253','254')).write_text(s)
print(P)
