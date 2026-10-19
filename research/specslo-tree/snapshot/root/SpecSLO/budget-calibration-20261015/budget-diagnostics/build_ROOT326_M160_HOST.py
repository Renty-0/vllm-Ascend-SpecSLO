"""Explicit M160/S32 source prototype after synthetic S33 FIA divergence."""
import ast
import io
import json
import re
import tokenize
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-M160-S32-N32-target-math-cost-v326'
BASE=B/'tree-uniform-q5-M165-S33-C128-host-v285'
PREP=B/'tree-M165-single-prepare-host-v313'
OLD=B/'tree-M165-N32-independent-target-math-cost-v317'

def project(text,versions,*,physical_groups=True):
    converted=[]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        value=token.string
        if token.type==tokenize.NUMBER:
            value={'165':'160','166':'161',**({'33':'32'}if physical_groups else {})}.get(value,value)
        else:
            for before,after in versions:
                value=re.sub(r'(?<![0-9])'+str(before)+r'(?![0-9])',str(after),value)
            value=value.replace('M165','M160').replace('S33','S32')
        converted.append(token._replace(string=value))
    result=tokenize.untokenize(converted)
    return re.sub(r'range\((1|25),\s*32\)',lambda m:'range('+m.group(1)+',33)',result)

def main():
    P.mkdir(exist_ok=False)
    for source in BASE.glob('*.py'):
        if source.name in ('check285_HOST.py','fixture285_HOST.py'):
            target={'check285_HOST.py':'check_base326_HOST.py','fixture285_HOST.py':'fixture_base326_HOST.py'}[source.name]
        elif source.name.startswith('check_'):
            continue
        else:
            target=source.name.replace('285','326')
        text=project(source.read_text(),[(285,326)])
        (P/target).write_text(text)
    for name in ('descriptors','write','static_state'):
        (P/f'{name}326.py').write_text(project((PREP/f'{name}313.py').read_text(),[(313,326),(285,326)]))
    text=project((PREP/'fixture313_HOST.py').read_text(),[(313,326),(285,326)])
    text=text.replace('import fixture326_HOST as original','import fixture_base326_HOST as original')
    (P/'fixture326_HOST.py').write_text(text)
    text=project((PREP/'check313_HOST.py').read_text(),[(313,326),(285,326)])
    text=text.replace('import check326_HOST as original','import check_base326_HOST as original')
    (P/'check_split326_HOST.py').write_text(text)
    for name in ('oracle','baseline','benchmark','bank','reference_schedule','audit_static_calls'):
        text=project((OLD/f'{name}317.py').read_text(),[(317,326),(285,326)])
        text=text.replace("'baseline234':'baseline326'","'baseline234':'baseline326'")
        (P/f'{name}326.py').write_text(text)
    # Original408 fixtures stay byte-identical in budgets/anchors. Only the
    # newly proposed N25..32 cases are reduced to32 primary/side groups.
    text=(P/'oracle326.py').read_text()
    assert 'from metadata326 import checked'in text
    # Keep old bank offsets32/64; only the old physical frame33 was reduced.
    assert 'owner=64+i'in(P/'reference_schedule326.py').read_text()
    text=project((OLD/'probes317.py').read_text(),[(317,326),(313,326),(285,326)])
    text=text.replace('from runtime326 import save','from runtime317 import save')
    (P/'probes326.py').write_text(text)
    (P/'FIXTURE326.json').write_bytes((OLD/'FIXTURE317.json').read_bytes())
    text=project((B/'tree-M165-raw-reference-api-v318/raw_controls318.py').read_text(),[(313,326)],physical_groups=False)
    assert text.count('torch.arange(512,677,')==1
    text=text.replace('torch.arange(512,677,','torch.arange(512,672,').replace('actual_165_program_graph_replay','actual_160_program_graph_replay')
    (P/'raw_controls326.py').write_text(text)
    text=project((OLD/'check_geometry317_HOST.py').read_text(),[(317,326),(285,326)])
    text=text.replace('import check313_HOST','import check_split326_HOST').replace('import check326_HOST as original','import check_base326_HOST as original').replace("'fixture313_HOST'","'fixture326_HOST'").replace('check313_HOST.verify_descriptors','check_split326_HOST.verify_descriptors')
    (P/'check_geometry326_HOST.py').write_text(text)
    # Direct synthetic kernel test before any new model constructor.
    text=(B/'tree-M165-raw-reference-api-v318/probe_raw318.py').read_text()
    text=text.replace('from raw_controls318 import run','from raw_controls326 import run').replace("q_size=22*128","q_size=24*128").replace('RAW318_MODEL_FREE_RESULT.json','RAW326_MODEL_FREE_RESULT.json')
    (P/'probe_raw326.py').write_text(text)
    # Dependencies remain on original private paths; module326 names are all
    # explicit and never alter CPython311 ABI paths or frozen old sources.
    for source in P.glob('*.py'):
        ast.parse(source.read_text())
    (P/'SOURCE326_REVIEW.json').write_text(json.dumps(dict(status='HOST_source_only',M=160,S=32,Q=5,real_N_max=32,C=128,future=64,global_selected=192,
        tree_primary_plus_side_groups_max=32,KV_extent=2560,bank_owners=96,bank_offsets=[32,64],original408_fixture_budgets_and_anchors_retained=True,
        reason='Synthetic324 identical roots differed only for33groups at long prefixes;24/32groups all tested roots exact, both mask capacities and precision modes',
        actual_MC2_profile_has_M160=False,new_MC2_shape_measurements_required_before_any_M160_fused_claim=True,neural_qualified=False,Goodput=None),indent=2)+'\n')
    print('M160/S32 source prototype created; no model or NPU used')

if __name__=='__main__':main()
