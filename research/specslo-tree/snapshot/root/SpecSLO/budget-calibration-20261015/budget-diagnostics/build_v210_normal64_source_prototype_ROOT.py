"""Build uninstalled bounded B3/M64 source, without a model entry or issuer."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
DONOR = B / 'tree-normal-cap3-original-geometries-full-v207'
OUT = D / 'v210-source-design-only'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    frozen = json.loads((DONOR/'FROZEN_SOURCE207.json').read_text())
    manifest = []

    def derive(old_name, new_name, replacements):
        old = DONOR/old_name
        assert frozen['files'][str(old)] == sha(old)
        text = old.read_text()
        for before, after, count in replacements:
            assert text.count(before) == count, (old_name, before)
            text = text.replace(before, after)
        restored = text
        for before, after, count in reversed(replacements):
            assert restored.count(after) == count, (old_name, 'inverse', after)
            restored = restored.replace(after, before)
        assert restored == old.read_text()
        compile(text, str(OUT/new_name), 'exec')
        with (OUT/new_name).open('x') as stream:
            stream.write(text)
        manifest.append(dict(source=str(old), source_sha256=sha(old), output=str(OUT/new_name),
                             output_sha256=sha(OUT/new_name), replacements=replacements,
                             exact_whole_text_inverse=True))

    derive('bank207.py', 'bank210.py', [
        ("m,s,pad=expected_geometry(16,64,'causal')", 'm,s,pad=64,16,[]', 1),
        ("(m,s,pad)==(96,32,[2]*16),'original197 N16 B3 is96/S32, never128'",
         "(m,s,pad)==(64,16,[]),'bounded new real-only64/S16, not a production grant'", 1),
        ("route='old',name='original96-B3'", "route='bounded64',name='bounded64-B3'", 1),
    ])
    derive('target207.py', 'target210.py', [('from bank207 import ', 'from bank210 import ', 1)])
    derive('actor207.py', 'actor210.py', [
        ('from capability207 import GREEDY,HIDDEN', 'from geometry210 import GREEDY,HIDDEN', 1),
        ('from bank207 import cases,build_bank', 'from bank210 import cases,build_bank', 1),
        ('from target207 import qualify,reference,commit_reference', 'from target210 import qualify,reference,commit_reference', 1),
        ("folder=Path(output)/'original96-B3-own-current-process'",
         "folder=Path(output)/'bounded64-B3-own-current-process'", 1),
        ("GREEDY in runner.target_entries and runner.target_entries[GREEDY].runtime_validated is True\n   and HIDDEN not in runner.target_entries,'old44 G96 already genuine, diagnostic H96 absent'",
         "GREEDY not in runner.target_entries and HIDDEN not in runner.target_entries,'bounded G64/H64 both absent before actual qualification'", 1),
        ("added=set()if engine.is_draft else {HIDDEN}", "added=set()if engine.is_draft else {GREEDY,HIDDEN}", 1),
        ("'only one diagnostic hidden key, zero production key'", "'two bounded diagnostic G64/H64 keys, zero production issuer'", 1),
        ("'original raw hidden cold/changed/restored gate'", "'original raw hidden cold/changed/restored gate'", 0),
        ("schema='V207.original96-B3-current-component.v1'", "schema='V210.bounded64-B3-current-component.v1'", 1),
        ("from audit3207 import audit", "from audit3210 import audit", 1),
        ("engine._v207_actual_B3=proof;return proof", "engine._v210_actual_B3_M64=proof;return proof", 1),
        ("production_mapper_changed=False,production_new_keys=0,diagnostic_new_hidden_keys=1)",
         "production_mapper_changed=False,production_new_keys=0,diagnostic_new_hidden_keys=1,diagnostic_new_greedy_keys=1)", 1),
    ])
    derive('audit3207.py', 'audit3210.py', [
        ('from capability207 import GREEDY,HIDDEN', 'from geometry210 import GREEDY,HIDDEN', 1),
        ("r['schema']=='V207.original96-B3-current-component.v1'", "r['schema']=='V210.bounded64-B3-current-component.v1'", 1),
        ("len(heads)==96,'full96 physical, first64 genuine queries'", "len(heads)==64,'all64 physical queries are genuine'", 1),
        ("r['diagnostic_new_hidden_keys']==1,'only current diagnostic H96 addition'",
         "r['diagnostic_new_hidden_keys']==1 and r['diagnostic_new_greedy_keys']==1,'bounded G64/H64 additions, no production issuer'", 1),
        ("row['case']['M']==96 and row['case']['S']==32 and row['case']['Q']==64 and row['case']['dummy_lengths']==[2]*16",
         "row['case']['M']==64 and row['case']['S']==16 and row['case']['Q']==64 and row['case']['dummy_lengths']==[]", 1),
        ("a['production_greedy_M']==a['separate_hidden_counterpart_M']==b['same_M']==96 and b['same_S']==32",
         "a['production_greedy_M']==a['separate_hidden_counterpart_M']==b['same_M']==64 and b['same_S']==16", 1),
        ("cold=family=='hidden'and p==0 and c==0", "cold=p==0 and c==0", 1),
        ("pp[0]['runtime_validated_after']is(not cold)", "pp[0]['runtime_validated_after']is(not cold or family=='greedy')", 1),
        ("len(meta['inputs'])==len(meta['positions'])==len(meta['slots'])==96 and len(meta['owners'])==32",
         "len(meta['inputs'])==len(meta['positions'])==len(meta['slots'])==64 and len(meta['owners'])==16", 1),
        ("meta['actual_seq_lengths_q']==list(range(4,65,4))+list(range(66,97,2))",
         "meta['actual_seq_lengths_q']==list(range(4,65,4))", 1),
        ("a['metadata']['owners']==list(range(16))+list(range(463,479))and b['metadata']['owners']==list(range(16,32))+list(range(479,495))",
         "a['metadata']['owners']==list(range(16))and b['metadata']['owners']==list(range(16,32))", 1),
        ("added=[]if rank==0 else[repr(HIDDEN)]", "added=[]if rank==0 else sorted(map(repr,(GREEDY,HIDDEN)))", 1),
        ("cm=row['commit'];require", "require((isinstance(row['draft'],dict)and row['target']is row['reference']is None)if rank==0 else\n     (row['draft']is None and isinstance(row['target'],dict)and isinstance(row['reference'],dict)),\n     'actual role-owned draft/target/reference fields; unused roles must be None')\n    cm=row['commit'];require", 1),
        ("if rank==0:missing|=repair(nxt['draft_proof'],previous)>0", "require(isinstance(nxt['draft_proof'],dict)if rank==0 else nxt['draft_proof']is None,'role-owned actual next draft proof')\n   if rank==0:missing|=repair(nxt['draft_proof'],previous)>0", 1),
        ("schema='V207.current-original96-B3-raw-audit.v1'", "schema='V210.bounded64-B3-raw-audit.v1'", 1),
        ('real16_B3_original_M96_S32=True', 'real16_B3_bounded_M64_S16=True', 1),
        ('all96_hidden_head_all128_queryKV_retained=True', 'all64_hidden_head_all128_queryKV_retained=True', 1),
        ('diagnostic_hidden_new_keys=1,checkpoint_sha256=pins', 'diagnostic_hidden_new_keys=1,diagnostic_greedy_new_keys=1,production_permission=False,checkpoint_sha256=pins', 1),
    ])
    record = dict(
        schema='ROOT.V210.uninstalled-normal64-source-projection.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='source_prototype_only_not_model_ready', donor=str(DONOR),
        donor_freeze_sha256=sha(DONOR/'FROZEN_SOURCE207.json'), derived=manifest,
        immutable_original_sources_modified=False, models_launched=False, NPU_used=False,
        source_functions_imported=False, current_numerical_qualification=False,
        production_issuer=False, full463=False, g_installed=False,
        pending=[
            'Current-engine bounded geometry owner/context; no production permission.',
            'Original current201/44/114+B4/M80 before new actual model paths.',
            'All64 target hidden/head/all128 queryKV; independent owners, original commit/next repair; discovery+two complete zero traces.',
            'Real producer/auditor HOST schema test, closure/freeze/launch guard, and actual bounded model run.',
            'A complete-origin promoted-child proof and E coverage are separate requirements before any production64 activation.',
        ],
    )
    path = OUT/'SOURCE_PROJECTION.json'
    with path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(path=str(path), sha256=sha(path), files=len(manifest))))


if __name__ == '__main__':
    main()
