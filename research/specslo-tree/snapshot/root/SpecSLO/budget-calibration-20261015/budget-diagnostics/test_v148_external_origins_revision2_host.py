"""Only the actual frozen producer schema correction; old6 regressions stay SHA evidence."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import analyze_v148_actual_full_cap4_revision2 as audit

D=Path(__file__).resolve().parent
OLD_SHA='b8e4bddae4c87a05d28da524b606f1432249f7d38cb53f4c4a0ad2f93109ad1a'
WRONG="""        for name,row in report['actual_source_origins'].items():
            need(sources.get(row['path'])==row['sha256'],'actual own component imported source '+name)
"""
RIGHT="""        component_origins(report['actual_source_origins'],mods['audit_candidate_v132'].KEYS,sources)
"""


def actual_producer_under_HOST_import_stubs(root):
    """Compile unchanged original producer functions; imports are stub module files only.

    Running the real full module import closure would import Torch. This test
    explicitly makes no such claim; completed V143 runtime reports are separate
    real-origin evidence.
    """
    path=root/'import_origins_v132.py';tree=ast.parse(path.read_text())
    assignments=[n for n in tree.body if isinstance(n,ast.Assign)]
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    assert [n.name for n in funcs]==['local_first','require_loaded_origins']
    sys_stub=SimpleNamespace(path=[],modules={})
    def importer(name):
        module=SimpleNamespace(__file__=str(root/(name+'.py')))
        sys_stub.modules[name]=module
        return module
    namespace=dict(Path=Path,sys=sys_stub,importlib=SimpleNamespace(import_module=importer),__file__=str(path))
    exec(compile(ast.Module(body=assignments+funcs,type_ignores=[]),str(path)+':original-producer-HOST-import-stubs','exec'),namespace)
    return namespace['require_loaded_origins'](),namespace['KEYS']


class ActualProducerOriginSchemaHOST(unittest.TestCase):
    def test_original_producer_actual_retained_format_and_strict_negative_cases(self):
        old=(D/'analyze_v148_actual_full_cap4.py').read_text()
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),OLD_SHA)
        revised=Path(audit.__file__).read_text();tree=ast.parse(revised)
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='component_origins')
        lines=revised.splitlines(keepends=True)
        del lines[helper.lineno-1:helper.end_lineno+2]
        inverse=''.join(lines).replace(RIGHT,WRONG)
        self.assertEqual(inverse,old)
        proof=json.loads((D/'V148_EXTERNAL_AUDITOR_HOST_20261017T030429964840Z.json').read_text())
        self.assertEqual((proof['status'],proof['tests_run']),('passed',6))
        self.assertEqual(proof['input_hashes'][str(D/'analyze_v148_actual_full_cap4.py')],OLD_SHA)
        frozen=json.loads((audit.P/'PREPARED_SOURCE_HASHES_V132.json').read_text())
        sources={**frozen['files'],**frozen['external_files']}
        rows,keys=actual_producer_under_HOST_import_stubs(audit.P)
        self.assertTrue(all(type(path) is str for path in rows.values()))
        self.assertEqual(set(audit.component_origins(rows,keys,sources)),set(keys))
        wrong=deepcopy(rows);name=next(iter(wrong));wrong[name]=dict(path=rows[name],sha256=sources[rows[name]])
        with self.assertRaises(ValueError):audit.component_origins(wrong,keys,sources)
        wrong=deepcopy(rows);wrong.pop(name)
        with self.assertRaises(ValueError):audit.component_origins(wrong,keys,sources)
        wrong=deepcopy(rows);wrong[name]=rows[name]+'/wrong'
        with self.assertRaises(ValueError):audit.component_origins(wrong,keys,sources)
        bad=dict(sources);bad[rows[name]]='0'*64
        with self.assertRaises(ValueError):audit.component_origins(rows,keys,bad)
        root=audit.B/'tree-standalone-prefill-control-v143'
        producer,oldkeys=actual_producer_under_HOST_import_stubs(root)
        source=json.loads((root/'PREPARED_SOURCE_HASHES_V132.json').read_text())['files']
        directory=root/'local-output/standalone-prefill-v143-one-full463-v1'
        for rank in range(4):
            actual=json.loads((directory/f'rank{rank}-local-diagnostic.json').read_text())
            for side in ('before','after'):
                observed=actual['actual_helper_origins_'+side+'_constructor']
                self.assertEqual(observed,producer)
                audit.component_origins(observed,oldkeys,source,root=root)
                with self.assertRaises(ValueError):audit.component_origins(observed,oldkeys,source)


if __name__=='__main__':unittest.main()
