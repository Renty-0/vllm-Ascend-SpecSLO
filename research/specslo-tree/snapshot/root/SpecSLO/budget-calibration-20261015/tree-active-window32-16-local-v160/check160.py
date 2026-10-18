"""Own source/HOST/ROOT gate. No model construction or device access."""
import json
from pathlib import Path
from window160 import P, checked, require, sha


def local_files():
    return {str(p): sha(p) for p in sorted(P.iterdir())
            if p.is_file() and (p.suffix == '.py' or p.name in ('PARENT_INPUTS160.json', 'SCOPE160.txt'))}


def external_files():
    values = json.loads((P / 'PARENT_INPUTS160.json').read_text())['files']
    for path in values:
        checked(path)
    return values


def require_ready():
    cpu_path, freeze_path = P / 'CPU_RESULT160.json', P / 'FROZEN_SOURCE160.json'
    require(cpu_path.is_file() and freeze_path.is_file(), 'own HOST proof and explicit ROOT freeze required')
    cpu, frozen = json.loads(cpu_path.read_text()), json.loads(freeze_path.read_text())
    require(cpu['status'] == 'passed' and cpu['tests_run'] == 4
            and cpu['failures'] == cpu['errors'] == 0
            and cpu['Torch_imported'] is cpu['NPU_used'] is False
            and cpu['local_before'] == cpu['local_after'] == frozen['files'] == local_files()
            and cpu['external_before'] == cpu['external_after'] == frozen['external_files'] == external_files()
            and frozen['CPU_sha256'] == sha(cpu_path) and frozen['ROOT_source_reviewed'] is True,
            'exact own source/CPU/ROOT identity')
    return dict(CPU_sha256=sha(cpu_path), freeze_sha256=sha(freeze_path),
                actual_device_qualification=False, formal_Goodput=False)
