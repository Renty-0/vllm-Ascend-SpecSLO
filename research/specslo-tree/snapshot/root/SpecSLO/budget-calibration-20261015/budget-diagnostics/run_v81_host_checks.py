"""Retained-runtime host checks with source hashes and an execution firewall."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--test', action='append', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use retained Python 3.11; AST receipts are version bound')
    sys.path.insert(0, str(candidate))
    files = {p.resolve() for p in candidate.glob('*.py')}
    files.add(Path(__file__).resolve())
    files.update(p.resolve() for p in candidate.glob('*.json'))
    provenance = json.loads((candidate/'PARENT_PROVENANCE_V75.json').read_text())
    files.update(Path(p).resolve() for p in provenance['native_contract_sources'])
    files.update(Path(p).resolve() for p in provenance['reference_sources'])
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    forbidden = []
    model_calls = []
    def firewall(frame, event, arg):
        if event == 'c_call' and getattr(arg, '__name__', '') == '_npu_init':
            forbidden.append(('C', '_npu_init'))
            raise RuntimeError('Host check attempted NPU initialization')
        if event != 'call':
            return
        filename = frame.f_code.co_filename
        name = frame.f_code.co_name
        module = frame.f_globals.get('__name__', '')
        if ('/pearl/' in filename and name in ('forward', '_run_packed_hidden',
                '_run_packed_hidden_graph', 'draft_forward', 'target_tree_forward')):
            model_calls.append((filename, name))
            raise RuntimeError('Host check attempted model execution')
        if (('/pearl/native_engine.py' in filename and name == '__init__'
                and type(frame.f_locals.get('self')).__name__ == 'NativePearlEngine')
                or (module.startswith('torch_npu.npu') and name in ('_lazy_init', 'init'))):
            forbidden.append((filename, name))
            raise RuntimeError('Host check attempted Native/NPU initialization')
    start = time.perf_counter()
    result = None
    fatal = None
    sys.setprofile(firewall)
    try:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in args.test)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    except BaseException:
        fatal = traceback.format_exc()
        print(fatal, file=sys.stderr)
    finally:
        sys.setprofile(None)
    imported = {str(Path(module.__file__).resolve()) for module in tuple(sys.modules.values())
        if isinstance(getattr(module, '__file__', None), str)
        and Path(module.__file__).suffix == '.py'}
    used = {str(candidate/(name.split('.')[0]+'.py')) for name in args.test}
    used.update(imported.intersection(before))
    # Reference source/provenance bytes can be inspected without importing them.
    used.update(str(p.resolve()) for p in files if not p.resolve().is_relative_to(candidate))
    used.update(str(p.resolve()) for p in candidate.glob('*.json'))
    hashes = {p: before[p] for p in sorted(used) if p in before}
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}
    passed = (result is not None and result.wasSuccessful() and fatal is None
        and hashes == after and not forbidden and not model_calls)
    receipt = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        sys_executable=sys.executable, sys_version=sys.version, tests=args.test,
        tests_run=None if result is None else result.testsRun,
        failures=None if result is None else len(result.failures),
        errors=None if result is None else len(result.errors), fatal=fatal,
        input_hashes=hashes, input_hashes_after=after,
        forbidden_initialization_attempts=forbidden, model_forward_attempts=model_calls,
        NPU_initialized=bool(getattr(sys.modules.get('torch_npu.npu'), '_initialized', False)),
        elapsed_seconds=time.perf_counter()-start,
        status='passed' if passed else 'failed', formal_Goodput=False)
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ('tests_run', 'failures', 'errors',
        'elapsed_seconds', 'status')}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
