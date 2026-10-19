"""Additive raw-reference API repair; retain original frozen317 qualification."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
ORIGINAL = P.parent/'tree-M165-N32-independent-target-math-cost-v317'
sys.path.insert(0,str(ORIGINAL))
import guard317_abi311_paths as previous

previous.EXPECTED_FREEZE='d9dda427f94bd3ecbb507fe0787f867f6e23437ae82fee0455476434eda1f775'
previous.EXPECTED_ARCHIVE='06c599ea5d44f2e6797fe58ed07b201782e0744430c18f69f819a37c0af25069'
PIN = None
MAX_AGGREGATE_BYTES = 2304*1024**2

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepared():
    old,sources,readiness = previous.prepared()
    manifest = P/'FREEZE318.json'
    previous.require(sha(manifest)==PIN,'explicit318 raw-reference repair source pin')
    fixed=json.loads(manifest.read_text())
    for path,wanted in fixed['files'].items():
        previous.require(sha(path)==wanted,'unchanged318 repair '+path)
    previous.require(fixed['original_freeze_sha256']==previous.EXPECTED_FREEZE,'unchanged complete original317 source closure')
    sources.update(fixed['files'])
    sources[str(manifest)]=PIN
    readiness.update(raw_reference_api_repaired=True,candidate_operator_unchanged=True,own318_freeze_sha256=PIN)
    return old,sources,readiness

def launch_argv(tag):
    argv=previous.launch_argv(tag)
    argv[argv.index(str(ORIGINAL/'launch317_abi311_paths.sh'))]=str(P/'launch318.sh')
    return argv

def storage(tag=None):
    helper=previous.donor_guard()
    actual=0 if tag is None else helper.bytes_under(previous.DONOR/'actual'/tag)
    total=helper.bytes_under(previous.DONOR)+helper.bytes_under(ORIGINAL)
    own=sum(p.stat().st_size for p in P.rglob('*')if p.is_file())
    previous.require(actual<=previous.MAX_TAG_BYTES and total+own<=MAX_AGGREGATE_BYTES,'256MiB current tag and2304MiB aggregate private bound')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total+own,additive318_bytes=own)

def actual_audit(directory):
    result=previous.actual_audit(directory)
    for rank in (1,2,3):
        data=json.loads((Path(directory)/f'rank{rank}-numeric.json').read_text())
        checks=data['fresh165_raw_operator']['checks']
        previous.require({c['slots_dtype']for c in checks}=={'torch.int32','torch.int64'}and all(c['reference_slots_dtype']=='torch.int32'and c['reference_slots_values_exactly_representable_int32']for c in checks),'both candidate slot dtypes retained; supported native reference inputs')
    return result

def main():
    global PIN
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    import re
    previous.require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'explicit safe318 pin and fresh tag')
    PIN=args.expected_freeze_sha256
    prepared()
    namespace=previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=actual_audit,storage=storage,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES)
    with previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
