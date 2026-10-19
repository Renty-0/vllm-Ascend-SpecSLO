"""Fresh independent model proof; actual165 MC2 routes and FIA160."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

P=Path(__file__).resolve().parent
PREVIOUS=P.parent/'tree-M160-C128-capacity-binding-v329'
sys.path.insert(0,str(PREVIOUS))
import guard329 as previous
previous.PIN='8ad8a253fbf54cb026b2a8498efc8d3b1a7b7a28e4d08f671d94f64b4e3fd949'
PIN=None

def prepared():
    old,sources,readiness=previous.prepared()
    from check331 import ready
    frozen=ready(PIN)
    sources.update(frozen['files']);sources[str(P/'FREEZE331.json')]=PIN
    readiness.update(own331_freeze_sha256=PIN,FIA160_groups32=True,
                     actual165_projection_MC2=True,no_profile160_claim=True,
                     fresh532_required=True)
    return old,sources,readiness

def launch_argv(tag):
    argv=previous.launch_argv(tag)
    argv.insert(argv.index('/bin/bash'),'SPECSLO_V331_ROOT_FREEZE_SHA='+PIN)
    argv[argv.index(str(PREVIOUS/'launch329.sh'))]=str(P/'launch331.sh')
    return argv

def storage(tag=None):
    row=previous.storage(tag)
    extra=[P,P.parent/'tree-M160-actual165-MC2-bridge-host-v330']
    row['aggregate_private_bytes']+=sum(f.stat().st_size for root in extra for f in root.rglob('*')if f.is_file())
    previous.previous.original.previous.require(row['aggregate_private_bytes']<=previous.previous.original.MAX_AGGREGATE_BYTES,'bounded private331 component')
    return row

def actual_audit(directory):
    result=previous.actual_audit(directory)
    require=previous.previous.original.previous.require
    for rank in range(1,4):
        report=json.loads((Path(directory)/f'rank{rank}-numeric.json').read_text())
        proof=json.loads((Path(directory)/f'rank{rank}-bridge330-before-tree.json').read_text())
        bridge=report['actual165_MC2_bridge330']
        require(proof['rank']==rank and proof['installed_before_diagnostic_tree']is True
                and proof['physical_FIA_rows160']is True and proof['actual_projection165']is True,
                'instance bridge installed before all fresh tree math')
        expected={(i,k)for i in range(64)for k in(3072,8576)}
        actual={(row['layer'],row['K'])for row in bridge['construction_witnesses']}
        require(actual==expected and all(row['calls']>0 for row in bridge['construction_witnesses'])
                and bridge['FIA_rows']==160 and bridge['actual_projection_MC2_rows']==165
                and bridge['profile160_claimed']is False and bridge['pending_tickets']==0
                and bridge['deferred_chain_enabled']is False,
                'all128 exact original165 projection constructions, no160 profile or deferred chain claim')
    result['bridge331_actual_FIA160_projection165_fresh532_passed']=True
    return result

def main():
    global PIN
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    args=parser.parse_args()
    require=previous.previous.original.previous.require
    require(re.fullmatch('[0-9a-f]{64}',args.expected_freeze_sha256)is not None and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'safe explicit331 pin and tag')
    PIN=args.expected_freeze_sha256;prepared()
    namespace=previous.previous.original.previous.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,actual_audit=actual_audit,storage=storage,
                     ACTION=previous.previous.ACTION,MAX_AGGREGATE_BYTES=previous.previous.original.MAX_AGGREGATE_BYTES)
    with previous.previous.original.previous.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)

if __name__=='__main__':main()
