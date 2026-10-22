"""Future exact-authorized ROOT210 audit. No actual is read on source import."""
import argparse
import ast
import collections
import copy
import datetime
import gzip
import hashlib
import importlib.abc
import json
import sys
import tarfile
import types
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=D/'v210-source-design-only';ACTUAL_ROOT=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'
HELPERS=D/'audit_v173_terminal_full463_HOST.py';PRIOR=D/'audit_v197_terminal_HOST.py'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
PINS={str(HELPERS):'ed8aef51646ccf60c4812782d080d119fa406d7e53138ce24d6b470bce74c6f5',
      str(PRIOR):'5118f3e4f2da7ad3de169dfce9e48f310df8097c41c7e0b350df8852eedc81b7',
      str(P/'FROZEN_ROOT210.json'):'e6f732126f31ef4cdb160e120870e80271d2d598eae0fe582e5d9396dcdfa49f',
      str(P/'PREPARED_SOURCE_ROOT210.tar.gz'):'63465a780b0dd364b11bd6ea37b6839dda8e2270bb3452c64e58b3919e152955',
      str(D/'ROOT_V210_PREPARED_BOUNDED64_SOURCE.json'):'797a8cfd3bb2b7c319ed1814dbc2b6ca69c5a01f551c4376feda2548a86950d1'}


def definitions(path,names,namespace):
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(path).read_text()).body
           if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
    if {n.name for n in nodes}!=set(names):raise RuntimeError('complete isolated helper definitions')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 str(path)+'::ROOT210-stdlib-only','exec'),namespace)


def metadata(row):
    source=row['source']['contexts'];rows=row['rows']
    wanted=[];positions=[];seq=[]
    for context,tokens in zip(source,rows,strict=True):
        need(all(type(v)is int and 0<=v<151936 for v in context['token_ids']),
             'actual integer source history, not rounded or placeholder tokens')
        prefix=len(context['token_ids'])-1
        wanted.extend([context['token_ids'][-1],*tokens]);positions.extend(range(prefix,prefix+4));seq.append(prefix+4)
    seen=[]
    for name,owners in(('target',list(range(16))),('reference',list(range(16,32)))):
        value=row[name]['metadata'];tables=value['block_tables']
        need(all(type(v)is int for key in('owners','inputs','positions','contexts','slots','sequence_lens','actual_seq_lengths_q')
                 for v in value[key])and all(len(table)==32 and all(type(v)is int for v in table)for table in tables),
             'actual integer FIA metadata and original32-page table width')
        need(value['owners']==owners and value['inputs']==wanted and value['positions']==positions
             and value['contexts']==[p+1 for p in positions]
             and value['sequence_lens']==seq and value['actual_seq_lengths_q']==list(range(4,65,4))
             and len(tables)==16 and value['request_block_tables']==tables,
             'all64 independently reconstructed actual token/position/length/table fields')
        expected=[]
        for i,table in enumerate(tables):
            for pos in positions[4*i:4*i+4]:
                page=table[pos//128]
                need(type(page)is int and 0<=page<544,'actual own page in original544 pool')
                expected.append(page*128+pos%128)
        need(value['slots']==expected and len(set(expected))==64,'exact own64 physical write slots')
        seen.append(set(expected))
    need(seen[0].isdisjoint(seen[1]),'candidate/reference query slots belong to separate histories')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--terminal-authorized',action='store_true',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    namespace=globals()
    for path,expected in PINS.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('pinned source changed: '+path)
    definitions(HELPERS,{'need','sha','desc','read','archive','Firewall','extracted','Value'},namespace)
    definitions(PRIOR,{'component_integer_verdicts'},namespace)
    sys.meta_path.insert(0,Firewall())
    sys.path[:0]=[str(P),str(ACTUAL_ROOT),str(B/'tree-normal-cap3-original-geometries-full-v207')]
    need(args.terminal_authorized and args.tag and '/'not in args.tag and args.tag not in('.','..'),'ROOT exact safe tag')
    receipt=ACTUAL_ROOT/f'receipt-{args.tag}.json'
    need(sha(receipt)==args.expected_receipt_sha256,'exact ROOT-authorized terminal receipt')
    r=read(receipt)
    need(r['schema']=='ROOT.V210.bounded-normal64-supervisor.v1'and r['status']=='failed'
         and len(r['errors'])==1 and "ModuleNotFoundError: No module named 'base207'" in r['errors'][0]
         and 'from result210 import audit' in r['errors'][0] and r['child']['exit_code']==0 and r['sources_unchanged']is True
         and r['owned_remaining']==[]and r['devices_after']['all_idle']is True,
         'complete actual current child, unchanged source and cleanup')
    need(r['action']=='bounded-normal-B3-causal64-diagnostic'and r['tag']==args.tag
         and r['production_permission']is r['full463']is r['formal_Goodput']is r['cost_measured']is False,
         'bounded normal64 scope, no performance/production qualification')
    need(all(r[k]['exit_code']==0 and r[k]['actual_result']['status']=='pass'for k in('pre_verifier','post_verifier')),
         'retained original actual pre/post verifier PASS')
    f=read(P/'FROZEN_ROOT210.json');sources={}
    for values in(f['source_files'],f['external_files'],f['retained_HOST_evidence'],
         {str(P/'CPU_RESULT210.json'):f['CPU_sha256'],str(P/'FROZEN_ROOT210.json'):PINS[str(P/'FROZEN_ROOT210.json')],
          f['root_review']['path']:f['root_review']['sha256']}):
        for path,digest in values.items():
            need(path not in sources or sources[path]==digest,'consistent source binding');sources[path]=digest
    need(len(sources)==1738 and all(sha(path)==digest for path,digest in sources.items()),'all1738 frozen source/evidence bytes')
    with tarfile.open(P/'PREPARED_SOURCE_ROOT210.tar.gz','r:gz')as saved:
        members=saved.getmembers()
        need(len(members)==len(sources)and{m.name for m in members}=={p.lstrip('/')for p in sources},'unique full source archive set')
        for member in members:
            need(member.isfile()and hashlib.sha256(saved.extractfile(member).read()).hexdigest()==sources['/'+member.name],
                 'exact archived source '+member.name)
    actual=ACTUAL_ROOT/'actual'/args.tag
    from result210 import audit
    original=audit(actual)
    need('actual_component_audit'not in r,'failed guardian did not invent a producer audit')
    need(read(actual/'completed.json')['audit']==original
         and read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'actual completion and worker exit')
    parent=component_integer_verdicts(actual)
    ht=types.SimpleNamespace(Tensor=Value,tensor=lambda values,**kw:Value(values),long='long')
    ns=dict(torch=ht,TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b))
    verify=extracted(NATIVE/'native_engine.py','_verify_tree_outputs_host',ns)
    pins={};calls=owners=0
    for rank in(1,2,3):
        for phase in range(3):
            for cycle in range(4):
                path=actual/'bounded64-B3-own-current-process'/f'rank{rank}-pass{phase}-cycle{cycle}.json.gz'
                row=read(path);pins[str(path)]=sha(path);metadata(row)
                heads=row['target']['greedy'][2]['full_physical_tokens'];need(len(heads)==64,'all64 physical target heads retained')
                need(all(type(v)is int and 0<=v<151936 for v in heads)
                     and all(type(v)is int and 0<=v<151936 for values in row['rows']for v in values),'exact integer vocabulary values')
                plans=[types.SimpleNamespace(candidate_budget=3,parent_indices=Value([-1,0,1]))for _ in range(16)]
                out=verify(row['rows'],plans,Value(heads),4,'explicit-HOST-values')
                tokens=[[v for v in values if v>=0]for values in out.token_ids.values]
                need(tokens==row['commit']['committed_tokens']
                     and out.accepted_node_indices.values==row['commit']['accepted_paths'],
                     'unchanged original Native B3 BONUS integer verifier on current64 heads')
                calls+=1;owners+=16
    need(calls==36 and owners==576,'all current target rank/trace/cycle integer verdicts')
    result=dict(schema='ROOT.V210.independent-terminal-bounded-HOST.v1',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed_bounded_normal64_independent_audit_after_HOST_loader_failure',
                receipt=desc(receipt),original_guardian_status_preserved='failed',
                recovery_scope='HOST import-path correction only; existing device outputs, no NPU rerun',producer_audit=original,original_B4_M80_integer=parent,
                new64_original_integer_verifier_calls=calls,new64_original_owner_verdicts=owners,
                new64_full_metadata_records=calls*2,independent_raw_pins=pins,
                NPU_used=False,Torch_imported=False,full463=False,Goodput=None,
                production_issuer=False,formal_Goodput=False,adopted=False,whole_workload_AR_qualified=False,
                limitations=['HOST recomputes integer/metadata/slots/counter arithmetic; unretained hidden/logit/KV tensors are not rerun.',
                             'No promoted-child/normal64 production origin, cost or complete workload result is established.',
                             'Original199/207 source reviews are donor provenance. No independent agent review of ROOT210 source is claimed.'])
    with args.output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=desc(args.output),status=result['status'])))


if __name__=='__main__':main()
