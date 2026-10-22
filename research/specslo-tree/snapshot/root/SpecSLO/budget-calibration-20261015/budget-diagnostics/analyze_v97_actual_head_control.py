"""Independent pure-HOST audit of terminal V97 raw BF16 head-control evidence."""
import argparse
import array
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tarfile
import zipfile

W=Path('/root/SpecSLO/budget-calibration-20261015/tree-head-control-v97')
D=Path(__file__).resolve().parent
FREEZE='d51537421c40d029b859db54c0dc09e4232156de50e5c1a1bea10ad0ee7998f4'
CPU_ARCHIVE='3117fb3dc45c65fb1ef0bf24d16a3aea27ff1fe7315b19acc5b3fba32d237f1a'
INPUTS='0d7b6c4d8335d021444c4113097cb535b0840ca397fa39fc450658d227e5b699'
FLAGS=('full_model_loaded','distributed_initialized','original_TP_hidden_equality_measured',
    'original_target_graph_replayed','KV_measured','numerical_qualified','online_qualified',
    'full48_AR','all65_qualified','full463_qualified','formal_Goodput')


def need(value,message):
    if not value:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def pointer(row, ledger, within=None):
    p=Path(row['path'])
    if within is not None:need(p.resolve().is_relative_to(Path(within).resolve()),'artifact outside owned directory')
    actual=dict(bytes=p.stat().st_size,sha256=digest(p))
    need(actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'artifact SHA/size mismatch: '+str(p))
    ledger[str(p)]=actual
    return p


def bf16(path,count):
    payload=Path(path).read_bytes();need(len(payload)==count*2,'incorrect BF16 row length')
    bits=array.array('H');bits.frombytes(payload)
    if sys.byteorder!='little':bits.byteswap()
    values=[struct.unpack('<f',struct.pack('<I',int(x)<<16))[0] for x in bits]
    need(all(math.isfinite(v) for v in values),'nonfinite actual vocabulary logits')
    return values


def vocabulary(values):
    need(len(values)==151936,'complete valid common vocabulary')
    maximum=max(values)
    # Native .max(dim=-1), then gathered TP values.argmax: first token and first shard.
    winner=values.index(maximum)
    top=sorted(range(len(values)),key=lambda token:(-values[token],token))[:2]
    return dict(top2=[dict(token_id=t,score=values[t]) for t in top],native_greedy_token=winner,
        global_max_tied_token_ids=[t for t,v in enumerate(values) if v==maximum],
        pair_scores={str(t):values[t] for t in (11,30)},vocabulary_size=len(values))


def difference(a,b):
    need(len(a)==len(b),'aligned numerical vectors')
    delta=[x-y for x,y in zip(a,b)]
    return dict(exact_equal=a==b,differing_values=sum(x!=0 for x in delta),
        max_abs=max(abs(x) for x in delta),rms=math.sqrt(sum(x*x for x in delta)/len(delta)))


def checkpoint(plan):
    cp=plan['checkpoint'];p=Path(cp['path']);need(p.stat().st_size==cp['bytes'],'checkpoint size')
    with p.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0];need(size<10*1024**2,'bounded checkpoint header')
        raw=stream.read(size);header=json.loads(raw)
    need(hashlib.sha256(struct.pack('<Q',size)+raw).hexdigest()==cp['header_sha256'],'checkpoint header SHA')
    need(size+8==cp['data_start'] and header['lm_head.weight']==cp['head_metadata'],'actual head header')
    need(header['lm_head.weight']['dtype']=='BF16' and header['lm_head.weight']['shape']==[151936,5120],'original BF16 head shape')
    whole=hashlib.sha256();ranges=[hashlib.sha256() for _ in range(3)]
    boundaries=[]
    for rank,s in enumerate(plan['shards']):
        lo=size+8+header['lm_head.weight']['data_offsets'][0]+s['start']*5120*2
        hi=lo+(s['valid_end']-s['start'])*5120*2
        need(plan['shard_data'][rank]['absolute_offset']==lo and plan['shard_data'][rank]['source_bytes']==hi-lo,'source slice byte range')
        boundaries.append((lo,hi))
    offset=0
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            whole.update(block)
            for h,(lo,hi) in zip(ranges,boundaries):
                left=max(lo,offset)-offset;right=min(hi,offset+len(block))-offset
                if left<right:h.update(block[left:right])
            offset+=len(block)
    need(whole.hexdigest()==cp['sha256'],'public full checkpoint SHA changed')
    output=[]
    for rank,h in enumerate(ranges):
        original=h.hexdigest();need(original==plan['shard_data'][rank]['source_slice_sha256'],'original source slice SHA')
        h.update(b'\0'*(plan['shards'][rank]['zero_padding_rows']*5120*2));padded=h.hexdigest()
        need(padded==plan['shard_data'][rank]['padded_shard_sha256'],'padded shard SHA')
        output.append(dict(tp_rank=rank,source_slice_sha256=original,padded_shard_sha256=padded))
    return dict(path=str(p),bytes=cp['bytes'],sha256=whole.hexdigest(),shards=output)


def archive_members(path,expected):
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers();need(len(members)==len(expected),'source archive member count')
        need(len({m.name for m in members})==len(members),'duplicate archive member')
        need({m.name for m in members}==set(expected),'source archive source closure')
        for member in members:
            need(member.isfile(),'nonregular source snapshot member')
            need(hashlib.sha256(archive.extractfile(member).read()).hexdigest()==expected[member.name],
                 'archived source bytes differ: '+member.name)


def audit(tag):
    need(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9-]{0,79}',tag),'safe tag')
    actual=W/'actual'/tag;receipt=read(actual/'receipt.json')
    need(receipt['status'] in ('complete','failed'),'actual receipt must be terminal; refusing running audit')
    output=dict(schema='v97-independent-actual-head-control-audit-v1',tag=tag,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_job_status=receipt['status'],
        original_job_status_modified=False,original_receipt=dict(path=str(actual/'receipt.json'),sha256=digest(actual/'receipt.json')),
        analyzer=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),
        execution=dict(argv=sys.argv,python_executable=sys.executable,python_version=sys.version,
            used_retained_runtime_wrapper_for_this_HOST_audit=False,
            SPECSLO_ENV_CLEANED=os.environ.get('SPECSLO_ENV_CLEANED'),
            ASCEND_RT_VISIBLE_DEVICES=os.environ.get('ASCEND_RT_VISIBLE_DEVICES')),
        NPU_initialized=False,device_or_model_commands_executed=False,
        numerical_qualified=False,root_cause_inferred=False,formal_Goodput=False,
        original_TP_hidden_equality_measured=False,original_target_graph_replayed=False,KV_measured=False,
        all65_qualified=False,full48_AR=False,full463_qualified=False,online_qualified=False,
        artifacts={},checks=[],errors=[])
    ledger=output['artifacts']
    def check(name,callback):
        try:
            result=callback();output['checks'].append(dict(name=name,passed=True));return result
        except Exception as error:
            output['checks'].append(dict(name=name,passed=False));output['errors'].append(dict(check=name,error=repr(error)));return None
    def frozen():
        need(digest(W/'SOURCE_MANIFEST_V97.json')==FREEZE,'frozen source manifest identity')
        freeze=read(W/'SOURCE_MANIFEST_V97.json')
        need(freeze['schema']=='v97-frozen-head-control-source-v1' and len(freeze['files'])==10,'exact static source count')
        for name,wanted in freeze['files'].items():need(digest(W/name)==wanted,'frozen local source changed: '+name)
        need(digest(W/'INPUTS_V97.json')==INPUTS,'reviewed input plan identity')
        cp=read(W/'CPU_RESULT_V97.json');need(cp['status']=='passed' and cp['tests']==15
            and cp['NPU_initialized'] is False and cp['NPU_or_model_launch_attempts']==0
            and cp['torch_or_model_dependency_imported'] is False,'actual retained CPU evidence')
        need('3.11' in cp['python_version'],'retained CPU interpreter identity')
        host=W/'host-archive/V97_FINAL_CPU_SOURCE.tar.gz';need(digest(host)==CPU_ARCHIVE,'final CPU archive identity')
        archive_members(host,{**freeze['files'],'SOURCE_MANIFEST_V97.json':FREEZE})
        need(receipt['source_before']==freeze['files']==receipt['source_after'],'actual pre/post local source identity')
        output['freeze']=dict(sha256=FREEZE,CPU_tests=15,CPU_archive_sha256=CPU_ARCHIVE)
        return freeze
    freeze=check('CPU/frozen source and archive',frozen)
    plan=read(W/'INPUTS_V97.json')
    def inputs():
        for path,wanted in plan['source_pins'].items():need(digest(path)==wanted,'external source changed: '+path)
        for ptr in [plan['fixture'],*plan['hidden'].values(),*plan['model_metadata']]:pointer(ptr,ledger)
        fixture=read(plan['fixture']['path'])
        need(fixture['completion_first_difference_index']==85 and fixture['root_token_id']==1852
            and fixture['root_logical_position']==170 and len(fixture['prompt_tokens'])==86
            and len(fixture['shared_output_prefix_tokens'])==85,'actual case0 source identity')
        for name,ptr in plan['hidden'].items():
            with zipfile.ZipFile(ptr['path']) as archive:
                storages=[n for n in archive.namelist() if '/data/' in n and n.rsplit('/',1)[-1].isdigit()]
                need(len(storages)==1,'single actual retained hidden storage');raw=archive.read(storages[0])
            need(len(raw)==10240 and hashlib.sha256(raw).hexdigest()==ptr['tensor_sha256'],'actual BF16 hidden bytes')
        actualpad={r:{s:e[s+'_projection'][0]['head_token_pad_multiple'] for s in ('candidate','reference')}
            for r,e in fixture['rank_local_actual_evidence'].items()}
        need(actualpad=={str(r):dict(candidate=1,reference=1) for r in (1,2,3)}
            ==plan['head_rule_evidence']['actual_capture_pad_multiples'],'original actual pad values')
        need(all(plan[k] is False for k in FLAGS),'input qualification scope')
        return fixture
    fixture=check('public source, actual hidden and controlled input binding',inputs)
    cp=check('full checkpoint and actual three padded BF16 shard hashes',lambda:checkpoint(plan))
    if cp:output['checkpoint']=cp
    def terminal():
        need(receipt['status']=='complete' and receipt['errors']==[],'actual job did not complete')
        need(receipt['child']['exit_code']==0 and receipt['local_head_control_complete'] is True,'actual child completion')
        need(receipt['cleanup']['owned_remaining']==[],'owned cleanup incomplete')
        need(receipt['devices_before']['all_idle'] is True and receipt['devices_after']['all_idle'] is True,'pre/post idle evidence')
        for label in ('pre','post'):
            verifier=receipt[label+'_verifier'];need(verifier['exit_code']==0,'retained lock verifier failed')
            pointer(verifier['log'],ledger,actual)
        need(receipt['full_checkpoint_before']==receipt['full_checkpoint_after']==plan['checkpoint']['sha256'],'actual pre/post checkpoint identity')
        need(receipt['private_bytes']<64*1024**2 and receipt['elapsed_seconds']<600,'actual bounded evidence/time')
        snap=pointer(receipt['source_snapshot'],ledger,actual)
        sources={str(W/name):wanted for name,wanted in receipt['source_before'].items()}
        sources[str(W/'SOURCE_MANIFEST_V97.json')]=FREEZE;sources.update(plan['source_pins'])
        for ptr in [plan['fixture'],*plan['hidden'].values(),*plan['model_metadata']]:sources[ptr['path']]=ptr['sha256']
        expected={('root/'+str(Path(p).relative_to('/root')) if Path(p).is_relative_to('/root') else 'external/'+p.lstrip('/')):v for p,v in sources.items()}
        archive_members(snap,expected)
        pointer(receipt['actual_summary'],ledger,actual)
        return dict(elapsed_seconds=receipt['elapsed_seconds'],private_bytes=receipt['private_bytes'],owned_remaining=[],
            pre_verifier_exit=0,post_verifier_exit=0,pre_post_idle=True,source_archive_members=len(expected))
    finished=check('actual terminal, all source snapshot bytes, pre/post verifier and owned cleanup',terminal)
    if finished:output['terminal']=finished
    def rows():
        reports=[read(actual/'worker-output'/f'shard{r}.json') for r in range(3)]
        keys={(s,m,repeat) for s in ('candidate','reference') for m in (1,96) for repeat in range(3)}
        compiled=ast.parse(Path(plan['native_model_path']).read_text())
        helper=next(n for n in compiled.body if isinstance(n,ast.FunctionDef) and n.name=='_pad_token_rows')
        cls=next(n for n in compiled.body if isinstance(n,ast.ClassDef) and n.name=='NativeLMHead')
        project=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_project_local')
        projectsha=hashlib.sha256((ast.unparse(ast.fix_missing_locations(ast.Module(body=[helper,project],type_ignores=[])))+'\n').encode()).hexdigest()
        raw={};allvectors={}
        for rank,report in enumerate(reports):
            need(report['tp_rank']==rank and report['status']=='complete' and report['errors']==[],'complete actual worker identity')
            need(report['physical_card']==[5,6,7][rank] and report['logical_device']==rank+1
                and report['actual_current_device']==rank+1,'actual target mapping')
            need(report['NPU_initialized'] is True and all(report[k] is False for k in FLAGS),'actual head-only scope')
            need(report['controlled_input']==plan['controlled_input'] and report['projection_source_sha256']==projectsha,'actual unchanged implementation/input')
            need(report['NativeLMHead_constructed'] is False and report['NativeWeightLoader_class_used'] is False
                and report['original_full_weight_loader_called'] is False and report['weight_quantized'] is False,'truthful loader/head API')
            need(report['weight_dtype']=='torch.bfloat16' and report['weight_shape']==[50646,5120],'original BF16 local head geometry')
            need(report['shard']==plan['shards'][rank],'exact vocabulary shard')
            data=plan['shard_data'][rank]
            need(report['checkpoint_slice_sha256']==data['source_slice_sha256']
                and report['loaded_padded_shard_sha256']==report['device_weight_readback_sha256']==data['padded_shard_sha256'],'loaded/device weight bytes')
            env=report['runtime']['env'];need(env['SPECSLO_ENV_CLEANED']=='1' and env['ASCEND_RT_VISIBLE_DEVICES']=='0,1,2,3'
                and env['VLLM_ASCEND_PEARL_TARGET_TOKEN_PAD_MULTIPLE']=='1' and env['VLLM_ASCEND_PEARL_TARGET_NZ_MODE']=='14','actual retained worker environment')
            need(report['runtime']['versions']==plan['runtime_versions'] and report['runtime']['python'].startswith('3.11'),'actual retained worker versions')
            need(len(report['observations'])==12 and {(r['source'],r['M'],r['repeat']) for r in report['observations']}==keys,'actual exact 12 observations')
            for obs in report['observations']:
                key=(obs['source'],obs['M'],obs['repeat'])
                need(obs['condition']==key[0]+'-M'+str(key[1]) and obs['finite'] is True
                    and obs['full_projection_equal_first_repeat'] is True and obs['real_row']==0
                    and obs['remaining_rows']=='explicit_zero_BF16_fill','actual finite/repeat/controlled row')
                need(obs['hidden_input_sha256']==plan['hidden'][key[0]]['tensor_sha256'],'actual input tensor SHA')
                p=pointer(obs['logits'],ledger,actual/'worker-output')
                payload=p.read_bytes();raw[(rank,*key)]=payload
                values=bf16(p,plan['shards'][rank]['valid_end']-plan['shards'][rank]['start'])
                allvectors.setdefault(key,[]).extend(values)
        results=[]
        for source in ('candidate','reference'):
            for m in (1,96):
                for rank in range(3):need(raw[(rank,source,m,0)]==raw[(rank,source,m,1)]==raw[(rank,source,m,2)],'raw full local vocabulary repeat changed')
                for repeat in range(3):results.append(dict(condition=source+'-M'+str(m),source=source,M=m,repeat=repeat,**vocabulary(allvectors[(source,m,repeat)])))
        summary=read(actual/'worker-output'/'summary.json')
        need(summary['status']=='complete' and len(summary['results'])==12 and all(summary[k] is False for k in FLAGS),'stored summary scope')
        for result in results:
            stored=[r for r in summary['results'] if (r['source'],r['M'],r['repeat'])==(result['source'],result['M'],result['repeat'])]
            need(len(stored)==1 and all(stored[0][k]==v for k,v in result.items()),'independent full vocabulary result differs from stored summary')
        same=[]
        for source in ('candidate','reference'):
            delta=difference(allvectors[(source,1,0)],allvectors[(source,96,0)])
            same.append(dict(source=source,**delta))
            saved=next(r for r in summary['same_hidden_M1_vs_M96'] if r['source']==source)
            need(saved['M1_vs_M96_full_vocabulary_row_exact_equal']==delta['exact_equal'],'stored M sensitivity comparison')
        source_delta=[dict(M=m,**difference(allvectors[('candidate',m,0)],allvectors[('reference',m,0)])) for m in (1,96)]
        output['actual_findings']=dict(independent_workers=3,conditions=4,repeats=3,raw_logit_artifacts=36,
            valid_vocabulary=151936,full_vocabulary_repeats_exact=True,
            cells=[r for r in results if r['repeat']==0],same_hidden_M1_vs_M96=same,candidate_vs_reference_hidden_projection=source_delta,
            controlled_input=plan['controlled_input'],original_96_row_hidden_matrix_recreated=False,
            head_direct_execution_matches_original_full_target_graph_claimed=False)
        output['actual_worker_runtime']=dict(python=[r['runtime']['python'] for r in reports],
            used_retained_runtime_wrapper=True,versions=reports[0]['runtime']['versions'],
            physical_cards=[r['physical_card'] for r in reports],weight_formats=[r['weight_format'] for r in reports])
        if fixture:
            original=fixture['same_prefix_numerics_from_independent_recovery_audit'][0]['dynamic_actual_competitors']
            expected={s:{str(r['token_id']):r['score'] for r in original[s+'_scores']} for s in ('candidate','reference')}
            output['actual_findings']['matches_V88_original_observed_11_30_scores']=[dict(condition=r['condition'],
                matches=r['pair_scores']==expected[r['source']]) for r in output['actual_findings']['cells']]
    check('independent raw BF16 full vocabulary, weights, all 36 row hashes/ties/repeats and stored summary',rows)
    output['evidence_unchanged_after_analysis']=dict(source_manifest=digest(W/'SOURCE_MANIFEST_V97.json')==FREEZE,
        original_receipt=digest(actual/'receipt.json')==output['original_receipt']['sha256'])
    need(all(output['evidence_unchanged_after_analysis'].values()),'input evidence mutated during HOST audit')
    need(not any(name.split('.')[0] in ('torch','torch_npu','safetensors','vllm','transformers') for name in sys.modules),'HOST audit imported device/model package')
    output['all_checks_passed']=not output['errors']
    output['status']='audited-terminal-head-control' if output['all_checks_passed'] else 'audited-terminal-partial-or-invalid-evidence'
    return output


def self_test():
    a=[0.]*151936;a[11]=35.25;a[30]=35.25
    result=vocabulary(a);need(result['native_greedy_token']==11 and result['global_max_tied_token_ids']==[11,30],'tie self-test')
    b=list(a);b[30]=35.5;need(vocabulary(b)['native_greedy_token']==30,'winner self-test')
    need(difference(a,b)['differing_values']==1,'difference self-test')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();print(json.dumps(dict(self_test='passed',NPU_initialized=False)));return 0
    need(args.tag is not None,'explicit terminal tag')
    destination=D/('ROOT_V97_ACTUAL_HEAD_CONTROL_'+args.tag+'_AUDIT.json');need(not destination.exists(),'refuse existing audit output')
    result=audit(args.tag)
    with destination.open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],all_checks_passed=result['all_checks_passed'],errors=result['errors'],
        output=str(destination),audit_sha256=digest(destination),actual_findings=result.get('actual_findings'))))
    return int(not result['all_checks_passed'])


if __name__=='__main__':raise SystemExit(main())
