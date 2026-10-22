"""Prepare private completed-output supervision; no backend/model/device import."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OUT=B/'target-rank-calibration-v1'
RAW=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/small-tail-role-aware-v209-full463-v3/windows/window32/full463/rank1-full463-actual-trace.json.gz'
AUDIT=D/'V209_V3_INDEPENDENT_TERMINAL_FULL463_AUDIT.json'
OLD=B/'tree-b4-joint-subtree-single-writer-diagnostic-v206/actual/truefork-single-writer-v206-bounded-gain-v1/joint-B4-current-diagnostic'
MODEL=Path('/root/data/shared-models/Qwen3-0.6B')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def desc(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def read(path):return json.loads(gzip.decompress(Path(path).read_bytes())if Path(path).suffix=='.gz'else Path(path).read_text())
def write(path,value):
    raw=json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    with Path(path).open('xb')as stream:stream.write(gzip.compress(raw,mtime=0)if Path(path).suffix=='.gz'else raw+b'\n')


def main():
    OUT.mkdir(exist_ok=False)
    assert sha(AUDIT)=='c4c8b3ccd6aa2b7a9be83c2520c6d75bf7cb756bc8c2bab39ac0887e123243da'
    audit=read(AUDIT);assert audit['status']=='passed_terminal_component_and_full463_audit'and audit['errors']==[]
    summary=read(D/'ROOT_V209_V3_CURRENT_FULL463_RESULT_AND_HANDOFF.json');assert sha(RAW)==summary['raw']['sha256']
    raw=read(RAW);outputs={r['request_id']:r for r in raw['generate_results']};records=[];groups={}
    category={'tight':'coding','normal':'chat','loose':'summarization'}
    assert len(raw['requests'])==len(raw['tokenized_inputs'])==len(outputs)==463
    for request,prompt in zip(raw['requests'],raw['tokenized_inputs'],strict=True):
        result=outputs[request['request_id']];tokens=result['completion_token_ids'];key=request['prompt_sha256']
        assert len(tokens)==256 and len(prompt)>=1 and len(prompt)+len(tokens)<=4096
        assert all(type(v)is int and 0<=v<151936 for v in[*prompt,*tokens])
        split='holdout'if int(key[:8],16)%5==0 else'train'
        assert key not in groups or groups[key]==split;groups[key]=split
        records.append(dict(request_id=request['request_id'],category=category[request['slo_class']],
            slo_class=request['slo_class'],prompt_group_sha256=key,split=split,
            input_token_ids=prompt+tokens[:-1],prediction_positions=list(range(len(prompt)-1,len(prompt)+255)),
            target_next_token_ids=tokens,prompt_tokens=len(prompt),target_tokens=256))
    dataset=OUT/'completed-output-teacherforce.json.gz';write(dataset,records)
    references=[];pins={}
    for path in sorted(OLD.glob('rank0-gain-case*.json.gz')):
        value=read(path)
        if value['case']['policy']!='old-primary-spine-B4-v1':continue
        pins[str(path)]=sha(path)
        work=value['work'];proof=work['draft_proof']
        for i,context in enumerate(work['source']['contexts']):
            if not context['measured_owner']:continue
            prefix=context['token_ids'];pool=proof['raw_pool'][i];probs=proof['node_confidences'][i]
            assert len(pool)==len(probs)==8
            references.append(dict(request_id=context['request_id'],category=context['category'],
                source_context_sha256=hashlib.sha256(json.dumps(prefix,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                input_token_ids=prefix+pool[:3],prediction_positions=list(range(len(prefix)-1,len(prefix)+3)),
                Native_top2_token_ids=[[pool[j],pool[j+4]]for j in range(4)],
                Native_top2_probabilities=[[probs[j],probs[j+4]]for j in range(4)],
                original_raw=desc(path),owner=i))
    assert len(references)==36 and len({r['source_context_sha256']for r in references})==36
    reference=OUT/'Native-draft-prefix-reference.json.gz';write(reference,references)
    config=read(MODEL/'config.json');assert config['model_type']=='qwen3'and config['vocab_size']==151936
    manifest=dict(schema='ROOT.target-rank-calibration.teacherforce-inputs.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='HOST_data_prepared_no_model',
        dataset=desc(dataset),Native_reference=desc(reference),completed_target_raw=desc(RAW),
        completed_target_independent_audit=desc(AUDIT),Native_reference_raw_pins=pins,
        requests=len(records),unique_prompt_groups=len(groups),labels=463*256,
        request_counts=dict(Counter(r['category']+'/'+r['split']for r in records)),
        group_counts=dict(Counter(groups.values())),
        model_path=str(MODEL),model_config=desc(MODEL/'config.json'),
        split_rule='SHA256 prompt group first32bits modulo5==0 holdout; every repeated prompt stays in one split',
        fixed_features=['category','draft-top1 probability','draft-top2 ratio'],
        probability_edges=[0,.5,.7,.85,.95,.99,1.000001],
        top2_ratio_edges=[0,.01,.05,.15,.30,.500001],
        prior_strength=4,minimum_training_bin_samples=32,hyperparameter_search=False,
        limitations=[
            'Supervision is the completed209 target output, not new target-model computation.',
            'Future teacher forcing uses HF eager BF16 draft; numerical correspondence to Native must be measured, not assumed.',
            'Holdout groups test this calibration within the original dataset; any later full463 run includes training groups and is not an independent generalization test.',
            'No g curve, production selector, NPU qualification, budget limit or Goodput is established by this dataset.',
        ],NPU_used=False,model_loaded=False,production_installed=False,Goodput=None)
    path=OUT/'DATA_MANIFEST.json';write(path,manifest)
    print(json.dumps(dict(manifest=desc(path),requests=len(records),groups=len(groups),Native_references=len(references),counts=manifest['request_counts'])))


if __name__=='__main__':main()
