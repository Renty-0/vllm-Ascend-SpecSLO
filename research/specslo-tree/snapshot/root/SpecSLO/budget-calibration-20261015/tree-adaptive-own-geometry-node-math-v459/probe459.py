"""Fresh all693 own-physical node cases and paired costs, no serving scope."""
from pathlib import Path
import hashlib,json
from source377 import function
import probe378 as original

def fingerprints(engine):
    import torch
    return {name:hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name,value in engine.model.named_parameters()}

def run(engine,native,stage,output,fixture):
    if engine.config.max_num_seqs!=64:
        raise RuntimeError('ROOT459 original global active ceiling64')
    before=stage('ROOT459/all-model-parameter-SHA-before',lambda:fingerprints(engine))
    pointers=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]
    manifest=engine.mc2_static_route_manifest
    digest=None if manifest is None else manifest.digest
    if not engine.is_draft and manifest.profile_sha256!='67feca4c924ce95b252812d2db606a65a43942c472c429a71d12e5270442deb7':
        raise RuntimeError('ROOT459 actual original Native canonical profile digest')
    body=function(original.run,(
        ('from oracle377 import run as neural, cases','from oracle459 import run as neural, cases'),
        ('from benchmark378 import run as costs','from benchmark459 import run as costs'),
        ('independent_reference_FULL120=True,','independent_reference_FULL120=False,independent_reference_same_physical_M=True,'),
    ))
    result=body(engine,native,stage,output,fixture)
    after=stage('ROOT459/all-model-parameter-SHA-after',lambda:fingerprints(engine))
    current=engine.mc2_static_route_manifest
    if before!=after or pointers!=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]or (None if current is None else current.digest)!=digest:
        raise RuntimeError('ROOT459 unchanged real weights, addresses and original MC2 route binding')
    value=dict(rank=engine.rank,parameter_SHA_before=before,parameter_SHA_after=after,
               original_MC2_manifest_digest=digest,actual_normalized_profile_sha256=None if manifest is None else manifest.profile_sha256,
               global_active_ceiling=64,parameter_addresses_unchanged=True,Native_flags_assigned=False)
    with(Path(output)/f'rank{engine.rank}-weights459-before-after.json').open('x')as f:
        json.dump(value,f,indent=2);f.write('\n')
    return dict(result,all_model_parameter_SHA_unchanged=True,global_active_ceiling=64,
                original_cross_M120_failures_preserved=True,Native_flags_assigned=False)
