"""Fresh532 independent target math and paired cost with actual165 MC2."""
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
B=P.parent
sys.path[:0]=[str(B/'tree-M160-C128-capacity-binding-v329'),str(B/'tree-M160-actual165-MC2-bridge-host-v330')]
import runtime329 as previous
base_worker=previous.worker329

def worker331(*args,**kwargs):
    from check331 import ready
    ready()
    import probes326
    base_run=probes326.run
    def with_bridge(engine,native,stage,output,fixture):
        bridge=None
        if not engine.is_draft:
            from vllm_ascend.spec_decode.pearl import native_model
            from bridge330 import install
            bridge=install(engine.model,native_model)
            previous.original.save(output,f'rank{engine.rank}-bridge330-before-tree.json',dict(
                rank=engine.rank,installed_before_diagnostic_tree=True,
                physical_FIA_rows160=True,actual_projection165=True,
                original_profile_sha256=engine.mc2_static_route_manifest.profile_sha256,
                original_source_functions_retained=True,production_qualified=False))
        report=base_run(engine,native,stage,output,fixture)
        report['actual165_MC2_bridge330']=bridge.summary()if bridge else None
        return report
    probes326.run=with_bridge
    try:
        return base_worker(*args,**kwargs)
    finally:
        probes326.run=base_run

previous.original.worker=worker331
if __name__=='__main__':previous.original.main()
