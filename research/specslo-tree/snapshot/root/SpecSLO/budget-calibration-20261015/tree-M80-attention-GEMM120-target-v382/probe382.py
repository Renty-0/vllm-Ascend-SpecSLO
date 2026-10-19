"""Fresh693 target proofs and balanced costs of the persistent scoped adapter."""
def run(engine,native,stage,output,fixture):
    from hybrid382 import install,snapshot
    stage('ROOT382/install-scoped-GEMM120-only-for-physical80-tree',lambda:install(engine))
    from probe378 import run as original
    result=original(engine,native,stage,output,fixture)
    result['hybrid382_final']=stage('ROOT382/retain-actual-conditional-matrix-routing',lambda:snapshot(engine))
    return result
