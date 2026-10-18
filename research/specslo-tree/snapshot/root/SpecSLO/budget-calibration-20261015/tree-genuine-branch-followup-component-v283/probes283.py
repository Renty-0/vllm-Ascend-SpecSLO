"""Fresh rawbyte gate then genuine postbranch deferred-repair controls."""
def run(engine,native,stage,output,fixture):
    from raw_probe277 import run as raw_probe
    raw=stage('ROOT283/rawbyte1024',lambda:raw_probe(engine)if engine.is_draft else None)
    from followup281 import run as followup
    from runtime283 import save
    result=followup(engine,native,stage,output,fixture,save=save)
    result['raw_copy']=raw
    return result
