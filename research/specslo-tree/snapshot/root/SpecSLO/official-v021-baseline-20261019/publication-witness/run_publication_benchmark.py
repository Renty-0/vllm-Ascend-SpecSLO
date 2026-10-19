"""Future private entry only; no models or inference imports on module import."""
import importlib,json,os,re,sys
from pathlib import Path
from publication_witness import P,PublicationWitness,install,pins,require

def main():
 data=pins();tag=os.environ.get('SPECSLO_PUBLICATION_WITNESS_TAG','')
 require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',tag)is not None,'fresh safe witness tag is required')
 root=P/'actual'/tag;root.parent.mkdir(exist_ok=True);root.mkdir(exist_ok=False)
 # The original ROOT launcher must supply the original official/plugin paths,
 # runtime and model arguments. This file does not install or query devices.
 baseline=importlib.import_module('run_baseline')
 require(Path(baseline.__file__).resolve()==Path(data['harness_path']),'exact original evidence harness')
 processor=importlib.import_module('vllm.v1.engine.output_processor')
 benchmark=baseline.benchmark;witness=PublicationWitness(root)
 binding=install(processor,benchmark,witness)
 with(root/'loader-binding.json').open('x')as stream:
  json.dump(dict(schema='official-publication-loader.v1',original_argv=sys.argv,source_pins=data,binding=binding,
   output_mode_model_sampling_scheduling_arrivals_MC2_unchanged=True,not_client_delivery=True),stream,indent=2);stream.write('\n')
 return baseline.main()
if __name__=='__main__':main()
