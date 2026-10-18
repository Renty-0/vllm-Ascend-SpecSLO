"""Only two fresh constructor fields differ from the immutable parent."""
from dataclasses import replace
from base190 import require

def make_config():
 from startup_config_v132 import make_config as original
 value=original()
 require(value.spec_rhythm_prefill_coalesce_min_requests==1 and value.spec_rhythm_prefill_coalesce_max_wait_ms==0,'exact parent1/0 config')
 return replace(value,spec_rhythm_prefill_coalesce_min_requests=4,spec_rhythm_prefill_coalesce_max_wait_ms=150.0)
