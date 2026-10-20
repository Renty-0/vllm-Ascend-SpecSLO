"""Original same-M target qualifier; independent full history and MC2 gates."""
from dataclasses import asdict
from scope154 import P151,require,target_shape
from reuse154 import functions
import source154 as epoch
functions(P151/'target151.py',globals())
