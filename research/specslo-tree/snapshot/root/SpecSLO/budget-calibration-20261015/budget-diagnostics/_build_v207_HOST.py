from pathlib import Path
import ast
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-normal-cap3-original-geometries-full-v207';D=B/'budget-diagnostics';OLD=B/'tree-high-normal-cap3-active32-full-v176/host176.py'
tree=ast.parse(OLD.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Checks')
helpers=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in('rhythm','states','test01_actual_policy_shared3_parent3_and_real_whole_owner_production_pack')]
text=ast.unparse(ast.Module(body=[ast.ClassDef(name='Tests',bases=[ast.Attribute(value=ast.Name(id='unittest',ctx=ast.Load()),attr='TestCase',ctx=ast.Load())],keywords=[],body=helpers,decorator_list=[])],type_ignores=[])).replace('shape176','shape207').replace('V176','V207')
head='''"""Five new source/control HOST checks, no old suite or tensor numerical proof."""
import ast,copy,json,math,struct,sys,types,unittest,tempfile,multiprocessing,os,datetime
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
from dataclasses import dataclass,field,replace,asdict
from enum import Enum
from collections.abc import Mapping,Sequence
from base207 import P,B,BASE,ACTIVE,ACTION,read,sha,checked,require,functions,parent_ready
DETAILS={}
'''
(P/'test_full207.py').write_text(head+text+'\n')
